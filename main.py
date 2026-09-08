import json
import os
import traceback
from typing import AsyncGenerator, List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anthropic import AsyncAnthropic, APIError

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Provide API key explicitly or ensure env var is set
client = AsyncAnthropic()

# 2. Local Tool
def get_stock_price(ticker: str) -> str:
    prices = {"AAPL": "185.50 USD", "GOOGL": "175.20 USD", "NVDA": "120.40 USD"}
    price = prices.get(ticker.upper(), "Ticker not found")
    return json.dumps({"ticker": ticker, "price": price})

TOOL_FUNCTIONS = {"get_stock_price": get_stock_price}

TOOLS_SCHEMA = [
    {
        "name": "get_stock_price",
        "description": "Retrieves current real-time stock price for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock symbol (e.g. NVDA)"}
            },
            "required": ["ticker"]
        }
    }
]

class ChatRequest(BaseModel):
    prompt: str
    history: List[Dict[str, Any]] = []

def format_sse(event_type: str, data: Dict[str, Any]) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"

async def agent_stream_generator(prompt: str, history: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    try:
        messages = list(history)
        messages.append({"role": "user", "content": prompt})

        # --- FIRST PASS: Stream Claude's initial turn ---
        async with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=TOOLS_SCHEMA,
            messages=messages,
        ) as stream:
            async for event in stream:
                if event.type == "text":
                    yield format_sse("text_delta", {"text": event.text})
            
            # Retrieve final message snapshot safely after stream completes
            final_message = await stream.get_final_message()

        messages.append({"role": "assistant", "content": final_message.content})

        # --- SECOND PASS: Tool Handling ---
        if final_message.stop_reason == "tool_use":
            tool_results_content = []

            for block in final_message.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_args = block.input
                    tool_id = block.id

                    yield format_sse("tool_start", {"tool": tool_name, "args": tool_args, "id": tool_id})

                    func = TOOL_FUNCTIONS.get(tool_name)
                    raw_result = func(**tool_args) if func else json.dumps({"error": "Tool not found"})
                    
                    parsed_result = json.loads(raw_result)
                    yield format_sse("tool_result", {"tool": tool_name, "result": parsed_result, "id": tool_id})

                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": raw_result
                    })

            messages.append({"role": "user", "content": tool_results_content})

            # Stream Claude's post-tool answer
            # My addition: this feeds the tool_results back into the model for a final response -
            # this is important for the model to generate a coherent final answer after tool usage.
            async with client.messages.stream(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                tools=TOOLS_SCHEMA,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type == "text":
                        yield format_sse("text_delta", {"text": event.text})

        yield format_sse("done", {})

    except APIError as e:
        # Catches Anthropic Auth/Rate Limit/Invalid Request errors safely
        yield format_sse("error", {"message": f"Anthropic API Error: {e.message}"})
    except Exception as e:
        # Prevents stream crash on generic Python exceptions
        print("STREAM EXCEPTION TRACEBACK:")
        traceback.print_exc()
        yield format_sse("error", {"message": f"Internal Error: {str(e)}"})

@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        agent_stream_generator(request.prompt, request.history),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)