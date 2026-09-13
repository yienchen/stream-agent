import os
import json
from typing import Dict, Any, List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from anthropic import AsyncAnthropic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DISCOVERY & RUNTIME INFO (Resolves 404 on load) ---
@app.get("/info")
@app.get("/api/info")
@app.get("/api/chat/stream/info")
async def get_runtime_info():
    return JSONResponse(
        content={
            "version": "1.0.0",
            "agents": {
                "default": {
                    "name": "default",
                    "description": "AG-UI Stream Agent"
                }
            }
        }
    )

# --- AGENT STREAM EXECUTION ---
@app.post("/api/agents/{agent_id}/run")
@app.post("/api/{agent_id}/run")
@app.post("/api/chat/stream")
async def run_agent(request: Request, agent_id: str = "default"):
    body = await request.json()
    raw_messages = body.get("messages", [])
    prompt = body.get("prompt", "")

    # Execute your streaming generator here
    return StreamingResponse(
        agent_stream_generator(prompt, raw_messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

anthropic_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL_NAME = "claude-sonnet-4-5"

TOOLS_SCHEMA = [
    {
        "name": "get_stock_price",
        "description": "Retrieves real-time stock price for a given ticker symbol.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock symbol (e.g., AAPL, NVDA)"}
            },
            "required": ["ticker"]
        }
    }
]

def execute_stock_tool(ticker: str) -> Dict[str, str]:
    prices = {"NVDA": "$135.50", "AAPL": "$224.30", "MSFT": "$448.90"}
    clean_ticker = ticker.upper()
    return {
        "ticker": clean_ticker,
        "price": prices.get(clean_ticker, "$180.00")
    }

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    body = await request.json()
    prompt = body.get("prompt", "")
    history = body.get("messages", [])

    formatted_messages = []
    for m in history:
        role = m.get("role", "user")
        content = m.get("content", "")
        if content:
            formatted_messages.append({"role": role, "content": content})

    if not formatted_messages or formatted_messages[-1]["content"] != prompt:
        formatted_messages.append({"role": "user", "content": prompt})

    response = await anthropic_client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        messages=formatted_messages,
        tools=TOOLS_SCHEMA,
    )

    stock_data = None
    response_text = ""

    for block in response.content:
        if block.type == "text":
            response_text += block.text
        elif block.type == "tool_use" and block.name == "get_stock_price":
            ticker = block.input.get("ticker", "AAPL")
            stock_data = execute_stock_tool(ticker)
            
            # Followup request to summarize with tool output
            formatted_messages.append({"role": "assistant", "content": response.content})
            formatted_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(stock_data)
                }]
            })

            followup = await anthropic_client.messages.create(
                model=MODEL_NAME,
                max_tokens=1024,
                messages=formatted_messages
            )
            for f_block in followup.content:
                if f_block.type == "text":
                    response_text += f_block.text

    return JSONResponse({
        "text": response_text,
        "stock_data": stock_data
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)