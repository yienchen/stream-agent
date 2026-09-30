"""
Weather agent as a plain AG-UI endpoint, with a real agent loop.

Contract:
  POST /agent  <- RunAgentInput JSON (camelCase)
  ->  text/event-stream of AG-UI events, starting with RUN_STARTED
      and ending with RUN_FINISHED or RUN_ERROR.

The agent loop, all inside one run:
  1. Call Claude with the conversation so far, streaming events out.
  2. If Claude stopped to use tools (stop_reason == "tool_use"): run them,
     emit TOOL_CALL_RESULT events, append Claude's turn plus a user turn
     of tool_result blocks to the conversation, and go back to step 1.
  3. Otherwise Claude has given its final answer: emit RUN_FINISHED.

No /info, no agent registry. Discovery is the CopilotKit *runtime's* job.

    pip install ag-ui-protocol fastapi uvicorn anthropic
    uvicorn agui_server:app --host 127.0.0.1 --port 8008 --reload
"""

import asyncio
import json
import logging
import uuid

from anthropic import AsyncAnthropic
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ag_ui.core import (
    EventType,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder

logger = logging.getLogger("weatheragent")
logging.basicConfig(level=logging.INFO)

MODEL = "claude-sonnet-4-5"
MAX_STEPS = 8  # safety cap: max Claude calls per run, so a loop can't run away
SYSTEM_PROMPT = (
    "You are a helpful weather assistant. Use the get_weather tool to look up "
    "current conditions, then answer the user in one or two friendly sentences."
)

app = FastAPI()

# Only needed if a browser talks to this server directly. Behind the Node
# runtime the request is server-to-server and CORS is irrelevant.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anthropic_client = AsyncAnthropic()


# --------------------------------------------------------------------------
# Tools. To add one: write an async handler, add its schema to TOOLS, and
# register the handler in TOOL_HANDLERS under the same name.
# --------------------------------------------------------------------------

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Fetch current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


async def get_weather(location: str) -> dict:
    # Keys must match the props WeatherCard destructures on the frontend.
    return {
        "location": location,
        "temperature": "72F",
        "condition": "Sunny",
        "humidity": "48%",
    }


TOOLS = [WEATHER_TOOL]
TOOL_HANDLERS = {"get_weather": get_weather}


async def run_tool(block) -> tuple[str, bool]:
    """Run one tool_use block. Returns (result_json_string, is_error).

    Errors are returned to Claude as an error result instead of raised, so
    the model can see what went wrong and recover or explain it.
    """
    handler = TOOL_HANDLERS.get(block.name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {block.name}"}), True
    try:
        return json.dumps(await handler(**block.input)), False
    except Exception as exc:
        logger.exception("tool %s failed", block.name)
        return json.dumps({"error": str(exc)}), True


def to_anthropic_messages(input_data: RunAgentInput) -> list[dict]:
    """Map AG-UI messages onto the Anthropic messages shape.

    Only user/assistant text is carried over. Tool calls and tool results
    from earlier runs are dropped, so on a follow-up turn Claude sees what
    was said but not the raw tool traffic.
    """
    out: list[dict] = []
    for m in input_data.messages:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out or [{"role": "user", "content": "Hello"}]


@app.post("/agent")
async def agent_endpoint(input_data: RunAgentInput, request: Request):
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def event_generator():
        yield encoder.encode(
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id,
            )
        )

        try:
            messages = to_anthropic_messages(input_data)

            # ---------------- the agent loop ----------------
            for _step in range(MAX_STEPS):
                # Anthropic identifies content blocks by index, AG-UI by id.
                # These maps translate between the two within one step.
                text_ids: dict[int, str] = {}
                tool_call_ids: dict[int, str] = {}
                parent_message_id = str(uuid.uuid4())

                async with anthropic_client.messages.stream(
                    model=MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                ) as stream:
                    async for event in stream:
                        if event.type == "content_block_start":
                            block = event.content_block
                            if block.type == "text":
                                message_id = str(uuid.uuid4())
                                text_ids[event.index] = message_id
                                parent_message_id = message_id
                                yield encoder.encode(
                                    TextMessageStartEvent(
                                        type=EventType.TEXT_MESSAGE_START,
                                        message_id=message_id,
                                        role="assistant",
                                    )
                                )
                            elif block.type == "tool_use":
                                tool_call_ids[event.index] = block.id
                                yield encoder.encode(
                                    ToolCallStartEvent(
                                        type=EventType.TOOL_CALL_START,
                                        tool_call_id=block.id,
                                        tool_call_name=block.name,
                                        parent_message_id=parent_message_id,
                                    )
                                )

                        elif event.type == "content_block_delta":
                            delta = event.delta
                            if (
                                delta.type == "text_delta"
                                and delta.text
                                and event.index in text_ids
                            ):
                                yield encoder.encode(
                                    TextMessageContentEvent(
                                        type=EventType.TEXT_MESSAGE_CONTENT,
                                        message_id=text_ids[event.index],
                                        delta=delta.text,
                                    )
                                )
                            elif (
                                delta.type == "input_json_delta"
                                and delta.partial_json
                                and event.index in tool_call_ids
                            ):
                                yield encoder.encode(
                                    ToolCallArgsEvent(
                                        type=EventType.TOOL_CALL_ARGS,
                                        tool_call_id=tool_call_ids[event.index],
                                        delta=delta.partial_json,
                                    )
                                )

                        elif event.type == "content_block_stop":
                            # Close each text message / tool call as its
                            # block ends, so events never interleave.
                            if event.index in text_ids:
                                yield encoder.encode(
                                    TextMessageEndEvent(
                                        type=EventType.TEXT_MESSAGE_END,
                                        message_id=text_ids[event.index],
                                    )
                                )
                            elif event.index in tool_call_ids:
                                yield encoder.encode(
                                    ToolCallEndEvent(
                                        type=EventType.TOOL_CALL_END,
                                        tool_call_id=tool_call_ids[event.index],
                                    )
                                )

                    final = await stream.get_final_message()
                    print(f"[step] stop_reason={final.stop_reason!r} blocks={[b.type for b in final.content]}")

                # Did Claude ask for tools, or is this its final answer?
                tool_blocks = [b for b in final.content if b.type == "tool_use"]
                if final.stop_reason != "tool_use" or not tool_blocks:
                    break

                # Run every requested tool concurrently.
                outcomes = await asyncio.gather(*(run_tool(b) for b in tool_blocks))

                tool_results = []
                for block, (content, is_error) in zip(tool_blocks, outcomes):
                    # To the UI: attach the result to the tool call it answers.
                    yield encoder.encode(
                        ToolCallResultEvent(
                            type=EventType.TOOL_CALL_RESULT,
                            message_id=str(uuid.uuid4()),
                            tool_call_id=block.id,
                            role="tool",
                            content=content,
                        )
                    )
                    # To Claude: the same result, in Anthropic's format.
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": content,
                            "is_error": is_error,
                        }
                    )

                # Extend the conversation: Claude's turn (with its tool_use
                # blocks), then our tool_result turn. Loop back so Claude
                # can read the results and continue.
                messages.append(
                    {
                        "role": "assistant",
                        "content": [b.model_dump(exclude_none=True) for b in final.content],
                    }
                )
                messages.append({"role": "user", "content": tool_results})
            else:
                # for/else: the loop ran MAX_STEPS times without a break.
                raise RuntimeError(f"Agent did not finish within {MAX_STEPS} steps")
            # ------------------------------------------------

            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

        except Exception as exc:  # a terminal event is mandatory
            logger.exception("agent run failed")
            yield encoder.encode(
                RunErrorEvent(type=EventType.RUN_ERROR, message=str(exc))
            )

    return StreamingResponse(
        event_generator(), media_type=encoder.get_content_type()
    )


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8008)