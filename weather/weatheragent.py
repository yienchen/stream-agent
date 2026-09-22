"""
Weather agent as a plain AG-UI endpoint.

Contract (this is the whole thing):
  POST /agent  <- RunAgentInput JSON (camelCase)
  ->  text/event-stream of AG-UI events, starting with RUN_STARTED
      and ending with RUN_FINISHED or RUN_ERROR.

No /info, no sdk.info(), no agent registry. Discovery is the
CopilotKit *runtime's* job, not this server's.

    pip install ag-ui-protocol fastapi uvicorn anthropic
    uvicorn agui_server:app --host 127.0.0.1 --port 8008 --reload
"""

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

app = FastAPI()

# Only needed if the browser talks to this server directly
# (agents__unsafe_dev_only / selfManagedAgents). Behind a Node
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

# Name must match `name` in the frontend's useRenderTool({ name: ... })
# call — that's how CopilotKit routes this tool call to WeatherCard
# instead of falling back to default rendering.
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Fetch current weather for a location.",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


async def execute_weather_lookup(location: str) -> dict:
    # Keys here must match the props WeatherCard destructures.
    return {
        "location": location,
        "temperature": "72F",
        "condition": "Sunny",
        "humidity": "48%",
    }


def to_anthropic_messages(input_data: RunAgentInput) -> list[dict]:
    """Map AG-UI messages onto the Anthropic messages shape."""
    out: list[dict] = []
    for m in input_data.messages:
        role = getattr(m, "role", None)
        content = getattr(m, "content", None)
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out or [{"role": "user", "content": "Hello"}]


@app.post("/agent")
async def agent_endpoint(input_data: RunAgentInput, request: Request):

    ## logger.log(logging.INFO, f"agent run: json={json.dumps(input_data)}")

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
            message_id = str(uuid.uuid4())
            started_text = False
            # Anthropic identifies content blocks by index; AG-UI identifies
            # tool calls by id. Track the mapping so TOOL_CALL_ARGS deltas
            # (which only carry the index) can reference the right id.
            tool_call_ids_by_index: dict[int, str] = {}

            async with anthropic_client.messages.stream(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                tools=[WEATHER_TOOL],
                messages=to_anthropic_messages(input_data),
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            tool_call_ids_by_index[event.index] = block.id
                            yield encoder.encode(
                                ToolCallStartEvent(
                                    type=EventType.TOOL_CALL_START,
                                    tool_call_id=block.id,
                                    tool_call_name=block.name,
                                    parent_message_id=message_id,
                                )
                            )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            if not started_text:
                                yield encoder.encode(
                                    TextMessageStartEvent(
                                        type=EventType.TEXT_MESSAGE_START,
                                        message_id=message_id,
                                        role="assistant",
                                    )
                                )
                                started_text = True
                            yield encoder.encode(
                                TextMessageContentEvent(
                                    type=EventType.TEXT_MESSAGE_CONTENT,
                                    message_id=message_id,
                                    delta=delta.text,
                                )
                            )
                        elif delta.type == "input_json_delta":
                            tool_call_id = tool_call_ids_by_index.get(event.index)
                            if tool_call_id is None:
                                # No TOOL_CALL_START seen for this index — skip
                                # rather than send an id the client will reject.
                                continue
                            yield encoder.encode(
                                ToolCallArgsEvent(
                                    type=EventType.TOOL_CALL_ARGS,
                                    tool_call_id=tool_call_id,
                                    delta=delta.partial_json,
                                )
                            )

                final = await stream.get_final_message()

            if started_text:
                yield encoder.encode(
                    TextMessageEndEvent(
                        type=EventType.TEXT_MESSAGE_END, message_id=message_id
                    )
                )

            for block in final.content:
                if getattr(block, "type", None) == "tool_use":
                    yield encoder.encode(
                        ToolCallEndEvent(
                            type=EventType.TOOL_CALL_END, tool_call_id=block.id
                        )
                    )
                    # Server-side tool: run it and attach the result to
                    # THIS tool call as structured JSON, via
                    # ToolCallResultEvent — not a narrated text message.
                    # useRenderTool's `result` prop is populated straight
                    # from this event's `content`, parsed as JSON.
                    result = await execute_weather_lookup(
                        block.input.get("location", "")
                    )
                    yield encoder.encode(
                        ToolCallResultEvent(
                            type=EventType.TOOL_CALL_RESULT,
                            message_id=str(uuid.uuid4()),
                            tool_call_id=block.id,
                            role="tool",
                            content=json.dumps(result),
                        )
                    )

            yield encoder.encode(
                RunFinishedEvent(
                    type=EventType.RUN_FINISHED,
                    thread_id=input_data.thread_id,
                    run_id=input_data.run_id,
                )
            )

        except Exception as exc:  # terminal event is mandatory
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