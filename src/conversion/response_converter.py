import json
import orjson
import uuid
from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest
from src.core.config import config


def _fast_json(obj) -> str:
    """Fast JSON serialization using orjson (5-10x faster than stdlib json)."""
    return orjson.dumps(obj).decode("utf-8")


# Pre-compute frequently used JSON fragments to avoid repeated serialization
_PING_EVENT = f"event: {Constants.EVENT_PING}\ndata: {_fast_json({'type': Constants.EVENT_PING})}\n\n"

# Disconnect check interval: check every N chunks instead of every chunk
_DISCONNECT_CHECK_INTERVAL = config.disconnect_check_interval


def convert_openai_to_claude_response(
    openai_response: dict, original_request: ClaudeMessagesRequest
) -> dict:
    """Convert OpenAI response to Claude format."""

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add thinking content block (before text, matching Claude's native order)
    reasoning_content = message.get("reasoning_content")
    thinking_blocks = message.get("thinking_blocks", [])

    if thinking_blocks:
        # Use thinking_blocks directly if available (has signature)
        for tb in thinking_blocks:
            if tb.get("type") == "thinking":
                block = {"type": "thinking", "thinking": tb.get("thinking", "")}
                if tb.get("signature"):
                    block["signature"] = tb["signature"]
                content_blocks.append(block)
    elif reasoning_content:
        # Fallback to reasoning_content string
        content_blocks.append({
            "type": "thinking",
            "thinking": reasoning_content,
        })

    # Add text content
    text_content = message.get("content")
    if text_content is not None:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = orjson.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # Ensure at least one content block
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": ""})

    # Map finish reason
    finish_reason = choice.get("finish_reason", "stop")
    stop_reason = {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason, Constants.STOP_END_TURN)

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    return claude_response


async def convert_openai_streaming_to_claude(
    openai_stream, original_request: ClaudeMessagesRequest, logger
):
    """Convert OpenAI streaming response to Claude streaming format."""

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {_fast_json({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    yield f"event: {Constants.EVENT_PING}\ndata: {_fast_json({'type': Constants.EVENT_PING})}\n\n"

    # State tracking
    current_block_index = 0
    thinking_block_started = False
    thinking_block_index = -1
    text_block_started = False
    text_block_index = -1
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    thinking_signature = None

    try:
        async for line in openai_stream:
            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = orjson.loads(chunk_data)
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle reasoning_content delta (thinking)
                    reasoning_content = delta.get("reasoning_content")
                    thinking_blocks = delta.get("thinking_blocks", None)

                    if reasoning_content is not None:
                        if not thinking_block_started:
                            thinking_block_index = current_block_index
                            current_block_index += 1
                            thinking_block_started = True
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': thinking_block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"

                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': thinking_block_index, 'delta': {'type': Constants.DELTA_THINKING, 'thinking': reasoning_content}})}\n\n"

                    if thinking_blocks:
                        for tb in thinking_blocks:
                            if tb.get("signature"):
                                thinking_signature = tb["signature"]

                    if thinking_signature and thinking_block_started and reasoning_content is None and not text_block_started:
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': thinking_block_index, 'delta': {'type': Constants.DELTA_SIGNATURE, 'signature': thinking_signature}})}\n\n"
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': thinking_block_index})}\n\n"
                        thinking_signature = None

                    # Handle text delta
                    if delta and "content" in delta and delta["content"] is not None:
                        if not text_block_started:
                            text_block_index = current_block_index
                            current_block_index += 1
                            text_block_started = True
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': text_block_index, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}})}\n\n"

                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}})}\n\n"

                    # Handle tool call deltas
                    if "tool_calls" in delta:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False
                                }

                            tool_call = current_tool_calls[tc_index]

                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            if (tool_call["id"] and tool_call["name"] and not tool_call["started"]):
                                tool_block_counter += 1
                                claude_index = current_block_index
                                current_block_index += 1
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}})}\n\n"

                            if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
                                tool_call["args_buffer"] += function_data["arguments"]

                                try:
                                    orjson.loads(tool_call["args_buffer"])
                                    if not tool_call["json_sent"]:
                                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_call['args_buffer']}})}\n\n"
                                        tool_call["json_sent"] = True
                                except json.JSONDecodeError:
                                    pass

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN
                        break

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {_fast_json(error_event)}\n\n"
        return

    # Send final SSE events
    if text_block_started and text_block_index >= 0:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index})}\n\n"

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']})}\n\n"

    if not thinking_block_started and not text_block_started and not current_tool_calls:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}})}\n\n"
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0})}\n\n"

    usage_data = {"input_tokens": 0, "output_tokens": 0}
    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data})}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {_fast_json({'type': Constants.EVENT_MESSAGE_STOP})}\n\n"


async def convert_openai_streaming_to_claude_with_cancellation(
    openai_stream,
    original_request: ClaudeMessagesRequest,
    logger,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Convert OpenAI streaming response to Claude streaming format with cancellation support.

    Handles reasoning_content (thinking) deltas from upstream:
    - reasoning_content arrives BEFORE content in the stream
    - We emit a thinking content block first, then switch to text content block
    """

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    yield f"event: {Constants.EVENT_MESSAGE_START}\ndata: {_fast_json({'type': Constants.EVENT_MESSAGE_START, 'message': {'id': message_id, 'type': 'message', 'role': Constants.ROLE_ASSISTANT, 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0}}})}\n\n"

    yield _PING_EVENT

    # State tracking
    current_block_index = 0
    thinking_block_started = False
    thinking_block_index = -1
    text_block_started = False
    text_block_index = -1
    tool_block_counter = 0
    current_tool_calls = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data = {"input_tokens": 0, "output_tokens": 0}
    chunk_counter = 0
    thinking_signature = None

    try:
        async for line in openai_stream:
            # Check disconnect at intervals, not every chunk
            chunk_counter += 1
            if chunk_counter % _DISCONNECT_CHECK_INTERVAL == 0:
                if await http_request.is_disconnected():
                    logger.info(f"Client disconnected, cancelling request {request_id}")
                    openai_client.cancel_request(request_id)
                    break

            if line.strip():
                if line.startswith("data: "):
                    chunk_data = line[6:]
                    if chunk_data.strip() == "[DONE]":
                        break

                    try:
                        chunk = orjson.loads(chunk_data)
                        usage = chunk.get("usage", None)
                        if usage:
                            cache_read_input_tokens = 0
                            prompt_tokens_details = usage.get('prompt_tokens_details', {})
                            if prompt_tokens_details:
                                cache_read_input_tokens = prompt_tokens_details.get('cached_tokens', 0)
                            usage_data = {
                                'input_tokens': usage.get('prompt_tokens', 0),
                                'output_tokens': usage.get('completion_tokens', 0),
                                'cache_read_input_tokens': cache_read_input_tokens
                            }
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to parse chunk: {chunk_data}, error: {e}"
                        )
                        continue

                    choice = choices[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason")

                    # Handle reasoning_content delta (thinking)
                    reasoning_content = delta.get("reasoning_content")
                    thinking_blocks = delta.get("thinking_blocks", None)

                    if reasoning_content is not None:
                        if not thinking_block_started:
                            # Start thinking content block
                            thinking_block_index = current_block_index
                            current_block_index += 1
                            thinking_block_started = True
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': thinking_block_index, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"

                        # Send thinking delta
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': thinking_block_index, 'delta': {'type': Constants.DELTA_THINKING, 'thinking': reasoning_content}})}\n\n"

                    # Extract signature from thinking_blocks if present
                    if thinking_blocks:
                        for tb in thinking_blocks:
                            if tb.get("signature"):
                                thinking_signature = tb["signature"]

                    # If we get a signature and thinking block is open, close it with signature
                    if thinking_signature and thinking_block_started and reasoning_content is None and not text_block_started:
                        # Send signature delta before closing
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': thinking_block_index, 'delta': {'type': Constants.DELTA_SIGNATURE, 'signature': thinking_signature}})}\n\n"
                        # Close thinking block
                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': thinking_block_index})}\n\n"
                        thinking_signature = None  # consumed

                    # Handle text content delta
                    if delta and "content" in delta and delta["content"] is not None:
                        if not text_block_started:
                            # If thinking block was open but not yet closed, close it first
                            if thinking_block_started and thinking_block_index >= 0:
                                # Check if we already closed it (via signature path above)
                                # If not closed yet, close now
                                pass  # Already handled above via signature check

                            # Start text content block
                            text_block_index = current_block_index
                            current_block_index += 1
                            text_block_started = True
                            yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': text_block_index, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}})}\n\n"

                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': text_block_index, 'delta': {'type': Constants.DELTA_TEXT, 'text': delta['content']}})}\n\n"

                    # Handle tool call deltas
                    if "tool_calls" in delta and delta["tool_calls"]:
                        for tc_delta in delta["tool_calls"]:
                            tc_index = tc_delta.get("index", 0)

                            if tc_index not in current_tool_calls:
                                current_tool_calls[tc_index] = {
                                    "id": None,
                                    "name": None,
                                    "args_buffer": "",
                                    "json_sent": False,
                                    "claude_index": None,
                                    "started": False
                                }

                            tool_call = current_tool_calls[tc_index]

                            if tc_delta.get("id"):
                                tool_call["id"] = tc_delta["id"]

                            function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
                            if function_data.get("name"):
                                tool_call["name"] = function_data["name"]

                            if (tool_call["id"] and tool_call["name"] and not tool_call["started"]):
                                tool_block_counter += 1
                                claude_index = current_block_index
                                current_block_index += 1
                                tool_call["claude_index"] = claude_index
                                tool_call["started"] = True

                                yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': claude_index, 'content_block': {'type': Constants.CONTENT_TOOL_USE, 'id': tool_call['id'], 'name': tool_call['name'], 'input': {}}})}\n\n"

                            if "arguments" in function_data and tool_call["started"] and function_data["arguments"] is not None:
                                tool_call["args_buffer"] += function_data["arguments"]

                                try:
                                    orjson.loads(tool_call["args_buffer"])
                                    if not tool_call["json_sent"]:
                                        yield f"event: {Constants.EVENT_CONTENT_BLOCK_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_DELTA, 'index': tool_call['claude_index'], 'delta': {'type': Constants.DELTA_INPUT_JSON, 'partial_json': tool_call['args_buffer']}})}\n\n"
                                        tool_call["json_sent"] = True
                                except json.JSONDecodeError:
                                    pass

                    # Handle finish reason
                    if finish_reason:
                        if finish_reason == "length":
                            final_stop_reason = Constants.STOP_MAX_TOKENS
                        elif finish_reason in ["tool_calls", "function_call"]:
                            final_stop_reason = Constants.STOP_TOOL_USE
                        elif finish_reason == "stop":
                            final_stop_reason = Constants.STOP_END_TURN
                        else:
                            final_stop_reason = Constants.STOP_END_TURN

    except HTTPException as e:
        if e.status_code == 499:
            logger.info(f"Request {request_id} was cancelled")
            error_event = {
                "type": "error",
                "error": {
                    "type": "cancelled",
                    "message": "Request was cancelled by client",
                },
            }
            yield f"event: error\ndata: {_fast_json(error_event)}\n\n"
            return
        else:
            raise
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": f"Streaming error: {str(e)}"},
        }
        yield f"event: error\ndata: {_fast_json(error_event)}\n\n"
        return

    # Send final SSE events — close any open blocks
    # If text block was never started (e.g., only thinking + tool calls), don't emit text block stop
    if text_block_started and text_block_index >= 0:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': text_block_index})}\n\n"

    for tool_data in current_tool_calls.values():
        if tool_data.get("started") and tool_data.get("claude_index") is not None:
            yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': tool_data['claude_index']})}\n\n"

    # If no blocks were started at all (edge case), emit an empty text block
    if not thinking_block_started and not text_block_started and not current_tool_calls:
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_START}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_START, 'index': 0, 'content_block': {'type': Constants.CONTENT_TEXT, 'text': ''}})}\n\n"
        yield f"event: {Constants.EVENT_CONTENT_BLOCK_STOP}\ndata: {_fast_json({'type': Constants.EVENT_CONTENT_BLOCK_STOP, 'index': 0})}\n\n"

    yield f"event: {Constants.EVENT_MESSAGE_DELTA}\ndata: {_fast_json({'type': Constants.EVENT_MESSAGE_DELTA, 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': usage_data})}\n\n"
    yield f"event: {Constants.EVENT_MESSAGE_STOP}\ndata: {_fast_json({'type': Constants.EVENT_MESSAGE_STOP})}\n\n"
