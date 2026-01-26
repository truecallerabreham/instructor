"""AWS Bedrock-specific utilities."""

from __future__ import annotations

import base64
import json
import mimetypes
from textwrap import dedent
from typing import Any

import requests


def generate_bedrock_schema(response_model: type[Any]) -> dict[str, Any]:
    """
    Generate Bedrock tool schema from a Pydantic model.
    """
    schema = response_model.model_json_schema()

    return {
        "toolSpec": {
            "name": response_model.__name__,
            "description": response_model.__doc__
            or f"Correctly extracted `{response_model.__name__}` with all the required parameters with correct types",
            "inputSchema": {"json": schema},
        }
    }


def reask_bedrock_json(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for Bedrock JSON mode when validation fails."""
    kwargs = kwargs.copy()
    reask_msgs = [response["output"]["message"]]
    reask_msgs.append(
        {
            "role": "user",
            "content": [
                {
                    "text": f"Correct your JSON ONLY RESPONSE, based on the following errors:\n{exception}"
                },
            ],
        }
    )
    kwargs["messages"].extend(reask_msgs)
    return kwargs


def reask_bedrock_tools(
    kwargs: dict[str, Any],
    response: Any,
    exception: Exception,
):
    """Handle reask for Bedrock tools mode when validation fails."""
    kwargs = kwargs.copy()

    assistant_message = response["output"]["message"]
    reask_msgs = [assistant_message]

    tool_use_id = None
    if "content" in assistant_message:
        for content_block in assistant_message["content"]:
            if "toolUse" in content_block:
                tool_use_id = content_block["toolUse"]["toolUseId"]
                break

    if tool_use_id:
        reask_msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [
                                {
                                    "text": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                                }
                            ],
                            "status": "error",
                        }
                    }
                ],
            }
        )
    else:
        reask_msgs.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": f"Validation Error due to no tool invocation:\n{exception}\nRecall the function correctly, fix the errors"
                    }
                ],
            }
        )

    kwargs["messages"].extend(reask_msgs)
    return kwargs


def _normalize_bedrock_image_format(mime_or_ext: str) -> str:
    """Map common image types to Bedrock format enum."""
    if not mime_or_ext:
        return "jpeg"
    val = mime_or_ext.strip().lower()
    if "/" in val:
        val = val.split("/", 1)[1]
    if val in ("jpg", "pjpeg", "x-jpeg", "x-jpg"):
        return "jpeg"
    if val in ("png", "x-png"):
        return "png"
    if val in ("gif", "x-gif"):
        return "gif"
    if val in ("webp", "image/webp"):
        return "webp"
    return "jpeg"


def _openai_image_part_to_bedrock(part: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenAI-style image parts to Bedrock content."""
    image_url = (part.get("image_url") or {}).get("url")
    if not image_url:
        raise ValueError("image_url.url is required for OpenAI-style image parts")

    guessed_mime = mimetypes.guess_type(image_url)[0] or "image/jpeg"
    fmt = _normalize_bedrock_image_format(guessed_mime)

    if image_url.startswith("data:"):
        try:
            header, b64 = image_url.split(",", 1)
        except ValueError as e:
            raise ValueError("Invalid data URL in image_url.url") from e
        if ";base64" not in header:
            raise ValueError("Only base64 data URLs are supported for Bedrock")
        return {"image": {"format": fmt, "source": {"bytes": base64.b64decode(b64)}}}

    if image_url.startswith(("http://", "https://")):
        try:
            resp = requests.get(image_url, timeout=15)
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type")
            if ctype and "/" in ctype:
                fmt = _normalize_bedrock_image_format(ctype)
            return {"image": {"format": fmt, "source": {"bytes": resp.content}}}
        except requests.exceptions.Timeout as e:  # type: ignore[attr-defined]
            raise ValueError(f"Timed out while fetching image from {image_url}") from e
        except requests.exceptions.ConnectionError as e:  # type: ignore[attr-defined]
            raise ValueError(
                f"Connection error while fetching image from {image_url}: {e}"
            ) from e
        except requests.exceptions.HTTPError as e:  # type: ignore[attr-defined]
            raise ValueError(
                f"HTTP error while fetching image from {image_url}: {e}"
            ) from e
        except requests.exceptions.RequestException as e:  # type: ignore[attr-defined]
            raise ValueError(
                f"Request error while fetching image from {image_url}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Unexpected error while fetching image from {image_url}: {e}"
            ) from e

    raise ValueError(
        "Unsupported image_url scheme. Use http(s) or data:image/...;base64,..."
    )


def _to_bedrock_content_items(content: Any) -> list[dict[str, Any]]:
    """Normalize content into Bedrock Converse content list."""
    if isinstance(content, str):
        return [{"text": content}]

    if isinstance(content, list):
        items: list[dict[str, Any]] = []
        for p in content:
            if isinstance(p, dict) and "type" in p:
                t = p.get("type")
                if t in ("text", "input_text"):
                    txt = p.get("text") or p.get("input_text") or ""
                    items.append({"text": txt})
                    continue
                if t == "image_url":
                    items.append(_openai_image_part_to_bedrock(p))
                    continue
                raise ValueError(f"Unsupported OpenAI-style part type for Bedrock: {t}")

            if isinstance(p, dict):
                if "text" in p and isinstance(p["text"], str) and set(p.keys()) == {"text"}:
                    items.append(p)
                    continue
                if "image" in p and isinstance(p["image"], dict):
                    items.append(p)
                    continue
                if "document" in p and isinstance(p["document"], dict):
                    items.append(p)
                    continue
                raise ValueError(f"Unsupported dict content for Bedrock: {p}")

            if isinstance(p, str):
                items.append({"text": p})
                continue

            raise ValueError(f"Unsupported content part for Bedrock: {type(p)}")
        return items

    raise ValueError(f"Unsupported message content type for Bedrock: {type(content)}")


def _prepare_bedrock_converse_kwargs_internal(
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Prepare kwargs for the Bedrock Converse API."""
    if "system" in call_kwargs and isinstance(call_kwargs["system"], list):
        system_content = call_kwargs.pop("system")
        if (
            system_content
            and isinstance(system_content[0], dict)
            and "text" in system_content[0]
        ):
            system_text = system_content[0]["text"]
            if "messages" not in call_kwargs:
                call_kwargs["messages"] = []
            call_kwargs["messages"].insert(0, {"role": "system", "content": system_text})

    if "model" in call_kwargs and "modelId" not in call_kwargs:
        call_kwargs["modelId"] = call_kwargs.pop("model")

    inference_config_params = {}
    if "temperature" in call_kwargs:
        inference_config_params["temperature"] = call_kwargs.pop("temperature")

    if "max_tokens" in call_kwargs:
        inference_config_params["maxTokens"] = call_kwargs.pop("max_tokens")
    elif "maxTokens" in call_kwargs:
        inference_config_params["maxTokens"] = call_kwargs.pop("maxTokens")

    if "top_p" in call_kwargs:
        inference_config_params["topP"] = call_kwargs.pop("top_p")
    elif "topP" in call_kwargs:
        inference_config_params["topP"] = call_kwargs.pop("topP")

    if "stop" in call_kwargs:
        stop_val = call_kwargs.pop("stop")
        if isinstance(stop_val, str):
            inference_config_params["stopSequences"] = [stop_val]
        elif isinstance(stop_val, list):
            inference_config_params["stopSequences"] = stop_val
    elif "stop_sequences" in call_kwargs:
        inference_config_params["stopSequences"] = call_kwargs.pop("stop_sequences")
    elif "stopSequences" in call_kwargs:
        inference_config_params["stopSequences"] = call_kwargs.pop("stopSequences")

    if inference_config_params:
        if "inferenceConfig" in call_kwargs:
            existing_inference_config = call_kwargs["inferenceConfig"]
            for key, value in inference_config_params.items():
                if key not in existing_inference_config:
                    existing_inference_config[key] = value
        else:
            call_kwargs["inferenceConfig"] = inference_config_params

    if "messages" in call_kwargs and isinstance(call_kwargs["messages"], list):
        original_input_messages = call_kwargs.pop("messages")
        bedrock_system_list: list[dict[str, Any]] = []
        bedrock_user_assistant_messages_list: list[dict[str, Any]] = []

        for msg_dict in original_input_messages:
            if not isinstance(msg_dict, dict):
                bedrock_user_assistant_messages_list.append(msg_dict)
                continue

            current_message_for_api = msg_dict.copy()
            role = current_message_for_api.get("role")
            content = current_message_for_api.get("content")

            if role == "system":
                if isinstance(content, str):
                    bedrock_system_list.append({"text": content})
                else:
                    raise ValueError(
                        "System message content must be a string for Bedrock processing by this handler. "
                        f"Found type: {type(content)}."
                    )
            else:
                if "content" in current_message_for_api:
                    current_message_for_api["content"] = _to_bedrock_content_items(
                        content
                    )
                bedrock_user_assistant_messages_list.append(current_message_for_api)

        if bedrock_system_list:
            call_kwargs["system"] = bedrock_system_list

        call_kwargs["messages"] = bedrock_user_assistant_messages_list
    return call_kwargs


def handle_bedrock_json(
    response_model: type[Any], new_kwargs: dict[str, Any]
) -> tuple[type[Any], dict[str, Any]]:
    """Handle Bedrock JSON mode."""
    new_kwargs = _prepare_bedrock_converse_kwargs_internal(new_kwargs)
    json_message = dedent(
        f"""
        As a genius expert, your task is to understand the content and provide
        the parsed objects in json that match the following json_schema:\n

        {json.dumps(response_model.model_json_schema(), indent=2, ensure_ascii=False)}

        Make sure to return an instance of the JSON, not the schema itself
        and don't include any other text in the response apart from the json
        """
    )
    system_message = new_kwargs.pop("system", None)
    if not system_message:
        new_kwargs["system"] = [{"text": json_message}]
    else:
        if not isinstance(system_message, list):
            raise ValueError(
                """system must be a list of SystemMessage, refer to:
                https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html
                """
            )
        system_message.append({"text": json_message})
        new_kwargs["system"] = system_message

    return response_model, new_kwargs


def handle_bedrock_tools(
    response_model: type[Any] | None, new_kwargs: dict[str, Any]
) -> tuple[type[Any] | None, dict[str, Any]]:
    """Handle Bedrock tools mode."""
    new_kwargs = _prepare_bedrock_converse_kwargs_internal(new_kwargs)

    if response_model is None:
        return None, new_kwargs

    tool_schema = generate_bedrock_schema(response_model)
    new_kwargs["toolConfig"] = {
        "tools": [tool_schema],
        "toolChoice": {"tool": {"name": response_model.__name__}},
    }

    return response_model, new_kwargs
