import httpx
import json
import time
import uuid
import logging
from typing import Dict, Any, Optional, AsyncGenerator, Union
import openai
from app.core.config import settings

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, base_url: str, key_manager=None):
        self.base_url = base_url
        self.key_manager = key_manager

    def convert_messages_to_gemini_format(self, messages: list) -> list:
        """Convert OpenAI message format to Gemini format"""
        converted_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            parts = []

            # 处理文本内容
            if isinstance(msg["content"], str):
                parts.append({"text": msg["content"]})
            # 处理包含图片的消息
            elif isinstance(msg["content"], list):
                for content in msg["content"]:
                    if isinstance(content, str):
                        parts.append({"text": content})
                    elif isinstance(content, dict) and content["type"] == "text":
                        parts.append({"text": content["text"]})
                    elif isinstance(content, dict) and content["type"] == "image_url":
                        # 处理图片URL
                        image_url = content["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            # 处理base64图片
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image_url.split(",")[1],
                                    }
                                }
                            )
                        else:
                            # 处理普通URL图片
                            parts.append(
                                {
                                    "inline_data": {
                                        "mime_type": "image/jpeg",
                                        "data": image_url,
                                    }
                                }
                            )

            converted_messages.append({"role": role, "parts": parts})

        return converted_messages

    def convert_gemini_response_to_openai(
        self, response: Dict[str, Any], model: str, stream: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Convert Gemini response to OpenAI format"""
        if stream:
            if not response.get("candidates"):
                return None

            try:
                candidate = response["candidates"][0]
                content = candidate.get("content", {})
                parts = content.get("parts", [])

                if not parts:
                    return None

                if "text" in parts[0]:
                    text = parts[0].get("text")
                elif "executableCode" in parts[0]:
                    text = self.format_code_block(parts[0]["executableCode"])
                elif "executableCodeResult" in parts[0]:
                    text = self.format_execution_result(parts[0]["executableCodeResult"])
                else:
                    text = ""

                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }
                    ],
                }
            except Exception as e:
                logger.error(f"Error converting Gemini response: {str(e)}")
                logger.debug(f"Raw response: {response}")
                return None
        else:
            return {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response["candidates"][0]["content"]["parts"][0][
                                "text"
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

    async def create_chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float,
        stream: bool,
        api_key: str,
        tools: Optional[list] = None,
        tool_choice: Optional[str] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Create chat completion using either Gemini or OpenAI API"""

        if tools is None:
            tools = []
        if settings.TOOLS_CODE_EXECUTION_ENABLED:
            tools.append({"code_execution": {}})
        if model.endswith("-search"):
            tools.append({"googleSearch": {}})
        return await self._gemini_chat_completion(
            messages, model, temperature, stream, api_key, tools
        )
        # else:
        #     return await self._openai_chat_completion(
        #         messages, model, temperature, stream, api_key, tools
        #     )

    async def _gemini_chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float,
        stream: bool,
        api_key: str,
        tools: Optional[list] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Handle Gemini API chat completion"""
        if model.endswith("-search"):
            gemini_model = model[:-7]  # Remove -search suffix
        else:
            gemini_model = model
        gemini_messages = self.convert_messages_to_gemini_format(messages)

        payload = {
            "contents": gemini_messages,
            "generationConfig": {"temperature": temperature},
            "tools": tools,
        }

        if stream:

            async def generate():
                retries = 0
                MAX_RETRIES = 3
                current_api_key = api_key

                while retries < MAX_RETRIES:
                    try:
                        async with httpx.AsyncClient() as client:
                            stream_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:streamGenerateContent?alt=sse&key={current_api_key}"
                            async with client.stream(
                                "POST", stream_url, json=payload
                            ) as response:
                                if response.status_code != 200:
                                    if retries < MAX_RETRIES - 1:
                                        logger.warning(
                                            f"API error: {response.status_code}, attempting retry {retries + 1}"
                                        )
                                        current_api_key = (
                                            await self.key_manager.handle_api_failure(
                                                current_api_key
                                            )
                                        )
                                        logger.info(
                                            f"Switched to new API key: {current_api_key}"
                                        )
                                        retries += 1
                                        continue
                                    else:
                                        logger.error(
                                            f"Max retries reached. Final error: {response.status_code}"
                                        )
                                        yield f"data: {json.dumps({'error': f'API error: {response.status_code}'})}\n\n"
                                        return

                                async for line in response.aiter_lines():
                                    if line.startswith("data: "):
                                        try:
                                            chunk = json.loads(line[6:])
                                            openai_chunk = (
                                                self.convert_gemini_response_to_openai(
                                                    chunk, model, stream=True
                                                )
                                            )
                                            if openai_chunk:
                                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                        except json.JSONDecodeError:
                                            continue
                                yield "data: [DONE]\n\n"
                                return  # 成功完成,退出重试循环

                    except Exception as e:
                        if retries < MAX_RETRIES - 1:
                            logger.warning(
                                f"Stream error: {str(e)}, attempting retry {retries + 1}"
                            )
                            current_api_key = await self.key_manager.handle_api_failure(
                                current_api_key
                            )
                            retries += 1
                            continue
                        else:
                            logger.error(f"Max retries reached. Final error: {str(e)}")
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"
                            return

            return generate()
        else:
            async with httpx.AsyncClient() as client:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
                response = await client.post(url, json=payload)
                gemini_response = response.json()
                return self.convert_gemini_response_to_openai(gemini_response, model)

    async def _openai_chat_completion(
        self,
        messages: list,
        model: str,
        temperature: float,
        stream: bool,
        api_key: str,
        tools: Optional[list] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Handle OpenAI API chat completion"""
        client = openai.OpenAI(api_key=api_key, base_url=self.base_url)
        if tools:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=stream,
                tools=tools,
            )
        else:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, stream=stream
            )

        if stream:

            async def generate():
                for chunk in response:
                    yield f"data: {chunk.model_dump_json()}\n\n"

            return generate()

        return response

    def format_code_block(self, code_data: dict) -> str:
        """格式化代码块输出"""
        language = code_data.get("language", "").lower()
        code = code_data.get("code", "").strip()

        return f"""\n```{language}\n{code}\n```\n"""


    def format_execution_result(result_data: dict) -> str:
        """格式化执行结果输出"""
        outcome = result_data.get("outcome", "")
        output = result_data.get("output", "").strip()
        return f"""\n【执行结果】\n{output}\n"""