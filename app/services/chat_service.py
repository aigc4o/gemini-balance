import httpx
import json
import time
import uuid
from typing import Dict, Any, Optional, AsyncGenerator, Union
from app.core.config import settings
from app.core.logger import get_chat_logger
from app.schemas.gemini_models import GeminiRequest
from app.schemas.openai_models import ChatRequest

logger = get_chat_logger()


def convert_messages_to_gemini_format(messages: list) -> list:
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
                                "image_url": {
                                    "url": image_url,
                                }
                            }
                        )

        converted_messages.append({"role": role, "parts": parts})

    return converted_messages


def format_execution_result(result_data: dict) -> str:
    """格式化执行结果输出"""
    outcome = result_data.get("outcome", "")
    output = result_data.get("output", "").strip()
    return f"""\n【执行结果】\n> outcome: {outcome}\n\n【输出结果】\n```plaintext\n{output}\n```\n"""


def create_search_link(web):
    return f'\n- [{web["title"]}]({web["uri"]})'


class ChatService:
    def __init__(self, base_url: str, key_manager=None):
        self.base_url = base_url
        self.key_manager = key_manager
        self.thinking_first = True
        self.thinking_status = False

    def convert_gemini_response_to_openai(
        self,
        response: Dict[str, Any],
        model: str,
        stream: bool = False,
        finish_reason: str = None,
    ) -> Optional[Dict[str, Any]]:
        """Convert Gemini response to OpenAI format"""
        if stream:
            try:
                text = ""
                if response.get("candidates"):
                    candidate = response["candidates"][0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])

                    if "thinking" in model:
                        if settings.SHOW_THINKING_PROCESS:  
                            if len(parts) == 1:
                                if self.thinking_first:
                                    self.thinking_first = False
                                    self.thinking_status = True
                                    text = "> thinking\n\n" + parts[0].get("text")
                                else:
                                    text = parts[0].get("text")

                            if len(parts) == 2:
                                self.thinking_status = False
                                if self.thinking_first:
                                    self.thinking_first = False
                                    text = (
                                        "> thinking\n\n"
                                        + parts[0].get("text")
                                        + "\n\n---\n> output\n\n"
                                        + parts[1].get("text")
                                    )
                                else:
                                    text = (
                                        parts[0].get("text")
                                        + "\n\n---\n> output\n\n"
                                        + parts[1].get("text")
                                    )
                        else:
                            if len(parts) == 1:
                                if self.thinking_first:
                                    self.thinking_first = False
                                    self.thinking_status = True
                                    text = ""
                                elif self.thinking_status:
                                    text = ""
                                else:
                                    text = parts[0].get("text")
            
                            if len(parts) == 2:
                                self.thinking_status = False
                                if self.thinking_first:
                                    self.thinking_first = False
                                    text = parts[1].get("text")
                                else:
                                    text = parts[1].get("text")
                    else:
                        if "text" in parts[0]:
                            text = parts[0].get("text")
                        elif "executableCode" in parts[0]:
                            text = self.format_code_block(parts[0]["executableCode"])
                        elif "codeExecution" in parts[0]:
                            text = self.format_code_block(parts[0]["codeExecution"])
                        elif "executableCodeResult" in parts[0]:
                            text = format_execution_result(
                                parts[0]["executableCodeResult"]
                            )
                        elif "codeExecutionResult" in parts[0]:
                            text = format_execution_result(
                                parts[0]["codeExecutionResult"]
                            )
                        else:
                            text = ""

                    text = self.add_search_link_text(model, candidate, text)

                return {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text} if text else {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
            except Exception as e:
                logger.error(f"Error converting Gemini response: {str(e)}")
                logger.debug(f"Raw response: {response}")
                return None
        else:
            res = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response["candidates"][0]["content"]["parts"][0]["text"],
                        },
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            try:
                if response.get("candidates"):
                    candidate = response["candidates"][0]
                    if "thinking" in model:
                        if settings.SHOW_THINKING_PROCESS:
                            if len(candidate["content"]["parts"]) == 2:
                                text = (
                                    "> thinking\n\n"
                                    + candidate["content"]["parts"][0]["text"]
                                    + "\n\n---\n> output\n\n"
                                    + candidate["content"]["parts"][1]["text"]
                                )
                            else:
                                text = candidate["content"]["parts"][0]["text"]
                        else:
                            if len(candidate["content"]["parts"]) == 2:
                                text = candidate["content"]["parts"][1]["text"] 
                            else:
                                text = candidate["content"]["parts"][0]["text"]
                    else:
                        text = candidate["content"]["parts"][0]["text"]

                    text = self.add_search_link_text(model, candidate, text)
                    res["choices"][0]["message"]["content"] = text
                    return res
                else:
                    res["choices"][0]["message"]["content"] = "暂无返回"
                    return res
            except Exception as e:
                logger.error(f"Error converting Gemini response: {str(e)}")
                logger.debug(f"Raw response: {response}")
                res["choices"][0]["message"][
                    "content"
                ] = f"Error converting Gemini response: {str(e)}"
                return res

    def add_search_link_text(self, model, candidate, text):
        if (
            settings.SHOW_SEARCH_LINK
            and model.endswith("-search")
            and "groundingMetadata" in candidate
            and "groundingChunks" in candidate["groundingMetadata"]
        ):
            grounding_chunks = candidate["groundingMetadata"]["groundingChunks"]
            text += "\n\n---\n\n"
            text += "**【引用来源】**\n\n"
            for _, grounding_chunk in enumerate(grounding_chunks, 1):
                if "web" in grounding_chunk:
                    text += create_search_link(grounding_chunk["web"])
            return text
        else:
            return text

    async def create_chat_completion(
        self,
        request: ChatRequest,
        api_key: str,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Create chat completion using either Gemini or OpenAI API"""
        model = request.model
        tools = request.tools
        if tools is None:
            tools = []
        if settings.TOOLS_CODE_EXECUTION_ENABLED and not (
            model.endswith("-search") or "-thinking" in model
        ):
            tools.append({"code_execution": {}})
        if model.endswith("-search"):
            tools.append({"googleSearch": {}})
        return await self._gemini_chat_completion(request, api_key, tools)

    async def _gemini_chat_completion(
        self,
        request: ChatRequest,
        api_key: str,
        tools: Optional[list] = None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Handle Gemini API chat completion"""
        model = request.model
        messages = request.messages
        temperature = request.temperature
        stream = request.stream
        max_tokens = request.max_tokens
        stop = request.stop
        top_p = request.top_p
        top_k = request.top_k
        if model.endswith("-search"):
            gemini_model = model[:-7]  # Remove -search suffix
        else:
            gemini_model = model
        gemini_messages = convert_messages_to_gemini_format(messages)

        if not stream:
            # 非流式模式下，移除代码执行工具
            if {"code_execution": {}} in tools:
                tools.remove({"code_execution": {}})
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "stopSequences": stop,
                "topP": top_p,
                "topK": top_k,
            },
            "tools": tools,
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_CIVIC_INTEGRITY",
                    "threshold": "BLOCK_NONE",
                },
            ],
        }

        if stream:

            async def generate():
                retries = 0
                max_retries = 3
                current_api_key = api_key

                while retries < max_retries:
                    try:
                        timeout = httpx.Timeout(
                            300.0, read=300.0
                        )  # 连接超时300秒，读取超时300秒
                        async with httpx.AsyncClient(timeout=timeout) as async_client:
                            stream_url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:streamGenerateContent?alt=sse&key={current_api_key}"
                            async with async_client.stream(
                                "POST", stream_url, json=payload
                            ) as async_response:
                                if async_response.status_code != 200:
                                    error_content = await async_response.aread()
                                    error_msg = error_content.decode("utf-8")
                                    logger.error(
                                        f"API error: {async_response.status_code}, {error_msg}"
                                    )
                                    if retries < max_retries - 1:
                                        current_api_key = (
                                            await self.key_manager.handle_api_failure(
                                                current_api_key
                                            )
                                        )
                                        retries += 1
                                        continue
                                    else:
                                        logger.error(
                                            f"Max retries reached. Final error: {async_response.status_code}, {error_msg}"
                                        )
                                        yield f"data: {json.dumps({'error': f'API error: {async_response.status_code}, {error_msg}'})}\n\n"
                                        return

                                async for line in async_response.aiter_lines():
                                    if line.startswith("data: "):
                                        try:
                                            chunk = json.loads(line[6:])
                                            openai_chunk = (
                                                self.convert_gemini_response_to_openai(
                                                    chunk,
                                                    model,
                                                    stream=True,
                                                    finish_reason=None,
                                                )
                                            )
                                            if openai_chunk:
                                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                        except json.JSONDecodeError:
                                            continue
                                yield f"data: {json.dumps(self.convert_gemini_response_to_openai({}, model, stream=True, finish_reason='stop'))}\n\n"
                                yield "data: [DONE]\n\n"
                                return

                    except httpx.ReadTimeout:
                        logger.warning(
                            f"Read timeout occurred, attempting retry {retries + 1}"
                        )
                        if retries < max_retries - 1:
                            current_api_key = await self.key_manager.handle_api_failure(
                                current_api_key
                            )
                            logger.info(f"Switched to new API key: {current_api_key}")
                            retries += 1
                            continue
                        else:
                            logger.error(
                                f"Max retries reached. Final error: Read timeout"
                            )
                            yield f"data: {json.dumps({'error': 'Read timeout'})}\n\n"
                            return
                    except Exception as e:
                        logger.exception(
                            f"Stream error: {str(e)}, attempting retry {retries + 1}"
                        )
                        if retries < max_retries - 1:
                            current_api_key = await self.key_manager.handle_api_failure(
                                current_api_key
                            )
                            logger.info(f"Switched to new API key: {current_api_key}")
                            retries += 1
                            continue
                        else:
                            logger.error(f"Max retries reached. Final error: {e}")
                            yield f"data: {json.dumps({'error': str(e)})}\n\n"
                            return

            return generate()
        else:
            try:
                timeout = httpx.Timeout(
                    300.0, read=300.0
                )  # 连接超时300秒，读取超时300秒
                async with httpx.AsyncClient(timeout=timeout) as client:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"
                    response = await client.post(url, json=payload)
                    if response.status_code != 200:
                        error_text = response.text
                        error_code = response.status_code
                        raise Exception(
                            f"API调用错误 - 状态码: {error_code}, 响应内容: {error_text}"
                        )
                    gemini_response = response.json()
                    return self.convert_gemini_response_to_openai(
                        gemini_response, model, stream=False, finish_reason="stop"
                    )
            except Exception as e:
                logger.error(f"Error in non-stream completion")
                raise

    def format_code_block(self, code_data: dict) -> str:
        """格式化代码块输出"""
        language = code_data.get("language", "").lower()
        code = code_data.get("code", "").strip()

        return f"""\n【代码执行】\n```{language}\n{code}\n```\n"""

    async def generate_content(
        self, model_name: str, request: GeminiRequest, api_key: str
    ) -> dict:
        """调用Gemini API生成内容"""
        url = f"{self.base_url}/models/{model_name}:generateContent?key={api_key}"

        timeout = httpx.Timeout(300.0, read=300.0)  # 连接超时300秒，读取超时300秒
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(url, json=request.model_dump())
                if response.status_code == 200:
                    return response.json()
                else:
                    error_text = response.text
                    logger.error(f"Error: {response.status_code}")
                    logger.error(error_text)
                    raise Exception(
                        f"API request failed with status {response.status_code}: {error_text}"
                    )
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                raise

    async def stream_generate_content(
        self, model_name: str, request: GeminiRequest, api_key: str
    ) -> AsyncGenerator:
        """调用Gemini API流式生成内容"""
        retries = 0
        MAX_RETRIES = 3
        current_api_key = api_key

        while retries < MAX_RETRIES:
            try:
                url = f"{self.base_url}/models/{model_name}:streamGenerateContent?alt=sse&key={current_api_key}"
                timeout = httpx.Timeout(300.0, read=300.0)

                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST", url, json=request.model_dump()
                    ) as response:
                        if response.status_code != 200:
                            error_text = await response.text()
                            logger.error(f"Error: {response.status_code}: {error_text}")
                            if retries < MAX_RETRIES - 1:
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
                            raise Exception(
                                f"API request failed with status {response.status_code}: {error_text}"
                            )

                        async for line in response.aiter_lines():
                            logger.info(f"{line}")
                            if line.startswith("data: "):
                                try:
                                    json_data = json.loads(line[6:])
                                    # 这里需要根据Gemini API的响应结构来提取文本内容
                                    # 示例：如果响应结构是 {'candidates': [{'content': {'parts': [{'text': '...'}]}}]}
                                    if 'candidates' in json_data and json_data['candidates']:
                                        for candidate in json_data['candidates']:
                                            if 'content' in candidate and 'parts' in candidate['content']:
                                                for part in candidate['content']['parts']:
                                                  if 'text' in part:
                                                      yield part['text']
                                    # 如果响应结构是其他形式，则需要修改这里的解析逻辑
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to decode JSON: {line}")
                                    continue # 如果解析失败，则忽略该行数据
                            elif line.strip() == "":
                                continue # 忽略空行
                            else:
                                logger.warning(f"Unexpected line format: {line}")
                        return

            except httpx.ReadTimeout:
                logger.warning(f"Read timeout occurred, attempting retry {retries + 1}")
                if retries < MAX_RETRIES - 1:
                    current_api_key = await self.key_manager.handle_api_failure(
                        current_api_key
                    )
                    logger.info(f"Switched to new API key: {current_api_key}")
                    retries += 1
                    continue
                raise

            except Exception as e:
                logger.error(f"Streaming request failed: {str(e)}")
                if retries < MAX_RETRIES - 1:
                    current_api_key = await self.key_manager.handle_api_failure(
                        current_api_key
                    )
                    logger.info(f"Switched to new API key: {current_api_key}")
                    retries += 1
                    continue
                raise
