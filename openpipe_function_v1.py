"""
title: OpenPipe Raw Capture Function (Pre-Processing)
author: Cline & Gwyn
version: 1.0.0
required_open_webui_version: 0.5.0
icon_url: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAAQlBMVEVHcEz5Tyr////vRB8AAAD/VjLgXEAAAAAAAAAAAADlNxEAAADbLAX/l4JnJRfeRif7z8f9qpgkCQT/YkG3Oh9/MyIY1MKeAAAACnRSTlMA////Mf//injo8EBKYgAAArFJREFUeJztW4tugkAQ7BZPhS2gIP//q+WEe/soSG8uOcY0MW3ijsPs63L9+tqxY8eOdHHeECvCH0reEOVhMYGSu5vB8Xi9KNS3P2LQ7xoul8Y/c9cK0atXfb18K1zk76efCXRHcUcl8TOjPSo0vPQpnLkhHcElcLX+IGgKX8iXjO9QMASGDwkIRwEK4msBxvDFfxB4pgBpBkqCWYEqigIkrPhT+Gp+AlsTeKGAJYGhsDmBJx4g24SFI0EkBYwJC6KQwf97gHwFYnvAegSKQ3QF6IEHVSncgkD/1gPCKQNV3DrgFCKrFMbzwDML4BTY3AN/zgI/EWNnAaoOUCABrhISSgG3FWyuwINCdLGUCbrBZt3wwDy0GuNH1FeFun2Oqn04lA4dL57LT8ydAz3jd8vBfFoaf2RQSrzbOLrXfOROIrEi/ojDiBM3LxR38NMq6W/Hdn41fJKfsir8rAM3YjZj39fGCN5aolKgvRvAXkjWffeQgJ8KNZnoqgroPhCRgBkH7MUsHgEzkxZQBUwjiKyANQ6gFHDnEcUgOgHPgwAFpmY4Hw5E9YCTBQAFyM0CgAcSUYBUKc4uC0i4IyGsDsA8QG4vwHkAqkDevYAMg0w9oEqxe1AMnIggCngHlZipuEBPxea0Pr+p2D8gAnhAYBUA7wUkwiTEzIS4XpDIVAzcC9KaiDLdC+xekK8H8t4L7GKc316gSzHujCiBvQDbCwjdDd0swHkglSzAbEb2VAw7KwZux+gs8E9Ko29G/l5QYPcCwG6I3o69iShTBfKeiAg9ESUzFee7HQsB3o6TmIiw23EqUzFsO05iM7KyAOMBgc2C7G9QROkF3WMCOrxdCwMFusU3KX3Im5Xzl+373rpNZ+TXTyB8BAMvv0kZSvARPn0CksEH//O18ibljh07duSFX1tOylaAmcq/AAAAAElFTkSuQmCC
"""

"""
OpenPipe Raw Capture Function v1.0 - Pre-Processing Interception

This function captures raw chat completions BEFORE OpenWebUI processes them by acting as a 
proxy/wrapper around model calls. It intercepts the raw API responses and reports them to 
OpenPipe while maintaining complete compatibility with all model types.

Key advantages over the filter approach:
- Captures truly raw, unprocessed responses
- Works at the model call level, not the UI processing level  
- Gets the actual HTTP response payloads from model providers
- No dependency on OpenWebUI's message formatting

Authors: Cline (AI Assistant) & Gwyn (Human Collaborator) 
Version: 1.0.0 - Raw Capture Function
Created: January 2025
"""

import asyncio
import json
import time
from typing import Optional, Dict, Any, List, Union, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import Request

class Pipe:
    """
    OpenPipe Raw Capture Function
    
    Acts as a proxy/wrapper to capture raw chat completion responses before OpenWebUI 
    processes them. This ensures we get the truly raw, unmodified API responses for 
    comprehensive analytics and training data collection.
    
    Unlike filters which work on processed OpenWebUI data, this function intercepts
    at the model call level to capture authentic API responses.
    """
    
    class Valves(BaseModel):
        # Core Configuration
        OPENPIPE_API_KEY: str = Field(
            default="", 
            description="OpenPipe API key for authentication"
        )
        OPENPIPE_BASE_URL: str = Field(
            default="https://api.openpipe.ai/api/v1", 
            description="OpenPipe API base URL"
        )
        
        # Target Model Configuration
        TARGET_MODEL: str = Field(
            default="", 
            description="The actual model to proxy calls to (e.g. 'gpt-4', 'claude-3-sonnet', 'llama3.2')"
        )
        FUNCTION_NAME_PREFIX: str = Field(
            default="📊 OpenPipe Capture: ", 
            description="Prefix shown in model selector for this capture function"
        )
        
        # Raw Capture Settings
        CAPTURE_STREAMING_CHUNKS: bool = Field(
            default=True, 
            description="Capture and report individual streaming response chunks"
        )
        CAPTURE_FULL_HTTP_HEADERS: bool = Field(
            default=False, 
            description="Include full HTTP response headers in capture (may contain sensitive info)"
        )
        PRESERVE_RAW_RESPONSE_FORMAT: bool = Field(
            default=True, 
            description="Keep original API response format instead of normalizing"
        )
        
        # Analytics Configuration  
        INCLUDE_REQUEST_TIMING: bool = Field(
            default=True, 
            description="Measure and report request/response timing"
        )
        INCLUDE_TOKEN_ANALYSIS: bool = Field(
            default=True, 
            description="Analyze and report token usage patterns"
        )
        INCLUDE_INTERACTION_METADATA: bool = Field(
            default=True, 
            description="Extract and report interaction classification metadata"
        )
        
        # Selective Reporting
        REPORT_SUCCESSFUL_COMPLETIONS: bool = Field(
            default=True, 
            description="Report successful chat completions"
        )
        REPORT_ERROR_RESPONSES: bool = Field(
            default=True, 
            description="Report failed/error responses for debugging"
        )
        REPORT_PARTIAL_RESPONSES: bool = Field(
            default=False, 
            description="Report incomplete/interrupted responses"
        )
        
        # Advanced Options
        DEBUG_LOGGING: bool = Field(
            default=False, 
            description="Enable detailed debug logging"
        )
        MAX_RESPONSE_SIZE_MB: float = Field(
            default=10.0, 
            description="Maximum response size to capture (MB)"
        )
        CUSTOM_METADATA: str = Field(
            default="{}", 
            description="Additional custom metadata as JSON string"
        )
        TIMEOUT_SECONDS: int = Field(
            default=30, 
            description="Timeout for model calls and OpenPipe reporting"
        )

    def __init__(self):
        self.valves = self.Valves()
        
        # Log initialization
        self._log("OpenPipe Raw Capture Function v1.0 initialized successfully")
        self._log("Function will act as a proxy to capture raw model responses")
        
    def _log(self, message: str, level: str = "INFO"):
        """Enhanced logging with timestamps and levels"""
        if self.valves.DEBUG_LOGGING or level in ["ERROR", "WARNING"]:
            import traceback
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [OpenPipe Function v1.0 {level}] {message}")
            if level == "ERROR":
                print(f"[{timestamp}] [OpenPipe Function v1.0 TRACE] {traceback.format_exc()}")

    def pipes(self):
        """
        Define available models for this function
        
        Creates model entries that appear in OpenWebUI's model selector.
        Each entry acts as a proxy to the target model while capturing raw responses.
        """
        if not self.valves.TARGET_MODEL:
            return [{
                "id": "openpipe-setup-required",
                "name": f"{self.valves.FUNCTION_NAME_PREFIX}⚠️ Configure TARGET_MODEL in settings"
            }]
        
        return [{
            "id": f"openpipe-capture-{self.valves.TARGET_MODEL.replace(':', '-').replace('/', '-')}",
            "name": f"{self.valves.FUNCTION_NAME_PREFIX}{self.valves.TARGET_MODEL}"
        }]
    
    def _get_custom_metadata(self) -> Dict[str, Any]:
        """Parse and validate custom metadata from JSON string"""
        try:
            metadata = json.loads(self.valves.CUSTOM_METADATA)
            if not isinstance(metadata, dict):
                self._log("Custom metadata must be a JSON object", "WARNING")
                return {}
            return metadata
        except (json.JSONDecodeError, TypeError) as e:
            self._log(f"Invalid custom metadata JSON: {e}", "WARNING")
            return {}
    
    def _classify_interaction(self, body: Dict[str, Any]) -> str:
        """
        Classify the type of interaction based on request content
        
        Args:
            body: Request payload
            
        Returns:
            Classification string
        """
        if not self.valves.INCLUDE_INTERACTION_METADATA:
            return "unknown"
        
        # Check metadata for explicit task classification
        metadata = body.get("metadata", {})
        if "task" in metadata:
            return metadata["task"]
        
        # Analyze message content for classification
        messages = body.get("messages", [])
        if not messages:
            return "empty_request"
        
        # Check for tool/function calls
        if self._has_tool_calls(body):
            return "function_calling"
        
        # Check for system prompts indicating specific tasks
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            role = message.get("role", "")
            content = str(message.get("content", "")).lower()
            
            if role == "system":
                # Classify based on common system prompt patterns
                if any(keyword in content for keyword in ["title", "generate a title"]):
                    return "title_generation"
                elif any(keyword in content for keyword in ["tag", "categorize", "classify"]):
                    return "tags_generation"
                elif any(keyword in content for keyword in ["emoji", "emoticon"]):
                    return "emoji_generation"
                elif any(keyword in content for keyword in ["follow up", "follow-up", "next question"]):
                    return "follow_up_generation"
                elif any(keyword in content for keyword in ["search", "query", "find"]):
                    return "query_generation"
                elif any(keyword in content for keyword in ["image", "picture", "visual"]):
                    return "image_prompt_generation"
        
        # Default classification
        return "chat_message"
    
    def _has_tool_calls(self, body: Dict[str, Any]) -> bool:
        """Check if the request involves tool/function calls"""
        # Check for tools definition
        if body.get("tools") or body.get("functions"):
            return True
        
        # Check messages for tool calls
        messages = body.get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            # Check for various tool call formats
            if any(key in message for key in ["tool_calls", "function_call", "tool_use"]):
                return True
                
            # Check content for tool usage indicators
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["tool_use", "function_call"]:
                        return True
        
        return False
    
    def _extract_tools_used(self, body: Dict[str, Any]) -> List[str]:
        """Extract list of tools/functions used in the request"""
        tools_used = []
        
        # Extract from tools definition
        for tool in body.get("tools", []):
            if isinstance(tool, dict):
                name = tool.get("name") or tool.get("function", {}).get("name")
                if name:
                    tools_used.append(name)
        
        # Extract from function definitions
        for func in body.get("functions", []):
            if isinstance(func, dict) and "name" in func:
                tools_used.append(func["name"])
        
        # Extract from messages
        messages = body.get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            # Tool calls in message
            for tool_call in message.get("tool_calls", []):
                if isinstance(tool_call, dict):
                    name = tool_call.get("function", {}).get("name") or tool_call.get("name")
                    if name:
                        tools_used.append(name)
            
            # Function call in message
            function_call = message.get("function_call", {})
            if isinstance(function_call, dict) and "name" in function_call:
                tools_used.append(function_call["name"])
                
            # Tool use in content
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        name = item.get("name")
                        if name:
                            tools_used.append(name)
        
        return list(set(tools_used))  # Remove duplicates
    
    def _should_report_response(self, response_data: Dict[str, Any], status_code: int) -> bool:
        """Determine if this response should be reported based on configuration"""
        if status_code >= 200 and status_code < 300:
            return self.valves.REPORT_SUCCESSFUL_COMPLETIONS
        elif status_code >= 400:
            return self.valves.REPORT_ERROR_RESPONSES
        else:
            return self.valves.REPORT_PARTIAL_RESPONSES
    
    def _check_response_size(self, response_data: Any) -> bool:
        """Check if response size is within limits"""
        try:
            response_str = json.dumps(response_data) if not isinstance(response_data, str) else response_data
            size_mb = len(response_str.encode('utf-8')) / (1024 * 1024)
            
            if size_mb > self.valves.MAX_RESPONSE_SIZE_MB:
                self._log(f"Response size ({size_mb:.2f}MB) exceeds limit ({self.valves.MAX_RESPONSE_SIZE_MB}MB)", "WARNING")
                return False
                
            return True
        except Exception as e:
            self._log(f"Error checking response size: {e}", "WARNING")
            return True  # Allow by default if check fails
    
    async def _send_to_openpipe(self, payload: Dict[str, Any]) -> None:
        """Send captured data to OpenPipe API"""
        if not self.valves.OPENPIPE_API_KEY:
            self._log("OpenPipe API key not configured - skipping report", "WARNING")
            return
            
        url = f"{self.valves.OPENPIPE_BASE_URL}/report"
        headers = {
            "Authorization": f"Bearer {self.valves.OPENPIPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            self._log(f"Sending raw capture report to OpenPipe")
            self._log(f"Payload structure: {list(payload.keys())}")
            
            # Validate required fields
            required_fields = ["requestedAt", "receivedAt", "reqPayload", "respPayload", "statusCode"]
            missing_fields = [field for field in required_fields if field not in payload]
            if missing_fields:
                self._log(f"Missing required payload fields: {missing_fields}", "ERROR")
                return
            
            # Async HTTP request using urllib in executor
            import urllib.request
            import urllib.parse
            import urllib.error
            
            loop = asyncio.get_event_loop()
            
            def make_request():
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=headers)
                
                try:
                    with urllib.request.urlopen(req, timeout=self.valves.TIMEOUT_SECONDS) as response:
                        return response.getcode(), response.read().decode('utf-8')
                except urllib.error.HTTPError as e:
                    return e.code, e.read().decode('utf-8')
                except urllib.error.URLError as e:
                    raise Exception(f"Network error: {str(e)}")
            
            status_code, response_text = await loop.run_in_executor(None, make_request)
            
            if status_code == 200:
                self._log("✅ Successfully sent raw capture to OpenPipe")
                try:
                    response_data = json.loads(response_text)
                    if self.valves.DEBUG_LOGGING:
                        self._log(f"OpenPipe response: {response_data}")
                except json.JSONDecodeError:
                    self._log("OpenPipe response received but not valid JSON")
            else:
                self._log(f"❌ OpenPipe API error: {status_code} - {response_text}", "ERROR")
                
        except Exception as e:
            self._log(f"❌ Failed to send to OpenPipe: {str(e)}", "ERROR")

    async def pipe(
        self, 
        body: Dict[str, Any], 
        __user__: Optional[Dict[str, Any]] = None, 
        __request__: Optional[Request] = None,
        __event_emitter__ = None
    ) -> Union[str, Dict[str, Any], AsyncGenerator]:
        """
        Main pipe function that proxies model calls while capturing raw responses
        
        Args:
            body: Chat completion request payload
            __user__: User information 
            __request__: FastAPI request object
            __event_emitter__: Optional event emitter for status updates
            
        Returns:
            Raw model response (string, dict, or async generator for streaming)
        """
        
        # Validate configuration
        if not self.valves.TARGET_MODEL:
            error_msg = "TARGET_MODEL not configured. Please set the target model in function settings."
            self._log(error_msg, "ERROR")
            return f"❌ Configuration Error: {error_msg}"
        
        # Initialize timing
        request_start = time.perf_counter()
        requested_at = int(time.time() * 1000)
        
        # Generate request ID for tracking
        request_id = f"raw_capture_{time.time()}_{id(body)}"
        
        try:
            # Emit status if available
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status", 
                    "data": {
                        "description": f"🔍 Capturing raw response from {self.valves.TARGET_MODEL}",
                        "done": False
                    }
                })
            
            # Classify interaction for analytics
            interaction_type = self._classify_interaction(body)
            tools_used = self._extract_tools_used(body)
            
            self._log(f"Processing {interaction_type} request with {len(tools_used)} tools: {tools_used}")
            
            # Import OpenWebUI's internal chat completion function
            try:
                from open_webui.utils.chat import generate_chat_completion
                from open_webui.models.users import Users
            except ImportError as e:
                error_msg = f"Failed to import OpenWebUI modules: {e}"
                self._log(error_msg, "ERROR")
                return f"❌ Import Error: {error_msg}"
            
            # Get user object if available
            user_obj = None
            if __user__ and isinstance(__user__, dict):
                try:
                    user_obj = Users.get_user_by_id(__user__.get("id"))
                except Exception as e:
                    self._log(f"Could not fetch user object: {e}", "WARNING")
                    user_obj = __user__  # Fallback to dict
            
            # Update body to use target model
            modified_body = body.copy()
            modified_body["model"] = self.valves.TARGET_MODEL
            
            self._log(f"Proxying to target model: {self.valves.TARGET_MODEL}")
            
            # Call the actual model via OpenWebUI's internal function
            try:
                raw_response = await generate_chat_completion(__request__ or Request, modified_body, user_obj)
            except Exception as e:
                error_msg = f"Model call failed: {str(e)}"
                self._log(error_msg, "ERROR")
                
                # Report error response to OpenPipe
                if self.valves.REPORT_ERROR_RESPONSES:
                    await self._report_error_response(
                        body, requested_at, request_start, interaction_type, tools_used, 
                        error_msg, __user__
                    )
                
                return f"❌ Model Error: {error_msg}"
            
            # Calculate timing
            received_at = int(time.time() * 1000)
            inference_time_ms = int((time.perf_counter() - request_start) * 1000)
            
            # Determine response type and status
            status_code = 200
            response_type = "unknown"
            
            if hasattr(raw_response, '__aiter__'):
                response_type = "streaming"
                # Handle streaming response
                return await self._handle_streaming_response(
                    raw_response, body, requested_at, received_at, inference_time_ms,
                    interaction_type, tools_used, __user__, __event_emitter__
                )
            elif isinstance(raw_response, dict):
                response_type = "json"
            elif isinstance(raw_response, str):
                response_type = "text"
            else:
                response_type = f"other_{type(raw_response).__name__}"
            
            self._log(f"Received {response_type} response ({inference_time_ms}ms)")
            
            # Check if we should report this response
            if self._should_report_response(raw_response, status_code):
                # Check response size limits
                if self._check_response_size(raw_response):
                    await self._report_completion_response(
                        body, raw_response, requested_at, received_at, inference_time_ms,
                        interaction_type, tools_used, status_code, __user__
                    )
                
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"✅ Raw response captured and reported ({inference_time_ms}ms)",
                            "done": True
                        }
                    })
            
            # Return the raw response unchanged
            return raw_response
            
        except Exception as e:
            error_msg = f"Unexpected error in pipe function: {str(e)}"
            self._log(error_msg, "ERROR")
            
            # Try to report the error
            try:
                if self.valves.REPORT_ERROR_RESPONSES:
                    inference_time_ms = int((time.perf_counter() - request_start) * 1000)
                    await self._report_error_response(
                        body, requested_at, request_start, "error", [], error_msg, __user__
                    )
            except Exception as report_error:
                self._log(f"Failed to report error: {report_error}", "ERROR")
            
            return f"❌ Function Error: {error_msg}"

    async def _handle_streaming_response(
        self, 
        stream_response, 
        original_body: Dict[str, Any],
        requested_at: int, 
        received_at: int, 
        inference_time_ms: int,
        interaction_type: str, 
        tools_used: List[str], 
        user: Optional[Dict[str, Any]], 
        event_emitter
    ) -> AsyncGenerator:
        """
        Handle streaming responses while capturing chunks for OpenPipe
        
        Args:
            stream_response: The streaming response from the model
            original_body: Original request payload
            requested_at: Request timestamp  
            received_at: Response timestamp
            inference_time_ms: Inference timing
            interaction_type: Classified interaction type
            tools_used: List of tools used
            user: User information
            event_emitter: Event emitter for status updates
            
        Yields:
            Streaming response chunks
        """
        captured_chunks = []
        full_content = ""
        chunk_count = 0
        
        try:
            async for chunk in stream_response:
                chunk_count += 1
                
                # Capture chunk if enabled
                if self.valves.CAPTURE_STREAMING_CHUNKS:
                    chunk_data = {
                        "chunk_index": chunk_count,
                        "timestamp": int(time.time() * 1000),
                        "content": chunk
                    }
                    captured_chunks.append(chunk_data)
                
                # Accumulate content for final reporting
                if isinstance(chunk, str):
                    full_content += chunk
                elif isinstance(chunk, dict):
                    # Try to extract content from structured chunk
                    if "choices" in chunk:
                        for choice in chunk["choices"]:
                            if "delta" in choice and "content" in choice["delta"]:
                                delta_content = choice["delta"]["content"]
                                if delta_content:
                                    full_content += str(delta_content)
                
                # Yield the original chunk unchanged
                yield chunk
            
            self._log(f"Streaming complete: {chunk_count} chunks, {len(full_content)} characters")
            
            # Report the complete streaming session
            if self.valves.REPORT_SUCCESSFUL_COMPLETIONS:
                await self._report_streaming_completion(
                    original_body, captured_chunks, full_content, requested_at, 
                    received_at, inference_time_ms, interaction_type, tools_used, user
                )
            
            if event_emitter:
                await event_emitter({
                    "type": "status",
                    "data": {
                        "description": f"✅ Streaming session captured: {chunk_count} chunks",
                        "done": True
                    }
                })
                
        except Exception as e:
            self._log(f"Error handling streaming response: {e}", "ERROR")
            
            # Report streaming error if we have partial data
            if self.valves.REPORT_ERROR_RESPONSES and captured_chunks:
                await self._report_streaming_error(
                    original_body, captured_chunks, str(e), requested_at, 
                    inference_time_ms, interaction_type, tools_used, user
                )

    async def _report_completion_response(
        self, 
        request_body: Dict[str, Any], 
        response_data: Any,
        requested_at: int, 
        received_at: int, 
        inference_time_ms: int,
        interaction_type: str, 
        tools_used: List[str], 
        status_code: int,
        user: Optional[Dict[str, Any]]
    ):
        """Report a completed non-streaming response to OpenPipe"""
        
        try:
            # Build comprehensive metadata
            metadata = {
                "interaction_type": interaction_type,
                "capture_method": "function_raw_pre_processing",
                "openwebui_version": "function_v1.0_raw_capture",
                "response_type": "completion",
                "target_model": self.valves.TARGET_MODEL,
                **self._get_custom_metadata()
            }
            
            if self.valves.INCLUDE_REQUEST_TIMING:
                metadata.update({
                    "inference_time_ms": inference_time_ms,
                    "total_request_time_ms": received_at - requested_at
                })
            
            if self.valves.INCLUDE_INTERACTION_METADATA:
                metadata.update({
                    "tools_used": tools_used,
                    "has_tool_calls": len(tools_used) > 0,
                    "message_count": len(request_body.get("messages", [])),
                    "user_id": user.get("id") if user and isinstance(user, dict) else None
                })
            
            # Analyze tokens if enabled
            if self.valves.INCLUDE_TOKEN_ANALYSIS:
                token_stats = self._analyze_tokens(request_body, response_data)
                metadata.update(token_stats)
            
            # Format request payload (remove function-specific metadata)
            req_payload = request_body.copy()
            if "metadata" in req_payload:
                req_payload["metadata"] = {k: v for k, v in req_payload["metadata"].items() 
                                         if not k.startswith("openpipe_")}
            
            # Prepare response payload  
            if self.valves.PRESERVE_RAW_RESPONSE_FORMAT:
                # Keep original format
                resp_payload = response_data
            else:
                # Normalize to OpenAI format
                resp_payload = self._normalize_response_format(response_data, request_body)
            
            # Build OpenPipe payload
            openpipe_payload = {
                "requestedAt": requested_at,
                "receivedAt": received_at,
                "reqPayload": req_payload,
                "respPayload": resp_payload,
                "statusCode": status_code,
                "metadata": metadata
            }
            
            # Send to OpenPipe
            await self._send_to_openpipe(openpipe_payload)
            
        except Exception as e:
            self._log(f"Error reporting completion response: {e}", "ERROR")

    async def _report_streaming_completion(
        self,
        request_body: Dict[str, Any],
        captured_chunks: List[Dict[str, Any]],
        full_content: str,
        requested_at: int,
        received_at: int,
        inference_time_ms: int,
        interaction_type: str,
        tools_used: List[str],
        user: Optional[Dict[str, Any]]
    ):
        """Report a completed streaming response to OpenPipe"""
        try:
            # Build metadata for streaming response
            metadata = {
                "interaction_type": interaction_type,
                "capture_method": "function_raw_streaming",
                "openwebui_version": "function_v1.0_raw_capture",
                "response_type": "streaming",
                "target_model": self.valves.TARGET_MODEL,
                "chunk_count": len(captured_chunks),
                "total_content_length": len(full_content),
                **self._get_custom_metadata()
            }
            
            if self.valves.INCLUDE_REQUEST_TIMING:
                metadata.update({
                    "inference_time_ms": inference_time_ms,
                    "total_request_time_ms": received_at - requested_at
                })
            
            if self.valves.INCLUDE_INTERACTION_METADATA:
                metadata.update({
                    "tools_used": tools_used,
                    "has_tool_calls": len(tools_used) > 0,
                    "message_count": len(request_body.get("messages", [])),
                    "user_id": user.get("id") if user and isinstance(user, dict) else None
                })
            
            # Format request payload
            req_payload = request_body.copy()
            if "metadata" in req_payload:
                req_payload["metadata"] = {k: v for k, v in req_payload["metadata"].items() 
                                         if not k.startswith("openpipe_")}
            
            # Create streaming response payload
            resp_payload = {
                "id": f"chatcmpl-stream-{int(time.time())}",
                "object": "chat.completion",
                "created": int(received_at / 1000),
                "model": self.valves.TARGET_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": full_content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "streaming_data": {
                    "chunks": captured_chunks if self.valves.CAPTURE_STREAMING_CHUNKS else [],
                    "chunk_count": len(captured_chunks)
                }
            }
            
            # Build OpenPipe payload
            openpipe_payload = {
                "requestedAt": requested_at,
                "receivedAt": received_at,
                "reqPayload": req_payload,
                "respPayload": resp_payload,
                "statusCode": 200,
                "metadata": metadata
            }
            
            # Send to OpenPipe
            await self._send_to_openpipe(openpipe_payload)
            
        except Exception as e:
            self._log(f"Error reporting streaming completion: {e}", "ERROR")

    async def _report_streaming_error(
        self,
        request_body: Dict[str, Any],
        captured_chunks: List[Dict[str, Any]],
        error_message: str,
        requested_at: int,
        inference_time_ms: int,
        interaction_type: str,
        tools_used: List[str],
        user: Optional[Dict[str, Any]]
    ):
        """Report a streaming error to OpenPipe"""
        try:
            received_at = int(time.time() * 1000)
            
            metadata = {
                "interaction_type": interaction_type,
                "capture_method": "function_raw_streaming_error",
                "openwebui_version": "function_v1.0_raw_capture",
                "response_type": "streaming_error",
                "target_model": self.valves.TARGET_MODEL,
                "error_message": error_message,
                "partial_chunk_count": len(captured_chunks),
                **self._get_custom_metadata()
            }
            
            if self.valves.INCLUDE_REQUEST_TIMING:
                metadata["inference_time_ms"] = inference_time_ms
            
            if self.valves.INCLUDE_INTERACTION_METADATA:
                metadata.update({
                    "tools_used": tools_used,
                    "has_tool_calls": len(tools_used) > 0,
                    "user_id": user.get("id") if user and isinstance(user, dict) else None
                })
            
            # Format request payload
            req_payload = request_body.copy()
            
            # Error response payload
            resp_payload = {
                "error": {
                    "message": error_message,
                    "type": "streaming_error",
                    "partial_data": {
                        "chunks": captured_chunks[:10],  # Only include first 10 chunks to avoid size issues
                        "total_chunks": len(captured_chunks)
                    }
                }
            }
            
            openpipe_payload = {
                "requestedAt": requested_at,
                "receivedAt": received_at,
                "reqPayload": req_payload,
                "respPayload": resp_payload,
                "statusCode": 500,
                "metadata": metadata
            }
            
            await self._send_to_openpipe(openpipe_payload)
            
        except Exception as e:
            self._log(f"Error reporting streaming error: {e}", "ERROR")

    async def _report_error_response(
        self,
        request_body: Dict[str, Any],
        requested_at: int,
        request_start: float,
        interaction_type: str,
        tools_used: List[str],
        error_message: str,
        user: Optional[Dict[str, Any]]
    ):
        """Report an error response to OpenPipe"""
        try:
            received_at = int(time.time() * 1000)
            inference_time_ms = int((time.perf_counter() - request_start) * 1000)
            
            metadata = {
                "interaction_type": interaction_type,
                "capture_method": "function_raw_error",
                "openwebui_version": "function_v1.0_raw_capture",
                "response_type": "error",
                "target_model": self.valves.TARGET_MODEL,
                "error_message": error_message,
                **self._get_custom_metadata()
            }
            
            if self.valves.INCLUDE_REQUEST_TIMING:
                metadata["inference_time_ms"] = inference_time_ms
            
            if self.valves.INCLUDE_INTERACTION_METADATA:
                metadata.update({
                    "tools_used": tools_used,
                    "has_tool_calls": len(tools_used) > 0,
                    "user_id": user.get("id") if user and isinstance(user, dict) else None
                })
            
            req_payload = request_body.copy()
            
            resp_payload = {
                "error": {
                    "message": error_message,
                    "type": "function_error"
                }
            }
            
            openpipe_payload = {
                "requestedAt": requested_at,
                "receivedAt": received_at,
                "reqPayload": req_payload,
                "respPayload": resp_payload,
                "statusCode": 500,
                "metadata": metadata
            }
            
            await self._send_to_openpipe(openpipe_payload)
            
        except Exception as e:
            self._log(f"Error reporting error response: {e}", "ERROR")

    def _analyze_tokens(self, request_body: Dict[str, Any], response_data: Any) -> Dict[str, Any]:
        """Analyze token usage patterns"""
        try:
            token_stats = {}
            
            # Analyze request tokens
            messages = request_body.get("messages", [])
            total_request_chars = sum(len(str(msg.get("content", ""))) for msg in messages if isinstance(msg, dict))
            token_stats["estimated_request_tokens"] = total_request_chars // 4  # Rough estimate
            
            # Analyze response tokens
            if isinstance(response_data, dict):
                if "usage" in response_data:
                    token_stats.update({
                        "actual_prompt_tokens": response_data["usage"].get("prompt_tokens", 0),
                        "actual_completion_tokens": response_data["usage"].get("completion_tokens", 0),
                        "actual_total_tokens": response_data["usage"].get("total_tokens", 0)
                    })
                
                # Extract response content for analysis
                response_content = ""
                if "choices" in response_data:
                    for choice in response_data["choices"]:
                        if "message" in choice and "content" in choice["message"]:
                            response_content += str(choice["message"]["content"])
                
                token_stats["estimated_response_tokens"] = len(response_content) // 4
                token_stats["response_content_length"] = len(response_content)
                
            elif isinstance(response_data, str):
                token_stats["estimated_response_tokens"] = len(response_data) // 4
                token_stats["response_content_length"] = len(response_data)
            
            return token_stats
            
        except Exception as e:
            self._log(f"Error analyzing tokens: {e}", "ERROR")
            return {}

    def _normalize_response_format(self, response_data: Any, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize response to OpenAI chat completion format"""
        try:
            if isinstance(response_data, dict):
                # Already in structured format, ensure OpenAI compatibility
                normalized = {
                    "id": response_data.get("id", f"chatcmpl-{int(time.time())}"),
                    "object": "chat.completion",
                    "created": response_data.get("created", int(time.time())),
                    "model": self.valves.TARGET_MODEL,
                    "choices": response_data.get("choices", []),
                    "usage": response_data.get("usage", {})
                }
                
                # Ensure choices have proper format
                if not normalized["choices"]:
                    normalized["choices"] = [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": str(response_data.get("content", response_data))
                            },
                            "finish_reason": "stop"
                        }
                    ]
                
                return normalized
                
            elif isinstance(response_data, str):
                # Convert string response to OpenAI format
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.valves.TARGET_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": response_data
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": len(response_data) // 4,
                        "total_tokens": len(response_data) // 4
                    }
                }
            
            else:
                # Fallback for other response types
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.valves.TARGET_MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": str(response_data)
                            },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {}
                }
                
        except Exception as e:
            self._log(f"Error normalizing response format: {e}", "ERROR")
            return {
                "error": f"Failed to normalize response: {str(e)}",
                "original_response": str(response_data)[:1000]  # Truncate for safety
            }
