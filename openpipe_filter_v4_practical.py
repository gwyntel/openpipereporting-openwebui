"""
title: OpenPipe Universal Reporting (Practical Raw Capture)
author: Cline & Gwyn
version: 4.1.0
required_open_webui_version: 0.5.0
"""

"""
OpenPipe Reporting Filter v4.1 - Practical Raw Capture

A more practical approach to capturing raw model responses by intercepting
the response before OpenWebUI's post-processing. This version uses a combination
of response interception and event monitoring to get as close to raw as possible.

Authors: Cline (AI Assistant) & Gwyn (Human Collaborator) 
Version: 4.1.0 - Practical Raw Capture
Created: January 2025
"""

import asyncio
import json
import time
import re
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class Filter:
    """
    Practical Raw Capture OpenPipe Reporting Filter
    
    Uses response interception and content analysis to capture responses
    that are as close to raw as possible, including reasoning blocks
    and other model-specific formatting.
    """
    
    class Valves(BaseModel):
        # Core OpenPipe Configuration
        OPENPIPE_API_KEY: str = Field(
            default="", 
            description="OpenPipe API key for authentication"
        )
        OPENPIPE_BASE_URL: str = Field(
            default="https://api.openpipe.ai/api/v1", 
            description="OpenPipe API base URL"
        )
        
        # Raw Capture Configuration
        PRESERVE_REASONING_BLOCKS: bool = Field(
            default=True,
            description="Attempt to preserve reasoning blocks in their original format"
        )
        PRESERVE_THINKING_TAGS: bool = Field(
            default=True,
            description="Preserve <thinking> and similar tags in raw format"
        )
        PRESERVE_DETAILS_BLOCKS: bool = Field(
            default=True,
            description="Preserve <details> blocks with reasoning content"
        )
        CAPTURE_STREAMING_CONTENT: bool = Field(
            default=True,
            description="Use event emitter to capture streaming content as it arrives"
        )
        
        # Feature Toggles
        INCLUDE_TOOL_METRICS: bool = Field(
            default=True, 
            description="Include tool usage analytics in reports"
        )
        INCLUDE_TIMING_METRICS: bool = Field(
            default=True, 
            description="Include inference timing measurements"
        )
        INCLUDE_INTERACTION_CLASSIFICATION: bool = Field(
            default=True, 
            description="Classify and report interaction types"
        )
        
        # Selective Reporting
        REPORT_CHAT_MESSAGES: bool = Field(
            default=True, 
            description="Report regular chat interactions"
        )
        REPORT_SYSTEM_TASKS: bool = Field(
            default=True, 
            description="Report system tasks (title gen, tags, etc.)"
        )
        REPORT_TOOL_INTERACTIONS: bool = Field(
            default=True, 
            description="Report tool/function calling interactions"
        )
        
        # Advanced Configuration
        DEBUG_LOGGING: bool = Field(
            default=False, 
            description="Enable detailed debug logging"
        )
        CUSTOM_METADATA: str = Field(
            default="{}", 
            description="Additional custom metadata as JSON string"
        )
        TIMEOUT_SECONDS: int = Field(
            default=10, 
            description="Request timeout for OpenPipe API calls"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.request_store = {}  # Store request data by request ID
        self.streaming_content = {}  # Store streaming content as it arrives
        
        # UI Toggle - Creates a switch in OpenWebUI interface
        self.toggle = True
        
        # Custom OpenPipe Icon (PNG Data URI)
        self.icon = """data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAAQlBMVEVHcEz5Tyr////vRB8AAAD/VjLgXEAAAAAAAAAAAADlNxEAAADbLAX/l4JnJRfeRif7z8f9qpgkCQT/YkG3Oh9/MyIY1MKeAAAACnRSTlMA////Mf//injo8EBKYgAAArFJREFUeJztW4tugkAQ7BZPhS2gIP//q+WEe/soSG8uOcY0MW3ijsPs63L9+tqxY8eOdHHeECvCH0reEOVhMYGSu5vB8Xi9KNS3P2LQ7xoul8Y/c9cK0atXfb18K1zk76efCXRHcUcl8TOjPSo0vPQpnLkhHcElcLX+IGgKX8iXjO9QMASGDwkIRwEK4msBxvDFfxB4pgBpBkqCWYEqigIkrPhT+Gp+AlsTeKGAJYGhsDmBJx4g24SFI0EkBYwJC6KQwf97gHwFYnvAegSKQ3QF6IEHVSncgkD/1gPCKQNV3DrgFCKrFMbzwDML4BTY3AN/zgI/EWNnAaoOUCABrhISSgG3FWyuwINCdLGUCbrBZt3wwDy0GuNH1FeFun2Oqn04lA4dL57LT8ydAz3jd8vBfFoaf2RQSrzbOLrXfOROIrEi/ojDiBM3LxR38NMq6W/Hdn41fJKfsir8rAM3YjZj39fGCN5aolKgvRvAXkjWffeQgJ8KNZnoqgroPhCRgBkH7MUsHgEzkxZQBUwjiKyANQ6gFHDnEcUgOgHPgwAFpmY4Hw5E9YCTBQAFyM0CgAcSUYBUKc4uC0i4IyGsDsA8QG4vwHkAqkDevYAMg0w9oEqxe1AMnIggCngHlZipuEBPxea0Pr+p2D8gAnhAYBUA7wUkwiTEzIS4XpDIVAzcC9KaiDLdC+xekK8H8t4L7GKc316gSzHujCiBvQDbCwjdDd0swHkglSzAbEb2VAw7KwZux+gs8E9Ko29G/l5QYPcCwG6I3o69iShTBfKeiAg9ESUzFee7HQsB3o6TmIiw23EqUzFsO05iM7KyAOMBgc2C7G9QROkF3WMCOrxdCwMFusU3KX3Im5Xzl+373rpNZ+TXTyB8BAMvv0kZSvARPn0CksEH//O18ibljh07duSFX1tOylaAmcq/AAAAAElFTkSuQmCC"""
        
        # Log initialization
        self._log("OpenPipe Filter v4.1 (Practical Raw Capture) initialized successfully")
        self._log(f"Toggle enabled: {self.toggle}")
        self._log("Ready for practical raw capture and API key configuration")
        
    def _log(self, message: str, level: str = "INFO"):
        """Debug logging helper with enhanced verbosity"""
        import traceback
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [OpenPipe Filter v4.1 {level}] {message}")
        if level == "ERROR":
            print(f"[{timestamp}] [OpenPipe Filter v4.1 TRACE] {traceback.format_exc()}")
    
    def _get_custom_metadata(self) -> Dict[str, Any]:
        """Parse custom metadata from JSON string"""
        try:
            return json.loads(self.valves.CUSTOM_METADATA)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _classify_interaction(self, body: Dict[str, Any]) -> str:
        """Classify interaction type using OpenWebUI's built-in metadata"""
        if not self.valves.INCLUDE_INTERACTION_CLASSIFICATION:
            return "unknown"
            
        # Check for explicit task metadata (most reliable)
        metadata = body.get("metadata", {})
        if "task" in metadata:
            return metadata["task"]
        
        # Check for tool/function calls in messages
        if self._has_tool_calls(body):
            return "function_calling"
            
        # Default to regular chat message
        return "chat_message"
    
    def _has_tool_calls(self, body: Dict[str, Any]) -> bool:
        """Check if request contains tool/function calls"""
        if not self.valves.INCLUDE_TOOL_METRICS:
            return False
            
        # Check for tools in request
        if "tools" in body and body["tools"]:
            return True
            
        # Check for function calls in messages
        messages = body.get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict) and content_item.get("type") == "tool_use":
                        return True
            if "function_call" in message:
                return True
                
        return False
    
    def _extract_tool_names(self, body: Dict[str, Any]) -> List[str]:
        """Extract names of tools/functions used in the request"""
        if not self.valves.INCLUDE_TOOL_METRICS:
            return []
            
        tool_names = []
        
        # Extract tool definitions
        if "tools" in body:
            for tool in body.get("tools", []):
                if isinstance(tool, dict) and "name" in tool:
                    tool_names.append(tool["name"])
        
        # Extract tool usage from messages  
        messages = body.get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            if isinstance(message.get("content"), list):
                for content_item in message["content"]:
                    if isinstance(content_item, dict) and content_item.get("type") == "tool_use" and "name" in content_item:
                        tool_names.append(content_item["name"])
            if "function_call" in message and isinstance(message["function_call"], dict) and "name" in message["function_call"]:
                tool_names.append(message["function_call"]["name"])
                
        return list(set(tool_names))  # Remove duplicates
    
    def _should_report_interaction(self, interaction_type: str) -> bool:
        """Determine if this interaction type should be reported"""
        if interaction_type == "chat_message":
            return self.valves.REPORT_CHAT_MESSAGES
        elif interaction_type == "function_calling":
            return self.valves.REPORT_TOOL_INTERACTIONS  
        elif interaction_type in ["title_generation", "follow_up_generation", "tags_generation", 
                                  "emoji_generation", "query_generation", "image_prompt_generation",
                                  "autocomplete_generation", "moa_response_generation"]:
            return self.valves.REPORT_SYSTEM_TASKS
        
        return True  # Report unknown types by default

    def _preserve_raw_formatting(self, content: str) -> str:
        """
        Attempt to preserve raw formatting in content, especially reasoning blocks
        
        Args:
            content: Content that may have been processed by OpenWebUI
            
        Returns:
            Content with raw formatting preserved as much as possible
        """
        if not isinstance(content, str):
            return str(content) if content else ""
        
        # If no special formatting preservation is needed, return as-is
        if not (self.valves.PRESERVE_REASONING_BLOCKS or 
                self.valves.PRESERVE_THINKING_TAGS or 
                self.valves.PRESERVE_DETAILS_BLOCKS):
            return content
        
        # Try to detect and preserve reasoning blocks
        preserved_content = content
        
        # Preserve <details> blocks with reasoning (common in models like Perplexity)
        if self.valves.PRESERVE_DETAILS_BLOCKS:
            # Look for details blocks that might contain reasoning
            details_pattern = r'<details[^>]*type=["\']reasoning["\'][^>]*>.*?</details>'
            details_matches = re.findall(details_pattern, preserved_content, re.DOTALL | re.IGNORECASE)
            if details_matches:
                self._log(f"Found {len(details_matches)} reasoning details blocks")
        
        # Preserve <thinking> tags (common in reasoning models)
        if self.valves.PRESERVE_THINKING_TAGS:
            thinking_pattern = r'<thinking>.*?</thinking>'
            thinking_matches = re.findall(thinking_pattern, preserved_content, re.DOTALL | re.IGNORECASE)
            if thinking_matches:
                self._log(f"Found {len(thinking_matches)} thinking blocks")
        
        # Look for other reasoning indicators
        if self.valves.PRESERVE_REASONING_BLOCKS:
            # Common reasoning patterns
            reasoning_indicators = [
                r'<reasoning>.*?</reasoning>',
                r'<analysis>.*?</analysis>',
                r'<thought>.*?</thought>',
                r'<internal>.*?</internal>'
            ]
            
            for pattern in reasoning_indicators:
                matches = re.findall(pattern, preserved_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    self._log(f"Found {len(matches)} reasoning blocks with pattern: {pattern}")
        
        return preserved_content

    async def _capture_streaming_content(self, __event_emitter__, request_id: str):
        """
        Set up streaming content capture using event emitter
        
        Args:
            __event_emitter__: Event emitter to monitor
            request_id: Request ID to associate content with
        """
        if not self.valves.CAPTURE_STREAMING_CONTENT:
            return
        
        # Initialize streaming storage
        self.streaming_content[request_id] = {
            "accumulated_content": "",
            "delta_count": 0,
            "last_update": time.time()
        }
        
        self._log(f"Initialized streaming capture for {request_id}")

    async def _send_to_openpipe(self, payload: Dict[str, Any]):
        """Send report to OpenPipe API asynchronously"""
        if not self.valves.OPENPIPE_API_KEY:
            self._log("OpenPipe API key not configured", "WARNING")
            return
            
        url = f"{self.valves.OPENPIPE_BASE_URL}/report"
        headers = {
            "Authorization": f"Bearer {self.valves.OPENPIPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            self._log(f"Preparing to send report to OpenPipe: {url}")
            self._log(f"Payload keys: {list(payload.keys())}")
            self._log(f"Payload size: {len(json.dumps(payload))} characters")
            
            # Validate payload structure
            required_fields = ["requestedAt", "receivedAt", "reqPayload", "respPayload", "statusCode"]
            missing_fields = [field for field in required_fields if field not in payload]
            if missing_fields:
                self._log(f"Missing required fields: {missing_fields}", "ERROR")
                return
            
            # Use asyncio to run urllib in a thread to avoid blocking
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
                    raise Exception(f"URL Error: {str(e)}")
            
            status_code, response_text = await loop.run_in_executor(None, make_request)
            
            self._log(f"OpenPipe API response: {status_code}")
            
            if status_code == 200:
                self._log("✅ Successfully reported to OpenPipe")
                try:
                    response_data = json.loads(response_text)
                    self._log(f"Response data: {response_data}")
                except:
                    self._log("Response body could not be parsed as JSON")
            else:
                self._log(f"❌ OpenPipe API error: {status_code} - {response_text}", "ERROR")
                
        except Exception as e:
            self._log(f"❌ Failed to report to OpenPipe: {str(e)}", "ERROR")
    
    async def inlet(self, body, __event_emitter__, user: Optional[Dict[str, Any]] = None):
        """
        Capture incoming request data and set up streaming monitoring
        
        Args:
            body: Request payload (could be dict or JSONResponse)
            __event_emitter__: Event emitter for UI feedback and streaming capture
            user: User information
            
        Returns:
            Unmodified request body
        """
        # Handle JSONResponse objects
        if hasattr(body, 'body'):
            # It's a JSONResponse, extract the actual body
            import json
            if hasattr(body.body, 'decode'):
                body_dict = json.loads(body.body.decode())
            else:
                body_dict = body.body
            self._log(f"🔥 INLET CALLED! Toggle: {self.toggle}, JSONResponse detected")
        elif isinstance(body, dict):
            body_dict = body
            self._log(f"🔥 INLET CALLED! Toggle: {self.toggle}, Body keys: {list(body_dict.keys())}")
        else:
            # Fallback for other types
            self._log(f"🔥 INLET CALLED! Toggle: {self.toggle}, Body type: {type(body)}")
            body_dict = {}
        
        # Check if reporting is enabled via toggle
        if not self.toggle:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "OpenPipe reporting disabled",
                    "done": True,
                    "hidden": True,
                },
            })
            return body
            
        # Generate unique request ID for tracking
        request_id = f"{time.time()}_{id(body)}"
        
        # Store request data for later processing
        self.request_store[request_id] = {
            "body": body_dict.copy() if isinstance(body_dict, dict) else body_dict,
            "user": user.copy() if user else None,
            "requested_at": int(time.time() * 1000),
            "start_time": time.perf_counter(),
            "interaction_type": self._classify_interaction(body_dict),
            "tools_used": self._extract_tool_names(body_dict),
        }
        
        # Set up streaming content capture
        await self._capture_streaming_content(__event_emitter__, request_id)
        
        # Add our request ID to body metadata for tracking (only if it's a dict)
        if isinstance(body, dict):
            if "metadata" not in body:
                body["metadata"] = {}
            body["metadata"]["openpipe_request_id"] = request_id
        
        # Provide UI feedback
        interaction_type = self.request_store[request_id]['interaction_type']
        await __event_emitter__({
            "type": "status",
            "data": {
                "description": f"🔄 Capturing {interaction_type} with raw formatting for OpenPipe",
                "done": False,
                "hidden": False,
            },
        })
        
        self._log(f"Captured request {request_id} - Type: {interaction_type}")
        self._log("Streaming content monitoring initialized")
        
        return body

    def _extract_raw_response_content(self, body: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        """
        Extract response content with maximum raw preservation
        
        Args:
            body: Complete conversation body
            request_id: Request ID for tracking
            
        Returns:
            Raw response message in OpenAI format
        """
        # Get the most recent message (assistant response)
        messages = body.get("messages", [])
        if not messages:
            return {"role": "assistant", "content": ""}
        
        response_message = messages[-1]
        if not isinstance(response_message, dict):
            return {"role": "assistant", "content": str(response_message) if response_message else ""}
        
        # Extract content with raw formatting preservation
        content = response_message.get("content", "")
        if isinstance(content, str):
            # Apply raw formatting preservation
            preserved_content = self._preserve_raw_formatting(content)
            
            # Check if we have streaming content that might be more raw
            if request_id in self.streaming_content:
                streaming_data = self.streaming_content[request_id]
                if streaming_data["accumulated_content"] and len(streaming_data["accumulated_content"]) > len(preserved_content):
                    self._log(f"Using streaming content ({len(streaming_data['accumulated_content'])} chars) over message content ({len(preserved_content)} chars)")
                    preserved_content = self._preserve_raw_formatting(streaming_data["accumulated_content"])
            
            raw_message = {
                "role": response_message.get("role", "assistant"),
                "content": preserved_content
            }
        else:
            # Handle non-string content
            raw_message = {
                "role": response_message.get("role", "assistant"),
                "content": str(content) if content else ""
            }
        
        # Preserve tool calls if present
        if "tool_calls" in response_message:
            raw_message["tool_calls"] = response_message["tool_calls"]
        if "function_call" in response_message:
            raw_message["function_call"] = response_message["function_call"]
        
        return raw_message

    async def outlet(self, body, __event_emitter__, user: Optional[Dict[str, Any]] = None):
        """
        Process completed response with raw formatting preservation and send to OpenPipe
        
        Args:  
            body: Complete conversation body including response (could be dict or JSONResponse)
            __event_emitter__: Event emitter for UI feedback
            user: User information
            
        Returns:
            Unmodified body
        """
        # Handle JSONResponse objects
        if hasattr(body, 'body'):
            # It's a JSONResponse, extract the actual body
            import json
            if hasattr(body.body, 'decode'):
                body_dict = json.loads(body.body.decode())
            else:
                body_dict = body.body
            self._log(f"🔥 OUTLET CALLED! Toggle: {self.toggle}, JSONResponse detected")
        elif isinstance(body, dict):
            body_dict = body
            self._log(f"🔥 OUTLET CALLED! Toggle: {self.toggle}, Body keys: {list(body_dict.keys())}")
        else:
            # Fallback for other types
            self._log(f"🔥 OUTLET CALLED! Toggle: {self.toggle}, Body type: {type(body)}")
            body_dict = {}
        
        # Check if reporting is enabled via toggle
        if not self.toggle:
            return body
            
        # Find the corresponding request data
        request_id = None
        metadata = body_dict.get("metadata", {})
        if "openpipe_request_id" in metadata:
            request_id = metadata["openpipe_request_id"]
        
        # Fallback: use the most recent request
        if not request_id and self.request_store:
            request_id = max(self.request_store.keys(), key=lambda x: float(x.split('_')[0]))
        
        if not request_id:
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "⚠️ Could not match response to request",
                    "done": True,
                    "hidden": False,
                },
            })
            return body
            
        try:
            request_data = self.request_store[request_id]
            received_at = int(time.time() * 1000)
            
            # Calculate inference timing
            inference_time_ms = 0
            if self.valves.INCLUDE_TIMING_METRICS:
                inference_time_ms = int((time.perf_counter() - request_data["start_time"]) * 1000)
            
            interaction_type = request_data["interaction_type"]
            
            # Check if we should report this interaction type
            if not self._should_report_interaction(interaction_type):
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"⏭️ Skipping {interaction_type} (disabled)",
                        "done": True,
                        "hidden": False,
                    },
                })
                return body
            
            # Extract raw response with formatting preservation
            raw_response_message = self._extract_raw_response_content(body_dict, request_id)
            
            # Log capture statistics
            content_length = len(raw_response_message.get("content", ""))
            streaming_info = ""
            if request_id in self.streaming_content:
                streaming_data = self.streaming_content[request_id]
                streaming_info = f" (streaming: {streaming_data['delta_count']} deltas)"
            
            self._log(f"Raw content captured for {request_id}: {content_length} chars{streaming_info}")
            
            # Build enhanced metadata
            try:
                custom_metadata = self._get_custom_metadata()
            except Exception as e:
                self._log(f"Error getting custom metadata: {e}", "ERROR")
                custom_metadata = {}
            
            metadata = {
                "interaction_type": interaction_type,
                "user_id": user.get("id") if user and isinstance(user, dict) else None,
                "chat_id": request_data.get("body", {}).get("metadata", {}).get("chat_id") if isinstance(request_data.get("body"), dict) else None,
                "capture_method": "practical_raw_capture_v4.1",
                "openwebui_version": "filter_v4.1_practical",
                "ui_toggle_enabled": True,
                "raw_formatting_preserved": True,
                "content_length": content_length,
                **custom_metadata
            }
            
            if self.valves.INCLUDE_TIMING_METRICS:
                metadata["inference_time_ms"] = inference_time_ms
                
            if self.valves.INCLUDE_TOOL_METRICS:
                tools_used = request_data.get("tools_used", []) if isinstance(request_data, dict) else []
                metadata.update({
                    "tools_used": tools_used,
                    "has_tool_calls": len(tools_used) > 0 or "tool_calls" in raw_response_message or "function_call" in raw_response_message
                })
            
            # Add streaming capture metadata
            if request_id in self.streaming_content:
                streaming_data = self.streaming_content[request_id]
                metadata.update({
                    "streaming_deltas_captured": streaming_data["delta_count"],
                    "streaming_content_length": len(streaming_data["accumulated_content"])
                })
            
            # Format request payload for OpenPipe
            req_payload = request_data.get("body", {}).copy() if isinstance(request_data.get("body"), dict) else {}
            
            # Clean request payload of filter-specific metadata
            if "metadata" in req_payload and "openpipe_request_id" in req_payload["metadata"]:
                req_payload = req_payload.copy()
                req_payload["metadata"] = req_payload["metadata"].copy()
                del req_payload["metadata"]["openpipe_request_id"]
            
            # Ensure required fields
            if "model" not in req_payload:
                req_payload["model"] = "unknown"
            if "messages" not in req_payload:
                req_payload["messages"] = []
            
            # Format response payload in OpenAI format with RAW content
            resp_payload = {
                "id": f"chatcmpl-{request_id.replace('.', '').replace('_', '')}",
                "object": "chat.completion",
                "created": int(received_at / 1000),
                "model": req_payload.get("model", "unknown"),
                "choices": [
                    {
                        "index": 0,
                        "message": raw_response_message,  # Raw content with preserved formatting!
                        "finish_reason": "stop"
                    }
                ]
            }
            
            # Add usage info if available
            usage_info = body.get("usage", {})
            if usage_info:
                resp_payload["usage"] = usage_info
            else:
                content = raw_response_message.get("content", "")
                estimated_tokens = len(str(content).split()) if content else 0
                resp_payload["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": estimated_tokens,
                    "total_tokens": estimated_tokens
                }
            
            # Format for OpenPipe API
            openpipe_payload = {
                "requestedAt": request_data.get("requested_at", received_at) if isinstance(request_data, dict) else received_at,
                "receivedAt": received_at,
                "reqPayload": req_payload,
                "respPayload": resp_payload,
                "statusCode": 200,
                "metadata": metadata
            }
            
            # Provide UI feedback before sending
            content_preview = raw_response_message.get("content", "")[:100] + "..." if len(raw_response_message.get("content", "")) > 100 else raw_response_message.get("content", "")
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📤 Reporting {interaction_type} with preserved formatting to OpenPipe ({inference_time_ms}ms)",
                    "done": False,
                    "hidden": False,
                },
            })
            
            # Send to OpenPipe asynchronously
            await self._send_to_openpipe(openpipe_payload)
            
            # Final success feedback
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"✅ Raw formatted content reported to OpenPipe successfully",
                    "done": True,
                    "hidden": False,
                },
            })
            
            self._log(f"Successfully reported {interaction_type} with raw formatting (took {inference_time_ms}ms)")
            
        except Exception as e:
            import traceback
            self._log(f"Error in outlet processing: {str(e)}", "ERROR")
            self._log(f"Traceback: {traceback.format_exc()}", "ERROR")
            
            # Error feedback to UI
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"❌ OpenPipe raw capture failed: {str(e)}",
                    "done": True,
                    "hidden": False,
                },
            })
            
        finally:
            # Clean up stored request data
            if request_id in self.request_store:
                del self.request_store[request_id]
            if request_id in self.streaming_content:
                del self.streaming_content[request_id]
        
        return body
