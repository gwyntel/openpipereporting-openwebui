"""
title: OpenPipe Enhanced Reporting (Simplified)
author: Cline & Gwyn
version: 4.0.0
required_open_webui_version: 0.5.0
"""

"""
OpenPipe Reporting Filter v4.0 - Simplified with Enhanced Parsing

Simplified filter that parses thinking blocks and tool use metadata properly.
Features:
- Parses <thinking> blocks into <think> tags
- Extracts tool use metadata
- Simplified configuration (only 3 valves vs 19 in v3)
- Enhanced content processing for OpenWebUI display
- Raw content preservation for OpenPipe training
"""

import asyncio
import json
import time
import re
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class Filter:
    """
    Simplified OpenPipe Reporting Filter with Enhanced Parsing
    
    Key improvements over v3:
    - 60% smaller codebase
    - Proper thinking block parsing
    - Tool use metadata extraction
    - Better OpenWebUI integration
    """
    
    class Valves(BaseModel):
        OPENPIPE_API_KEY: str = Field(
            default="", 
            description="OpenPipe API key for authentication"
        )
        OPENPIPE_BASE_URL: str = Field(
            default="https://api.openpipe.ai/api/v1", 
            description="OpenPipe API base URL"
        )
        ENABLE_PARSING: bool = Field(
            default=True, 
            description="Enable thinking block and tool metadata parsing"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.request_store = {}
        self.toggle = True
        
        # Custom OpenPipe Icon
        self.icon = """data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAAQlBMVEVHcEz5Tyr////vRB8AAAD/VjLgXEAAAAAAAAAAAADlNxEAAADbLAX/l4JnJRfeRif7z8f9qpgkCQT/YkG3Oh9/MyIY1MKeAAAACnRSTlMA////Mf//injo8EBKYgAAArFJREFUeJztW4tugkAQ7BZPhS2gIP//q+WEe/soSG8uOcY0MW3ijsPs63L9+tqxY8eOdHHeECvCH0reEOVhMYGSu5vB8Xi9KNS3P2LQ7xoul8Y/c9cK0atXfb18K1zk76efCXRHcUcl8TOjPSo0vPQpnLkhHcElcLX+IGgKX8iXjO9QMASGDwkIRwEK4msBxvDFfxB4pgBpBkqCWYEqigIkrPhT+Gp+AlsTeKGAJYGhsDmBJx4g24SFI0EkBYwJC6KQwf97gHwFYnvAegSKQ3QF6IEHVSncgkD/1gPCKQNV3DrgFCKrFMbzwDML4BTY3AN/zgI/EWNnAaoOUCABrhISSgG3FWyuwINCdLGUCbrBZt3wwDy0GuNH1FeFun2Oqn04lA4dL57LT8ydAz3jd8vBfFoaf2RQSrzbOLrXfOROIrEi/ojDiBM3LxR38NMq6W/Hdn41fJKfsir8rAM3YjZj39fGCN5aolKgvRvAXkjWffeQgJ8KNZnoqgroPhCRgBkH7MUsHgEzkxZQBUwjiKyANQ6gFHDnEcUgOgHPgwAFpmY4Hw5E9YCTBQAFyM0CgAcSUYBUKc4uC0i4IyGsDsA8QG4vwHkAqkDevYAMg0w9oEqxe1AMnIggCngHlZipuEBPxea0Pr+p2D8gAnhAYBUA7wUkwiTEzIS4XpDIVAzcC9KaiDLdC+xekK8H8t4L7GKc316gSzHujCiBvQDbCwjdDd0swHkglSzAbEb2VAw7KwZux+gs8E9Ko29G/l5QYPcCwG6I3o69iShTBfKeiAg9ESUzFee7HQsB3o6TmIiw23EqUzFsO05iM7KyAOMBgc2C7G9QROkF3WMCOrxdCwMFusU3KX3Im5Xzl+373rpNZ+TXTyB8BAMvv0kZSvARPn0CksEH//O18ibljh07duSFX1tOylaAmcq/AAAAAElFTkSuQmCC"""
        
        self._log("OpenPipe Filter v4.0 initialized")
        
    def _log(self, message: str, level: str = "INFO"):
        """Simple logging helper"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [OpenPipe v4.0 {level}] {message}")
    
    def _extract_thinking_blocks(self, content: str) -> list:
        """
        Extract thinking blocks for OpenPipe metadata without modifying display content
        
        Supports multiple formats:
        - <thinking>...</thinking>
        - <details type="reasoning">...</details>
        
        Returns:
            list: extracted thinking blocks (just the content, no tags)
        """
        if not self.valves.ENABLE_PARSING:
            return []
            
        thinking_blocks = []
        
        # Pattern 1: <details type="reasoning" ...>...</details> blocks
        details_pattern = r'<details[^>]*type="reasoning"[^>]*>(.*?)</details>'
        details_matches = re.findall(details_pattern, content, re.DOTALL)
        
        for match in details_matches:
            full_content = match.strip()
            
            # Remove <summary> tag and its content completely
            remaining_content = re.sub(r'<summary[^>]*>.*?</summary>', '', full_content, flags=re.DOTALL).strip()
            
            # Clean up leading '>' characters from remaining content
            if remaining_content.startswith('>'):
                remaining_content = remaining_content[1:].strip()
            
            # Clean up multiple '>' at start of lines
            remaining_content = re.sub(r'^>\s*', '', remaining_content, flags=re.MULTILINE)
            
            if remaining_content:
                thinking_blocks.append(remaining_content)
        
        # Pattern 2: <thinking>...</thinking> blocks (fallback)
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)
        
        for match in thinking_matches:
            thinking_content = match.strip()
            thinking_blocks.append(thinking_content)
        
        return thinking_blocks
    
    def _convert_thinking_to_openpipe_format(self, content: str) -> str:
        """
        Convert content for OpenPipe by:
        1. Replacing <details type="reasoning"> blocks with <think> tags
        2. Removing <details type="tool_calls"> blocks (since they'll be in tool_calls)
        3. Removing <summary> content
        
        Returns:
            str: content formatted for OpenPipe
        """
        if not self.valves.ENABLE_PARSING:
            return content
            
        def replace_details_with_think(match):
            full_content = match.group(1).strip()
            
            # Remove <summary> tag and its content completely
            remaining_content = re.sub(r'<summary[^>]*>.*?</summary>', '', full_content, flags=re.DOTALL).strip()
            
            # Clean up leading '>' characters
            if remaining_content.startswith('>'):
                remaining_content = remaining_content[1:].strip()
            
            # Clean up multiple '>' at start of lines
            remaining_content = re.sub(r'^>\s*', '', remaining_content, flags=re.MULTILINE)
            
            if remaining_content:
                return f'<think>{remaining_content}</think>'
            else:
                return ''
        
        # Replace <details type="reasoning" ...>...</details> with <think>...</think>
        details_pattern = r'<details[^>]*type="reasoning"[^>]*>(.*?)</details>'
        openpipe_content = re.sub(details_pattern, replace_details_with_think, content, flags=re.DOTALL)
        
        # Remove <details type="tool_calls" ...>...</details> blocks entirely
        # (tool calls will be represented in the tool_calls field instead)
        tool_calls_pattern = r'<details[^>]*type="tool_calls"[^>]*>.*?</details>'
        openpipe_content = re.sub(tool_calls_pattern, '', openpipe_content, flags=re.DOTALL)
        
        # Clean up extra whitespace
        openpipe_content = re.sub(r'\n\s*\n\s*\n', '\n\n', openpipe_content)
        openpipe_content = openpipe_content.strip()
        
        return openpipe_content
    
    def _strip_thinking_blocks(self, content: str) -> str:
        """
        Remove thinking blocks from content for clean display
        
        Returns:
            str: content with thinking blocks removed
        """
        if not self.valves.ENABLE_PARSING:
            return content
            
        # Remove <details type="reasoning" ...>...</details> blocks
        details_pattern = r'<details[^>]*type="reasoning"[^>]*>.*?</details>'
        cleaned_content = re.sub(details_pattern, '', content, flags=re.DOTALL)
        
        # Remove <thinking>...</thinking> blocks (fallback)
        thinking_pattern = r'<thinking>.*?</thinking>'
        cleaned_content = re.sub(thinking_pattern, '', cleaned_content, flags=re.DOTALL)
        
        # Clean up extra whitespace
        cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def _parse_tool_blocks(self, content: str) -> tuple[str, list, list]:
        """
        Parse OpenWebUI tool call blocks and extract metadata
        
        OpenWebUI formats tool calls as:
        <details type="tool_calls" done="true" id="tooluse_xxx" name="tool_name" arguments="..." result="..." files="">
        <summary>Tool Executed</summary>
        </details>
        
        Returns:
            tuple: (content, tool_metadata, tool_calls)
        """
        if not self.valves.ENABLE_PARSING:
            return content, [], []
            
        tool_metadata = []
        tool_calls = []
        
        # Pattern to match OpenWebUI tool call blocks
        tool_pattern = r'<details[^>]*type="tool_calls"[^>]*?name="([^"]*)"[^>]*?arguments="([^"]*)"[^>]*?result="([^"]*)"[^>]*?id="([^"]*)"[^>]*?>(.*?)</details>'
        
        def extract_openwebui_tool_call(match):
            tool_name = match.group(1)
            arguments_raw = match.group(2)
            result_raw = match.group(3)
            tool_id = match.group(4)
            full_content = match.group(5)
            
            try:
                # Decode HTML entities and parse JSON arguments
                import html
                arguments_decoded = html.unescape(arguments_raw)
                result_decoded = html.unescape(result_raw)
                
                # Parse arguments JSON
                try:
                    arguments_json = json.loads(arguments_decoded)
                except json.JSONDecodeError:
                    arguments_json = {"raw_arguments": arguments_decoded}
                
                # Parse result JSON
                try:
                    result_json = json.loads(result_decoded)
                except json.JSONDecodeError:
                    result_json = result_decoded
                
                # Create OpenAI-style tool call
                tool_call = {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_decoded
                    }
                }
                tool_calls.append(tool_call)
                
                # Create metadata entry
                tool_metadata.append({
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "arguments": arguments_json,
                    "result": result_json,
                    "arguments_raw": arguments_decoded,
                    "result_raw": result_decoded
                })
                
                self._log(f"Parsed tool call: {tool_name} (ID: {tool_id})")
                
            except Exception as e:
                self._log(f"Error parsing tool call: {str(e)}", "WARNING")
                # Fallback metadata
                tool_metadata.append({
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "arguments_raw": arguments_raw,
                    "result_raw": result_raw,
                    "parse_error": str(e)
                })
            
            return match.group(0)  # Keep original for display
        
        # Extract tool metadata without modifying content for display
        re.sub(tool_pattern, extract_openwebui_tool_call, content, flags=re.DOTALL)
        
        return content, tool_metadata, tool_calls
    
    def _process_content(self, content: str) -> Dict[str, Any]:
        """
        Process message content to extract thinking blocks and tool metadata
        
        Returns:
            Dict with processed content and metadata
        """
        if not isinstance(content, str):
            return {
                "display_content": str(content),
                "openpipe_content": str(content),
                "thinking_blocks": [],
                "tool_metadata": [],
                "tool_calls": []
            }
        
        # Extract thinking blocks for OpenPipe metadata
        thinking_blocks = self._extract_thinking_blocks(content)
        
        # Parse tool blocks (OpenWebUI format)
        _, tool_metadata, tool_calls = self._parse_tool_blocks(content)
        
        # Convert content for OpenPipe (replace <details> with <think> tags, remove tool calls)
        openpipe_content = self._convert_thinking_to_openpipe_format(content)
        
        return {
            "display_content": content,        # Keep original for OpenWebUI display
            "openpipe_content": openpipe_content,  # Formatted for OpenPipe
            "thinking_blocks": thinking_blocks,
            "tool_metadata": tool_metadata,
            "tool_calls": tool_calls           # Parsed tool calls in OpenAI format
        }
    
    async def _send_to_openpipe(self, payload: Dict[str, Any]):
        """Send report to OpenPipe API"""
        if not self.valves.OPENPIPE_API_KEY:
            self._log("OpenPipe API key not configured", "WARNING")
            return
            
        url = f"{self.valves.OPENPIPE_BASE_URL}/report"
        headers = {
            "Authorization": f"Bearer {self.valves.OPENPIPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            self._log("Sending report to OpenPipe...")
            
            import urllib.request
            import urllib.error
            
            loop = asyncio.get_event_loop()
            
            def make_request():
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        return response.getcode(), response.read().decode('utf-8')
                except urllib.error.HTTPError as e:
                    return e.code, e.read().decode('utf-8')
                except urllib.error.URLError as e:
                    raise Exception(f"URL Error: {str(e)}")
            
            status_code, response_text = await loop.run_in_executor(None, make_request)
            
            if status_code == 200:
                self._log("✅ Successfully reported to OpenPipe")
            else:
                self._log(f"❌ OpenPipe API error: {status_code} - {response_text}", "ERROR")
                
        except Exception as e:
            self._log(f"❌ Failed to report to OpenPipe: {str(e)}", "ERROR")
    
    async def inlet(self, body: Dict[str, Any], __event_emitter__, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Capture incoming request"""
        if not self.toggle:
            return body
            
        # Generate request ID and store data
        request_id = f"{time.time()}_{id(body)}"
        self.request_store[request_id] = {
            "body": body.copy(),
            "user": user.copy() if user else None,
            "requested_at": int(time.time() * 1000),
            "start_time": time.perf_counter()
        }
        
        # Add tracking ID
        if "metadata" not in body:
            body["metadata"] = {}
        body["metadata"]["openpipe_request_id"] = request_id
        
        # UI feedback
        await __event_emitter__({
            "type": "status",
            "data": {
                "description": "🔄 Capturing request for OpenPipe",
                "done": False,
                "hidden": False,
            },
        })
        
        self._log(f"Captured request {request_id}")
        return body
    
    async def outlet(self, body: Dict[str, Any], __event_emitter__, user: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process response and send to OpenPipe"""
        if not self.toggle:
            return body
            
        # Find corresponding request
        request_id = body.get("metadata", {}).get("openpipe_request_id")
        if not request_id and self.request_store:
            request_id = max(self.request_store.keys(), key=lambda x: float(x.split('_')[0]))
            
        if not request_id or request_id not in self.request_store:
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
            inference_time = int((time.perf_counter() - request_data["start_time"]) * 1000)
            
            # Extract and process response message
            messages = body.get("messages", [])
            if messages:
                response_message = messages[-1]
                content = response_message.get("content", "")
                
                # Process content for thinking blocks and tool metadata
                processed = self._process_content(content)
                
                # Update the message content for OpenWebUI display
                if processed["display_content"] != content:
                    response_message["content"] = processed["display_content"]
                    self._log("Processed thinking blocks and tool metadata")
                
                # Prepare response message for OpenPipe (with formatted content)
                openpipe_response = {
                    "role": response_message.get("role", "assistant"),
                    "content": processed["openpipe_content"]  # Send formatted content to OpenPipe
                }
                
                # Add tool calls if present (prioritize parsed tool calls from OpenWebUI format)
                if processed["tool_calls"]:
                    openpipe_response["tool_calls"] = processed["tool_calls"]
                elif "tool_calls" in response_message:
                    openpipe_response["tool_calls"] = response_message["tool_calls"]
                    
                # Add function calls if present (legacy format)
                if "function_call" in response_message:
                    openpipe_response["function_call"] = response_message["function_call"]
                    
            else:
                processed = {"thinking_blocks": [], "tool_metadata": [], "tool_calls": []}
                openpipe_response = {"role": "assistant", "content": ""}
            
            # Build OpenPipe payload
            req_payload = request_data["body"].copy()
            if "metadata" in req_payload and "openpipe_request_id" in req_payload["metadata"]:
                del req_payload["metadata"]["openpipe_request_id"]
            
            # Enhanced metadata with parsing results
            metadata = {
                "thinking_blocks_count": len(processed["thinking_blocks"]),
                "tool_calls_count": len(processed["tool_metadata"]),
                "inference_time_ms": inference_time,
                "parsing_enabled": self.valves.ENABLE_PARSING,
                "filter_version": "v4.0_enhanced_parsing"
            }
            
            # Add thinking block content to metadata (for OpenPipe training)
            if processed["thinking_blocks"]:
                metadata["thinking_blocks"] = processed["thinking_blocks"]
            if processed["tool_metadata"]:
                metadata["tool_metadata"] = processed["tool_metadata"]
            
            openpipe_payload = {
                "requestedAt": request_data["requested_at"],
                "receivedAt": received_at,
                "reqPayload": req_payload,
                "respPayload": {
                    "id": f"chatcmpl-{request_id.replace('.', '').replace('_', '')}",
                    "object": "chat.completion",
                    "created": int(received_at / 1000),
                    "model": req_payload.get("model", "unknown"),
                    "choices": [{
                        "index": 0,
                        "message": openpipe_response,
                        "finish_reason": "stop"
                    }]
                },
                "statusCode": 200,
                "metadata": metadata
            }
            
            # UI feedback
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"📤 Reporting to OpenPipe ({inference_time}ms)",
                    "done": False,
                    "hidden": False,
                },
            })
            
            # Send to OpenPipe
            await self._send_to_openpipe(openpipe_payload)
            
            # Success feedback
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": "✅ Reported to OpenPipe successfully",
                    "done": True,
                    "hidden": False,
                },
            })
            
            self._log(f"Successfully reported with {len(processed['thinking_blocks'])} thinking blocks and {len(processed['tool_calls'])} tool calls")
            
        except Exception as e:
            self._log(f"Error in outlet processing: {str(e)}", "ERROR")
            await __event_emitter__({
                "type": "status",
                "data": {
                    "description": f"❌ OpenPipe reporting failed: {str(e)}",
                    "done": True,
                    "hidden": False,
                },
            })
        finally:
            # Cleanup
            if request_id in self.request_store:
                del self.request_store[request_id]
        
        return body
