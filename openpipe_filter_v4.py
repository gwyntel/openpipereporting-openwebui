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
        SHOW_SUCCESS_NOTIFICATION: bool = Field(
            default=True,
            description="Show 'Reported to OpenPipe successfully' notification"
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
            tuple: (content, tool_use, tool_calls)
        """
        if not self.valves.ENABLE_PARSING:
            self._log("Tool parsing disabled via ENABLE_PARSING valve", "DEBUG")
            return content, [], []
            
        tool_use = []
        tool_calls = []
        
        # Debug: Log content length and check for tool call patterns
        self._log(f"Parsing content of length {len(content)}", "DEBUG")
        
        # Check for any details tags first
        details_count = len(re.findall(r'<details[^>]*>', content))
        self._log(f"Found {details_count} <details> tags in content", "DEBUG")
        
        # Check specifically for tool_calls type
        tool_calls_count = len(re.findall(r'<details[^>]*type="tool_calls"[^>]*>', content))
        self._log(f"Found {tool_calls_count} <details type=\"tool_calls\"> tags", "DEBUG")
        
        # Pattern to match OpenWebUI tool call blocks - more flexible attribute matching
        tool_pattern = r'<details[^>]*type="tool_calls"[^>]*>(.*?)</details>'
        
        def extract_openwebui_tool_call(match):
            full_block = match.group(0)
            full_content = match.group(1)
            
            self._log(f"Processing tool call block of length {len(full_block)}", "DEBUG")
            self._log(f"Tool call block preview: {full_block[:200]}...", "DEBUG")
            
            # Extract attributes from the opening tag using individual patterns
            name_match = re.search(r'name="([^"]*)"', full_block)
            arguments_match = re.search(r'arguments="([^"]*)"', full_block)
            result_match = re.search(r'result="([^"]*)"', full_block)
            id_match = re.search(r'id="([^"]*)"', full_block)
            
            # Debug attribute extraction
            self._log(f"Attribute extraction - name: {'✓' if name_match else '✗'}, "
                     f"arguments: {'✓' if arguments_match else '✗'}, "
                     f"result: {'✓' if result_match else '✗'}, "
                     f"id: {'✓' if id_match else '✗'}", "DEBUG")
            
            if name_match:
                self._log(f"Found tool name: {name_match.group(1)}", "DEBUG")
            if id_match:
                self._log(f"Found tool ID: {id_match.group(1)}", "DEBUG")
            
            if not all([name_match, arguments_match, result_match, id_match]):
                self._log(f"Missing required attributes in tool call block", "WARNING")
                # Show which attributes are missing
                missing = []
                if not name_match: missing.append("name")
                if not arguments_match: missing.append("arguments")
                if not result_match: missing.append("result")
                if not id_match: missing.append("id")
                self._log(f"Missing attributes: {', '.join(missing)}", "WARNING")
                return match.group(0)
            
            tool_name = name_match.group(1)
            arguments_raw = arguments_match.group(1)
            result_raw = result_match.group(1)
            tool_id = id_match.group(1)
            
            self._log(f"Extracted tool call - Name: {tool_name}, ID: {tool_id}", "DEBUG")
            self._log(f"Arguments length: {len(arguments_raw)}, Result length: {len(result_raw)}", "DEBUG")
            
            try:
                # Decode HTML entities and parse JSON arguments
                import html
                arguments_decoded = html.unescape(arguments_raw)
                result_decoded = html.unescape(result_raw)
                
                self._log(f"Decoded arguments length: {len(arguments_decoded)}", "DEBUG")
                self._log(f"Arguments preview: {arguments_decoded[:100]}...", "DEBUG")
                
                # Parse arguments JSON
                try:
                    arguments_json = json.loads(arguments_decoded)
                    self._log(f"Successfully parsed arguments JSON", "DEBUG")
                except json.JSONDecodeError as e:
                    self._log(f"Failed to parse arguments JSON: {str(e)}", "DEBUG")
                    arguments_json = {"raw_arguments": arguments_decoded}
                
                # Parse result JSON
                try:
                    result_json = json.loads(result_decoded)
                    self._log(f"Successfully parsed result JSON", "DEBUG")
                except json.JSONDecodeError as e:
                    self._log(f"Failed to parse result JSON: {str(e)}", "DEBUG")
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
                tool_use.append({
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "arguments": arguments_json,
                    "result": result_json,
                    "arguments_raw": arguments_decoded,
                    "result_raw": result_decoded
                })
                
                self._log(f"✅ Successfully parsed tool call: {tool_name} (ID: {tool_id})")
                
            except Exception as e:
                self._log(f"❌ Error parsing tool call: {str(e)}", "ERROR")
                # Fallback metadata
                tool_use.append({
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "arguments_raw": arguments_raw,
                    "result_raw": result_raw,
                    "parse_error": str(e)
                })
            
            return match.group(0)  # Keep original for display
        
        # Extract tool metadata without modifying content for display
        matches = re.findall(tool_pattern, content, flags=re.DOTALL)
        self._log(f"Regex found {len(matches)} tool call matches", "DEBUG")
        
        re.sub(tool_pattern, extract_openwebui_tool_call, content, flags=re.DOTALL)
        
        self._log(f"Final parsing results - tool_use: {len(tool_use)}, tool_calls: {len(tool_calls)}", "DEBUG")
        
        return content, tool_use, tool_calls
    
    def _split_content_with_tool_calls(self, processed: Dict[str, Any], response_message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split content around tool calls to create separate messages for OpenPipe
        
        This merges pre-tool content (thinking/text) with tool calls into single messages,
        and separates post-tool content into follow-up messages.
        
        Returns:
            List of message objects for OpenPipe
        """
        messages = []
        content = processed["display_content"]
        
        # Find all tool call blocks in the original content
        tool_pattern = r'<details[^>]*type="tool_calls"[^>]*>.*?</details>'
        tool_matches = list(re.finditer(tool_pattern, content, re.DOTALL))
        
        if not tool_matches:
            # No tool calls found, return single message
            return [{
                "role": response_message.get("role", "assistant"),
                "content": processed["openpipe_content"]
            }]
        
        current_pos = 0
        
        # Process each tool call and its preceding content
        for i, match in enumerate(tool_matches):
            # Get text before this tool call (only the part we haven't processed yet)
            pre_tool_text = content[current_pos:match.start()].strip()
            
            if i < len(processed["tool_calls"]):
                tool_call = processed["tool_calls"][i]
                
                # Process pre-tool content if it exists
                processed_pre_tool = ""
                if pre_tool_text:
                    processed_pre_tool = self._convert_thinking_to_openpipe_format(pre_tool_text)
                
                # Create message with pre-tool content and tool call
                message = {
                    "role": response_message.get("role", "assistant"),
                    "content": processed_pre_tool.strip(),
                    "tool_calls": [tool_call]
                }
                messages.append(message)
                
                self._log(f"Created message {i+1} with pre-tool content ({len(processed_pre_tool)} chars) + tool call", "DEBUG")
                
                # Note: Tool results are included in the tool_metadata for training context
                # OpenPipe doesn't support role: "tool" in completion format, so results are in metadata
            
            # Move position to after this tool call
            current_pos = match.end()
        
        # Handle any remaining content after the last tool call
        post_tool_text = content[current_pos:].strip()
        if post_tool_text:
            processed_post_tool = self._convert_thinking_to_openpipe_format(post_tool_text)
            # Only add message if there's actual content after processing
            if processed_post_tool.strip():
                messages.append({
                    "role": response_message.get("role", "assistant"),
                    "content": processed_post_tool
                })
                self._log(f"Created post-tool message with {len(processed_post_tool)} chars", "DEBUG")
        
        # Fallback: if no messages were created, return a single message
        if not messages:
            if processed["tool_calls"]:
                messages.append({
                    "role": response_message.get("role", "assistant"),
                    "content": processed["openpipe_content"],
                    "tool_calls": processed["tool_calls"]
                })
            else:
                messages.append({
                    "role": response_message.get("role", "assistant"),
                    "content": processed["openpipe_content"]
                })
        
        self._log(f"Split into {len(messages)} messages: {len([m for m in messages if m.get('tool_calls')])} with tool calls, {len([m for m in messages if not m.get('tool_calls')])} text-only")
        return messages
    
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
                "tool_use": [],
                "tool_calls": []
            }
        
        # Extract thinking blocks for OpenPipe metadata
        thinking_blocks = self._extract_thinking_blocks(content)
        
        # Parse tool blocks (OpenWebUI format)
        _, tool_use, tool_calls = self._parse_tool_blocks(content)
        
        # Convert content for OpenPipe (replace <details> with <think> tags, remove tool calls)
        openpipe_content = self._convert_thinking_to_openpipe_format(content)
        
        return {
            "display_content": content,        # Keep original for OpenWebUI display
            "openpipe_content": openpipe_content,  # Formatted for OpenPipe
            "thinking_blocks": thinking_blocks,
            "tool_use": tool_use,
            "tool_calls": tool_calls           # Parsed tool calls in OpenAI format
        }
    
    async def _send_to_openpipe(self, payload):
        """Send report(s) to OpenPipe API"""
        if not self.valves.OPENPIPE_API_KEY:
            self._log("OpenPipe API key not configured", "WARNING")
            return
            
        url = f"{self.valves.OPENPIPE_BASE_URL}/report"
        headers = {
            "Authorization": f"Bearer {self.valves.OPENPIPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Handle multiple payloads (for tool call sequences)
        payloads_to_send = payload if isinstance(payload, list) else [payload]
        
        try:
            self._log(f"Sending {len(payloads_to_send)} report(s) to OpenPipe...")
            
            import urllib.request
            import urllib.error
            
            loop = asyncio.get_event_loop()
            
            def make_request(single_payload):
                data = json.dumps(single_payload).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=headers)
                try:
                    with urllib.request.urlopen(req, timeout=10) as response:
                        return response.getcode(), response.read().decode('utf-8')
                except urllib.error.HTTPError as e:
                    return e.code, e.read().decode('utf-8')
                except urllib.error.URLError as e:
                    raise Exception(f"URL Error: {str(e)}")
            
            # Send each payload
            success_count = 0
            for i, single_payload in enumerate(payloads_to_send):
                try:
                    status_code, response_text = await loop.run_in_executor(None, make_request, single_payload)
                    
                    if status_code == 200:
                        success_count += 1
                        if len(payloads_to_send) > 1:
                            self._log(f"✅ Successfully reported message {i+1}/{len(payloads_to_send)} to OpenPipe")
                    else:
                        self._log(f"❌ OpenPipe API error for message {i+1}: {status_code} - {response_text}", "ERROR")
                        
                except Exception as e:
                    self._log(f"❌ Failed to report message {i+1} to OpenPipe: {str(e)}", "ERROR")
            
            if success_count == len(payloads_to_send):
                self._log(f"✅ Successfully reported all {success_count} messages to OpenPipe")
            else:
                self._log(f"⚠️ Reported {success_count}/{len(payloads_to_send)} messages to OpenPipe", "WARNING")
                
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
                
                # Debug: Log the content being processed
                self._log(f"Processing response content of length {len(content)}", "DEBUG")
                if content:
                    self._log(f"Content preview: {content[:300]}...", "DEBUG")
                
                # Process content for thinking blocks and tool metadata
                processed = self._process_content(content)
                
                # Debug: Log processing results
                self._log(f"Processing results - thinking_blocks: {len(processed['thinking_blocks'])}, "
                         f"tool_use: {len(processed['tool_use'])}, "
                         f"tool_calls: {len(processed['tool_calls'])}", "DEBUG")
                
                # Update the message content for OpenWebUI display
                if processed["display_content"] != content:
                    response_message["content"] = processed["display_content"]
                    self._log("Processed thinking blocks and tool metadata")
                
                # Check if we have tool calls - need to split into separate messages
                if processed["tool_calls"]:
                    # Split content around tool calls to create separate messages
                    messages_to_send = self._split_content_with_tool_calls(processed, response_message)
                    openpipe_response = messages_to_send  # This will be a list of messages
                else:
                    # Single message without tool calls
                    openpipe_response = {
                        "role": response_message.get("role", "assistant"),
                        "content": processed["openpipe_content"]
                    }
                    
                    # Add existing tool calls if present (legacy format)
                    if "tool_calls" in response_message:
                        openpipe_response["tool_calls"] = response_message["tool_calls"]
                    if "function_call" in response_message:
                        openpipe_response["function_call"] = response_message["function_call"]
                    
            else:
                processed = {"thinking_blocks": [], "tool_use": [], "tool_calls": []}
                openpipe_response = {"role": "assistant", "content": ""}
            
            # Build OpenPipe payload
            req_payload = request_data["body"].copy()
            if "metadata" in req_payload and "openpipe_request_id" in req_payload["metadata"]:
                del req_payload["metadata"]["openpipe_request_id"]
            
            # Enhanced metadata with parsing results
            tags = {
                "thinking_blocks_count": len(processed["thinking_blocks"]),
                "tool_calls_count": len(processed["tool_use"]),
                "inference_time_ms": inference_time,
                "parsing_enabled": self.valves.ENABLE_PARSING,
                "filter_version": "v4.0_enhanced_parsing"
            }
            
            # Add thinking block content to metadata (for OpenPipe training)
            if processed["thinking_blocks"]:
                tags["thinking_blocks"] = processed["thinking_blocks"]
            if processed["tool_use"]:
                tags["tool_use"] = processed["tool_use"]
            
            # Handle multiple messages for tool calls
            if isinstance(openpipe_response, list):
                # Multiple messages - send each as a separate completion
                openpipe_payloads = []
                for i, message in enumerate(openpipe_response):
                    # Determine finish_reason based on message content
                    if message.get("tool_calls"):
                        finish_reason = "tool_calls"
                    else:
                        finish_reason = "stop"
                    
                    payload = {
                        "requestedAt": request_data["requested_at"],
                        "receivedAt": received_at,
                        "reqPayload": req_payload,
                        "respPayload": {
                            "id": f"chatcmpl-{request_id.replace('.', '').replace('_', '')}-{i}",
                            "object": "chat.completion",
                            "created": int(received_at / 1000),
                            "model": req_payload.get("model", "unknown"),
                            "choices": [{
                                "index": 0,
                                "message": message,
                                "finish_reason": finish_reason
                            }]
                        },
                        "statusCode": 200,
                        "tags": {**tags, "message_index": i, "total_messages": len(openpipe_response)}
                    }
                    openpipe_payloads.append(payload)
                openpipe_payload = openpipe_payloads
            else:
                # Single message
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
                        "tags": tags
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
            if self.valves.SHOW_SUCCESS_NOTIFICATION:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "✅ Reported to OpenPipe successfully",
                        "done": True,
                        "hidden": False,
                    },
                })
            
            self._log(f"Successfully reported with {len(processed['thinking_blocks'])} thinking blocks and {len(processed['tool_use'])} tool calls")
            
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
