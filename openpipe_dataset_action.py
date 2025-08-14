"""
title: OpenPipe Dataset Collection (Training Data)
author: Cline & Gwyn
version: 1.1.0
required_open_webui_version: 0.5.0
icon_url: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAMAAAD04JH5AAAAQlBMVEVHcEz5Tyr////vRB8AAAD/VjLgXEAAAAAAAAAAAADlNxEAAADbLAX/l4JnJRfeRif7z8f9qpgkCQT/YkG3Oh9/MyIY1MKeAAAACnRSTlMA////Mf//injo8EBKYgAAArFJREFUeJztW4tugkAQ7BZPhS2gIP//q+WEe/soSG8uOcY0MW3ijsPs63L9+tqxY8eOdHHeECvCH0reEOVhMYGSu5vB8Xi9KNS3P2LQ7xoul8Y/c9cK0atXfb18K1zk76efCXRHcUcl8TOjPSo0vPQpnLkhHcElcLX+IGgKX8iXjO9QMASGDwkIRwEK4msBxvDFfxB4pgBpBkqCWYEqigIkrPhT+Gp+AlsTeKGAJYGhsDmBJx4g24SFI0EkBYwJC6KQwf97gHwFYnvAegSKQ3QF6IEHVSncgkD/1gPCKQNV3DrgFCKrFMbzwDML4BTY3AN/zgI/EWNnAaoOUCABrhISSgG3FWyuwINCdLGUCbrBZt3wwDy0GuNH1FeFun2Oqn04lA4dL57LT8ydAz3jd8vBfFoaf2RQSrzbOLrXfOROIrEi/ojDiBM3LxR38NMq6W/Hdn41fJKfsir8rAM3YjZj39fGCN5aolKgvRvAXkjWffeQgJ8KNZnoqgroPhCRgBkH7MUsHgEzkxZQBUwjiKyANQ6gFHDnEcUgOgHPgwAFpmY4Hw5E9YCTBQAFyM0CgAcSUYBUKc4uC0i4IyGsDsA8QG4vwHkAqkDevYAMg0w9oEqxe1AMnIggCngHlZipuEBPxea0Pr+p2D8gAnhAYBUA7wUkwiTEzIS4XpDIVAzcC9KaiDLdC+xekK8H8t4L7GKc316gSzHujCiBvQDbCwjdDd0swHkglSzAbEb2VAw7KwZux+gs8E9Ko29G/l5QYPcCwG6I3o69iShTBfKeiAg9ESUzFee7HQsB3o6TmIiw23EqUzFsO05iM7KyAOMBgc2C7G9QROkF3WMCOrxdCwMFusU3KX3Im5Xzl+373rpNZ+TXTyB8BAMvv0kZSvARPn0CksEH//O18ibljh07duSFX1tOylaAmcq/AAAAAElFTkSuQmCC
"""

import asyncio
import json
import re
from typing import Dict, Any, List
from pydantic import BaseModel, Field

class Action:
    """
    OpenPipe Dataset Action
    
    Adds conversations to OpenPipe datasets for fine-tuning and training.
    Supports creating new datasets and adding entries to existing ones.
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
        DEFAULT_DATASET_NAME: str = Field(
            default="OpenWebUI Conversations", 
            description="Default dataset name for new datasets"
        )
        DEFAULT_SPLIT: str = Field(
            default="TRAIN",
            description="Default split for dataset entries (TRAIN or TEST)",
            json_schema_extra={"enum": ["TRAIN", "TEST"]}
        )
        AUTO_CREATE_DATASET: bool = Field(
            default=True,
            description="Automatically create dataset if it doesn't exist"
        )
        INCLUDE_SYSTEM_MESSAGES: bool = Field(
            default=True,
            description="Include system messages in dataset entries"
        )
        DEBUG_LOGGING: bool = Field(
            default=False,
            description="Enable debug logging"
        )

    def __init__(self):
        self.valves = self.Valves()
        self._datasets_cache = {}  # Cache for dataset list
        
    def _log(self, message: str, level: str = "INFO"):
        """Debug logging helper with enhanced verbosity"""
        if self.valves.DEBUG_LOGGING or level in ["ERROR", "WARNING"]:
            import traceback
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [OpenPipe Dataset Action {level}] {message}")
            if level == "ERROR":
                print(f"[{timestamp}] [OpenPipe Dataset Action TRACE] {traceback.format_exc()}")
    
    async def _make_api_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make authenticated request to OpenPipe API"""
        if not self.valves.OPENPIPE_API_KEY:
            raise Exception("OpenPipe API key not configured")
            
        url = f"{self.valves.OPENPIPE_BASE_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.valves.OPENPIPE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        try:
            self._log(f"Making {method} request to: {url}")
            if data:
                self._log(f"Request data keys: {list(data.keys())}")
                self._log(f"Request data size: {len(json.dumps(data))} characters")
            
            import urllib.request
            import urllib.parse
            import urllib.error
            
            loop = asyncio.get_event_loop()
            
            def make_request():
                if method.upper() == "GET":
                    req = urllib.request.Request(url, headers=headers)
                elif method.upper() == "POST":
                    json_data = json.dumps(data).encode('utf-8') if data else b''
                    req = urllib.request.Request(url, data=json_data, headers=headers)
                else:
                    raise Exception(f"Unsupported HTTP method: {method}")
                
                try:
                    with urllib.request.urlopen(req, timeout=30) as response:
                        return response.getcode(), response.read().decode('utf-8')
                except urllib.error.HTTPError as e:
                    return e.code, e.read().decode('utf-8')
                except urllib.error.URLError as e:
                    raise Exception(f"URL Error: {str(e)}")
            
            status_code, response_text = await loop.run_in_executor(None, make_request)
            
            self._log(f"API response status: {status_code}")
            
            if status_code not in [200, 201]:
                self._log(f"API error response: {response_text}", "ERROR")
                self._log(f"Request headers: {headers}", "ERROR")
                if data:
                    self._log(f"Request data preview: {json.dumps(data, indent=2)[:500]}...", "ERROR")
                raise Exception(f"API request failed: {status_code} - {response_text}")
            
            response_data = json.loads(response_text)
            self._log(f"API response keys: {list(response_data.keys())}")
            return response_data
            
        except Exception as e:
            self._log(f"API request failed: {str(e)}", "ERROR")
            self._log(f"Exception type: {type(e).__name__}", "ERROR")
            raise
    
    async def _get_datasets(self) -> List[Dict[str, Any]]:
        """Get list of available datasets"""
        try:
            result = await self._make_api_request("GET", "datasets")
            datasets = result.get("data", [])
            
            # Cache the datasets
            self._datasets_cache = {ds["name"]: ds for ds in datasets}
            
            return datasets
        except Exception as e:
            self._log(f"Failed to fetch datasets: {str(e)}", "ERROR")
            return []
    
    async def _create_dataset(self, name: str) -> Dict[str, Any]:
        """Create a new dataset"""
        try:
            data = {"name": name}
            result = await self._make_api_request("POST", "datasets", data)
            
            # Update cache
            self._datasets_cache[name] = result
            
            return result
        except Exception as e:
            self._log(f"Failed to create dataset '{name}': {str(e)}", "ERROR")
            raise
    
    async def _add_entries_to_dataset(self, dataset_id: str, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add entries to a dataset"""
        try:
            data = {"entries": entries}
            result = await self._make_api_request("POST", f"datasets/{dataset_id}/entries", data)
            return result
        except Exception as e:
            self._log(f"Failed to add entries to dataset {dataset_id}: {str(e)}", "ERROR")
            raise
    
    def _convert_thinking_to_openpipe_format(self, content: str) -> str:
        """
        Convert content for OpenPipe by:
        1. Replacing <details type="reasoning"> blocks with <think> tags
        2. Removing <details type="tool_calls"> blocks (since they'll be in tool_calls)
        3. Removing <summary> content
        
        Returns:
            str: content formatted for OpenPipe
        """
        if not isinstance(content, str):
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
    
    def _extract_thinking_blocks(self, content: str) -> list:
        """
        Extract thinking blocks for metadata without modifying display content
        
        Supports multiple formats:
        - <thinking>...</thinking>
        - <details type="reasoning">...</details>
        
        Returns:
            list: extracted thinking blocks (just the content, no tags)
        """
        if not isinstance(content, str):
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
        if not isinstance(content, str):
            return content, [], []
            
        tool_metadata = []
        tool_calls = []
        
        # Debug: Log content analysis
        self._log(f"Parsing content of length {len(content)} for tool calls")
        
        # Check for tool_calls type
        tool_calls_count = len(re.findall(r'<details[^>]*type="tool_calls"[^>]*>', content))
        if tool_calls_count > 0:
            self._log(f"Found {tool_calls_count} <details type=\"tool_calls\"> tags")
        
        # Pattern to match OpenWebUI tool call blocks
        tool_pattern = r'<details[^>]*type="tool_calls"[^>]*>(.*?)</details>'
        
        def extract_openwebui_tool_call(match):
            full_block = match.group(0)
            full_content = match.group(1)
            
            self._log(f"Processing tool call block of length {len(full_block)}")
            
            # Extract attributes from the opening tag using individual patterns
            name_match = re.search(r'name="([^"]*)"', full_block)
            arguments_match = re.search(r'arguments="([^"]*)"', full_block)
            result_match = re.search(r'result="([^"]*)"', full_block)
            id_match = re.search(r'id="([^"]*)"', full_block)
            
            if not all([name_match, arguments_match, result_match, id_match]):
                self._log(f"Missing required attributes in tool call block", "WARNING")
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
            
            self._log(f"Extracted tool call - Name: {tool_name}, ID: {tool_id}")
            
            try:
                # Decode HTML entities and parse JSON arguments
                import html
                arguments_decoded = html.unescape(arguments_raw)
                result_decoded = html.unescape(result_raw)
                
                # Parse arguments JSON
                try:
                    arguments_json = json.loads(arguments_decoded)
                    self._log(f"Successfully parsed arguments JSON")
                except json.JSONDecodeError as e:
                    self._log(f"Failed to parse arguments JSON: {str(e)}")
                    arguments_json = {"raw_arguments": arguments_decoded}
                
                # Parse result JSON
                try:
                    result_json = json.loads(result_decoded)
                    self._log(f"Successfully parsed result JSON")
                except json.JSONDecodeError as e:
                    self._log(f"Failed to parse result JSON: {str(e)}")
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
                
                self._log(f"✅ Successfully parsed tool call: {tool_name} (ID: {tool_id})")
                
            except Exception as e:
                self._log(f"❌ Error parsing tool call: {str(e)}", "ERROR")
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
        matches = re.findall(tool_pattern, content, flags=re.DOTALL)
        self._log(f"Regex found {len(matches)} tool call matches")
        
        re.sub(tool_pattern, extract_openwebui_tool_call, content, flags=re.DOTALL)
        
        self._log(f"Final parsing results - tool_metadata: {len(tool_metadata)}, tool_calls: {len(tool_calls)}")
        
        return content, tool_metadata, tool_calls
    
    def _extract_conversation_messages(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and format messages from conversation body"""
        messages = body.get("messages", [])
        if not messages:
            return []
        
        formatted_messages = []
        
        for message in messages:
            if not isinstance(message, dict):
                continue
                
            role = message.get("role", "")
            content = message.get("content", "")
            
            # Handle multimodal content (images and other media)
            if isinstance(content, list):
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            formatted_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item_type in ["image_url", "image"]:
                            # Handle various image URL formats
                            image_url = None
                            if item.get("image_url"):
                                if isinstance(item["image_url"], dict):
                                    image_url = item["image_url"].get("url", "")
                                else:
                                    image_url = str(item["image_url"])
                            elif item.get("image"):
                                image_url = str(item["image"])
                            elif item.get("url"):
                                image_url = str(item["url"])
                            
                            if image_url:
                                formatted_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_url,
                                        "detail": item.get("detail", "auto")  # Support detail parameter
                                    }
                                })
                        elif item_type == "audio":
                            # Support audio content
                            formatted_content.append({
                                "type": "audio",
                                "audio": item.get("audio", {})
                            })
                        else:
                            # Preserve other content types as-is
                            formatted_content.append(item)
                    else:
                        # Handle non-dict items (strings, etc.)
                        if isinstance(item, str):
                            formatted_content.append({
                                "type": "text",
                                "text": item
                            })
                        else:
                            formatted_content.append(item)
                content = formatted_content
            
            # Skip empty messages (but allow messages with only images)
            if not content and not message.get("tool_calls") and not message.get("function_call"):
                continue
            
            # Skip system messages if configured
            if role == "system" and not self.valves.INCLUDE_SYSTEM_MESSAGES:
                continue
            
            # Apply thinking block and tool call parsing for assistant messages
            if role == "assistant" and isinstance(content, str):
                # Parse OpenWebUI tool calls first
                _, tool_metadata, parsed_tool_calls = self._parse_tool_blocks(content)
                
                # Convert <details> blocks to <think> tags and remove tool call blocks
                content = self._convert_thinking_to_openpipe_format(content)
                
                # Extract thinking blocks for metadata
                thinking_blocks = self._extract_thinking_blocks(message.get("content", ""))
                if thinking_blocks:
                    self._log(f"Extracted {len(thinking_blocks)} thinking blocks from assistant message")
                
                # If we parsed tool calls from OpenWebUI format, use those
                if parsed_tool_calls:
                    self._log(f"Using {len(parsed_tool_calls)} parsed tool calls from OpenWebUI format")
            
            formatted_message = {
                "role": role,
                "content": content
            }
            
            # Enhanced tool calls and function calls preservation
            # Prioritize parsed tool calls from OpenWebUI format
            if role == "assistant" and isinstance(message.get("content", ""), str):
                _, _, parsed_tool_calls = self._parse_tool_blocks(message.get("content", ""))
                if parsed_tool_calls:
                    formatted_message["tool_calls"] = parsed_tool_calls
                    self._log(f"Added {len(parsed_tool_calls)} parsed tool calls to message")
                elif "tool_calls" in message:
                    tool_calls = message["tool_calls"]
                    if isinstance(tool_calls, list) and tool_calls:
                        formatted_message["tool_calls"] = tool_calls
                        self._log(f"Preserved {len(tool_calls)} existing tool calls in message")
            elif "tool_calls" in message:
                tool_calls = message["tool_calls"]
                if isinstance(tool_calls, list) and tool_calls:
                    formatted_message["tool_calls"] = tool_calls
                    self._log(f"Preserved {len(tool_calls)} tool calls in message")
            
            if "function_call" in message:
                function_call = message["function_call"]
                if function_call:
                    formatted_message["function_call"] = function_call
                    self._log("Preserved function call in message")
            
            # Preserve tool/function response metadata
            if "name" in message:
                formatted_message["name"] = message["name"]
                self._log(f"Preserved function name: {message['name']}")
            
            if "tool_call_id" in message:
                formatted_message["tool_call_id"] = message["tool_call_id"]
                self._log(f"Preserved tool call ID: {message['tool_call_id']}")
            
            # Preserve additional tool-related fields
            if "function_response" in message:
                formatted_message["function_response"] = message["function_response"]
            
            if "tool_result" in message:
                formatted_message["tool_result"] = message["tool_result"]
                
            formatted_messages.append(formatted_message)
        
        return formatted_messages
    
    def _create_dataset_entry(self, messages: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a dataset entry from messages"""
        entry = {
            "messages": messages,
            "split": self.valves.DEFAULT_SPLIT
        }
        
        if metadata:
            # Convert metadata values to strings as required by API
            string_metadata = {k: str(v) for k, v in metadata.items() if v is not None}
            if string_metadata:
                entry["metadata"] = string_metadata
        
        return entry

    async def action(
        self,
        body: dict,
        __user__: dict = None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> str:
        """
        Add conversation to OpenPipe dataset
        
        Args:
            body: Conversation data
            __user__: User information
            __event_emitter__: Event emitter for status updates
            __event_call__: Event call function
            
        Returns:
            Status message
        """
        try:
            # Emit initial status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "🔄 Preparing to add conversation to OpenPipe dataset...",
                        "done": False,
                    },
                })
            
            # Extract messages from conversation
            messages = self._extract_conversation_messages(body)
            if not messages:
                return "❌ No valid messages found in conversation"
            
            # Log multimodal and tool content detection
            image_count = 0
            tool_call_count = 0
            audio_count = 0
            for msg in messages:
                if isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_count += 1
                        elif isinstance(item, dict) and item.get("type") == "audio":
                            audio_count += 1
                if msg.get("tool_calls"):
                    tool_call_count += len(msg["tool_calls"])
            
            content_summary = []
            if image_count > 0:
                content_summary.append(f"{image_count} images")
            if audio_count > 0:
                content_summary.append(f"{audio_count} audio files")
            if tool_call_count > 0:
                content_summary.append(f"{tool_call_count} tool calls")
            
            if content_summary:
                self._log(f"Detected multimodal/tool content: {', '.join(content_summary)}")
            
            # Get available datasets
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "📋 Fetching available datasets...",
                        "done": False,
                    },
                })
            
            datasets = await self._get_datasets()
            
            # Find or create dataset
            dataset_name = self.valves.DEFAULT_DATASET_NAME
            dataset = self._datasets_cache.get(dataset_name)
            
            if not dataset and self.valves.AUTO_CREATE_DATASET:
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": f"🆕 Creating new dataset: {dataset_name}",
                            "done": False,
                        },
                    })
                
                dataset = await self._create_dataset(dataset_name)
                self._log(f"Created new dataset: {dataset_name} (ID: {dataset['id']})")
            
            if not dataset:
                return f"❌ Dataset '{dataset_name}' not found and auto-creation is disabled"
            
            # Prepare metadata
            metadata = {
                "source": "openwebui",
                "user_id": __user__.get("id") if __user__ else None,
                "chat_id": body.get("metadata", {}).get("chat_id"),
                "model": body.get("model", "unknown"),
                "timestamp": str(int(body.get("timestamp", 0))),
                "message_count": str(len(messages))
            }
            
            # Create dataset entry
            entry = self._create_dataset_entry(messages, metadata)
            
            # Add to dataset
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"📤 Adding conversation to dataset: {dataset_name}",
                        "done": False,
                    },
                })
            
            result = await self._add_entries_to_dataset(dataset["id"], [entry])
            
            entries_created = result.get("entries_created", 0)
            errors = result.get("errors", {}).get("data", [])
            
            if errors:
                error_messages = [error.get("message", "Unknown error") for error in errors]
                return f"⚠️ Added to dataset with errors: {'; '.join(error_messages)}"
            
            # Success feedback
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": f"✅ Successfully added to dataset: {dataset_name}",
                        "done": True,
                    },
                })
            
            self._log(f"Successfully added {entries_created} entries to dataset {dataset_name}")
            
            return f"✅ Successfully added conversation to dataset '{dataset_name}' ({entries_created} entries created)"
            
        except Exception as e:
            error_msg = f"❌ Failed to add to dataset: {str(e)}"
            self._log(error_msg, "ERROR")
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": error_msg,
                        "done": True,
                    },
                })
            
            return error_msg
