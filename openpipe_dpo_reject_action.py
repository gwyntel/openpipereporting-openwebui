"""
title: OpenPipe DPO Reject Response
author: Cline & Gwyn
version: 1.2.0
required_open_webui_version: 0.5.0
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0iI0RDMjYyNiIgc3Ryb2tlPSIjQjkxQzFDIiBzdHJva2Utd2lkdGg9IjIiLz4KICA8cGF0aCBkPSJtOSA5IDYgNm0wLTYtNiA2IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K
"""

import asyncio
import json
import os
import time
import threading
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path

# Embedded shared state management for DPO preference pairs
class DPOSharedState:
    """Thread-safe file-based shared state for DPO preference pairs"""
    
    def __init__(self, state_file: str = "dpo_pending_pairs.json"):
        self.state_file = Path(state_file)
        self._lock = threading.Lock()
        self._ensure_state_file()
    
    def _ensure_state_file(self):
        """Ensure the state file exists"""
        if not self.state_file.exists():
            self._write_state({})
    
    def _read_state(self) -> Dict[str, Any]:
        """Read state from file with error handling"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Clean up old entries (older than 24 hours)
                    self._cleanup_old_entries(data)
                    return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"[DPO SharedState] Error reading state file: {e}")
            # Return empty state if file is corrupted
            return {}
        return {}
    
    def _write_state(self, state: Dict[str, Any]):
        """Write state to file with error handling"""
        try:
            # Write to temporary file first, then rename for atomicity
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.state_file)
        except IOError as e:
            print(f"[DPO SharedState] Error writing state file: {e}")
            raise
    
    def _cleanup_old_entries(self, state: Dict[str, Any]):
        """Remove entries older than 24 hours"""
        current_time = time.time()
        cutoff_time = current_time - (24 * 60 * 60)  # 24 hours ago
        
        to_remove = []
        for chat_id, data in state.items():
            if isinstance(data, dict) and 'timestamp' in data:
                if data['timestamp'] < cutoff_time:
                    to_remove.append(chat_id)
        
        for chat_id in to_remove:
            del state[chat_id]
    
    def get_pending_pair(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get pending pair data for a chat"""
        with self._lock:
            state = self._read_state()
            return state.get(chat_id)
    
    def set_rejected_response(self, chat_id: str, context: list, response: Dict[str, Any]):
        """Store rejected response for a chat"""
        with self._lock:
            state = self._read_state()
            if chat_id not in state:
                state[chat_id] = {
                    "context": context,
                    "timestamp": time.time()
                }
            state[chat_id]["rejected"] = response
            self._write_state(state)
    
    def has_complete_pair(self, chat_id: str) -> bool:
        """Check if a chat has both accepted and rejected responses"""
        pending = self.get_pending_pair(chat_id)
        if not pending:
            return False
        return "accepted" in pending and "rejected" in pending
    
    def remove_pair(self, chat_id: str):
        """Remove a completed pair from state"""
        with self._lock:
            state = self._read_state()
            if chat_id in state:
                del state[chat_id]
                self._write_state(state)

# Global instance for shared use
_shared_state = DPOSharedState()

class Action:
    """
    OpenPipe DPO Reject Response Action ❌
    
    Marks the current response as REJECTED for DPO training.
    Works together with the Accept Response action to create preference pairs.
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
        DPO_DATASET_NAME: str = Field(
            default="DPO Training Data", 
            description="Dataset name for DPO training entries"
        )
        DEFAULT_SPLIT: str = Field(
            default="TRAIN",
            description="Default split for dataset entries (TRAIN or TEST)",
            json_schema_extra={"enum": ["TRAIN", "TEST"]}
        )
        AUTO_CREATE_DATASET: bool = Field(
            default=True,
            description="Automatically create DPO dataset if it doesn't exist"
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
        self._datasets_cache = {}
        
    def _log(self, message: str, level: str = "INFO"):
        """Debug logging helper"""
        if self.valves.DEBUG_LOGGING or level in ["ERROR", "WARNING"]:
            import traceback
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [OpenPipe DPO Reject {level}] {message}")
            if level == "ERROR":
                print(f"[{timestamp}] [OpenPipe DPO Reject TRACE] {traceback.format_exc()}")
    
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
            import urllib.request
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
            
            if status_code not in [200, 201]:
                self._log(f"API error: {status_code} - {response_text}", "ERROR")
                raise Exception(f"API request failed: {status_code} - {response_text}")
            
            return json.loads(response_text)
            
        except Exception as e:
            self._log(f"API request failed: {str(e)}", "ERROR")
            raise
    
    async def _get_or_create_dpo_dataset(self) -> Dict[str, Any]:
        """Get or create the DPO dataset"""
        try:
            result = await self._make_api_request("GET", "datasets")
            datasets = result.get("data", [])
            
            # Look for existing DPO dataset
            for dataset in datasets:
                if dataset["name"] == self.valves.DPO_DATASET_NAME:
                    self._datasets_cache[dataset["name"]] = dataset
                    return dataset
            
            # Create new DPO dataset if not found
            if self.valves.AUTO_CREATE_DATASET:
                data = {"name": self.valves.DPO_DATASET_NAME}
                dataset = await self._make_api_request("POST", "datasets", data)
                self._datasets_cache[dataset["name"]] = dataset
                return dataset
            else:
                raise Exception(f"DPO dataset '{self.valves.DPO_DATASET_NAME}' not found")
                
        except Exception as e:
            self._log(f"Failed to get/create DPO dataset: {str(e)}", "ERROR")
            raise
    
    def _extract_conversation_context(self, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract conversation context (all messages except the last assistant response)"""
        messages = body.get("messages", [])
        if not messages:
            return []
        
        context_messages = []
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
                
            role = message.get("role", "")
            
            # Skip system messages if configured
            if role == "system" and not self.valves.INCLUDE_SYSTEM_MESSAGES:
                continue
            
            # Include all messages except the last assistant response
            if role == "assistant" and i == len(messages) - 1:
                break
                
            # Handle multimodal content (images)
            content = message.get("content", "")
            if isinstance(content, list):
                formatted_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            formatted_content.append({
                                "type": "text",
                                "text": item.get("text", "")
                            })
                        elif item.get("type") in ["image_url", "image"]:
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": item.get("image_url", {}).get("url", "") or item.get("image", "")
                                }
                            })
                content = formatted_content
            
            formatted_message = {
                "role": role,
                "content": content
            }
            
            # Preserve additional fields
            for field in ["name", "tool_call_id", "tool_calls", "function_call"]:
                if field in message:
                    formatted_message[field] = message[field]
                    
            context_messages.append(formatted_message)
        
        return context_messages
    
    def _extract_assistant_response(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract the last assistant response from conversation"""
        messages = body.get("messages", [])
        if not messages:
            return None
        
        last_message = messages[-1]
        if not isinstance(last_message, dict) or last_message.get("role") != "assistant":
            return None
        
        # Handle multimodal content (images)
        content = last_message.get("content", "")
        if isinstance(content, list):
            formatted_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        formatted_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item.get("type") in ["image_url", "image"]:
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": item.get("image_url", {}).get("url", "") or item.get("image", "")
                            }
                        })
            content = formatted_content
        
        response = {
            "role": "assistant",
            "content": content
        }
        
        # Preserve additional fields
        for field in ["tool_calls", "function_call", "refusal"]:
            if field in last_message:
                response[field] = last_message[field]
        
        return response
    
    def _create_dpo_entry(
        self, 
        context_messages: List[Dict[str, Any]], 
        preferred_response: Dict[str, Any],
        rejected_response: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a DPO dataset entry with preferred and rejected responses"""
        messages = context_messages + [preferred_response]
        
        entry = {
            "messages": messages,
            "rejected_message": rejected_response,
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
        Mark current response as REJECTED ❌ for DPO training
        """
        try:
            shared_state = _shared_state
            
            # Emit initial status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "❌ Marking response as REJECTED for DPO...",
                        "done": False,
                    },
                })
            
            # Extract conversation context and assistant response
            context_messages = self._extract_conversation_context(body)
            assistant_response = self._extract_assistant_response(body)
            
            if not context_messages:
                return "❌ No conversation context found"
            
            if not assistant_response:
                return "❌ No assistant response found to mark as rejected"
            
            # Get chat_id for tracking preference pairs
            chat_id = body.get("metadata", {}).get("chat_id")
            if not chat_id:
                return "❌ No chat ID found - cannot track preference pairs"
            
            # Store this response in shared state
            shared_state.set_rejected_response(chat_id, context_messages, assistant_response)
            
            # Check if we have both accepted and rejected responses
            if shared_state.has_complete_pair(chat_id):
                pending = shared_state.get_pending_pair(chat_id)
                # We have a complete pair! Send to OpenPipe
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "🎯 Complete preference pair found! Sending to OpenPipe...",
                            "done": False,
                        },
                    })
                
                # Get or create DPO dataset
                dataset = await self._get_or_create_dpo_dataset()
                
                # Prepare metadata
                metadata = {
                    "source": "openwebui_dpo_pairs",
                    "user_id": __user__.get("id") if __user__ else None,
                    "chat_id": chat_id,
                    "model": body.get("model", "unknown"),
                    "timestamp": str(int(body.get("timestamp", 0))),
                    "context_length": str(len(context_messages)),
                    "pair_complete": "true"
                }
                
                # Create DPO entry with actual preference pair
                dpo_entry = self._create_dpo_entry(
                    context_messages,
                    pending["accepted"],  # preferred
                    pending["rejected"],  # rejected
                    metadata
                )
                
                # Send to OpenPipe
                data = {"entries": [dpo_entry]}
                result = await self._make_api_request("POST", f"datasets/{dataset['id']}/entries", data)
                
                entries_created = result.get("entries_created", 0)
                errors = result.get("errors", {}).get("data", [])
                
                # Clean up pending pair
                shared_state.remove_pair(chat_id)
                
                if errors:
                    error_messages = [error.get("message", "Unknown error") for error in errors]
                    return f"⚠️ Sent complete DPO pair with errors: {'; '.join(error_messages)}"
                
                # Success feedback
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "🎯 Complete DPO preference pair sent successfully!",
                            "done": True,
                        },
                    })
                
                self._log(f"Successfully sent complete DPO pair for chat {chat_id}")
                return f"🎯 Complete DPO preference pair sent to '{dataset['name']}' ({entries_created} entries created)"
            
            else:
                # Incomplete pair - waiting for the accepted response
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "❌ Response marked as REJECTED. Waiting for ACCEPTED response...",
                            "done": True,
                        },
                    })
                
                self._log(f"Stored rejected response for chat {chat_id}, waiting for accepted")
                return "❌ Response marked as REJECTED! Now use the ✅ Accept Response button on another response to complete the preference pair."
            
        except Exception as e:
            error_msg = f"❌ Failed to mark response as rejected: {str(e)}"
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
