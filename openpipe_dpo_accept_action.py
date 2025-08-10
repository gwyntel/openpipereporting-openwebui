"""
title: OpenPipe DPO Accept Response
author: Cline & Gwyn
version: 1.6.1
required_open_webui_version: 0.5.0
icon_url: data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgZmlsbD0iIzAwODAwMCIgc3Ryb2tlPSIjMDA2MDAwIiBzdHJva2Utd2lkdGg9IjIiLz4KICA8cGF0aCBkPSJtOSAxMiAyIDIgNC00IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjMiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# In-memory storage for accepted responses (event-based communication)
_accepted_responses = {}

class Action:
    """
    OpenPipe DPO Accept Response Action ✅
    
    Marks the current response as ACCEPTED/PREFERRED for DPO training.
    Works together with the Reject Response action to create preference pairs.
    Uses pure event-based communication without global state.
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
            print(f"[{timestamp}] [OpenPipe DPO Accept {level}] {message}")
            if level == "ERROR":
                print(f"[{timestamp}] [OpenPipe DPO Accept TRACE] {traceback.format_exc()}")
    
    async def _emit_debug_event(self, __event_emitter__, message: str, data: dict = None):
        """Emit debug notification event"""
        if __event_emitter__ and self.valves.DEBUG_LOGGING:
            await __event_emitter__({
                "type": "notification",
                "data": {
                    "type": "info",
                    "content": f"[DEBUG ACCEPT] {message}"
                }
            })
            if data:
                self._log(f"Debug data: {json.dumps(data, indent=2, default=str)}", "DEBUG")
    
    async def _check_for_rejected_response(self, __event_call__, chat_id: str, context_messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Check if there's a rejected response for this chat via event call"""
        if not __event_call__:
            return None
            
        try:
            self._log(f"Checking for rejected response for chat {chat_id}", "DEBUG")
            
            # Use event call to check for rejected response
            result = await __event_call__({
                "type": "dpo:check_rejected",
                "data": {
                    "chat_id": chat_id,
                    "context_hash": hash(str(context_messages)),
                    "requester": "accept_action"
                }
            })
            
            if result and isinstance(result, dict) and "rejected_response" in result:
                self._log(f"Found rejected response for chat {chat_id}", "DEBUG")
                return result["rejected_response"]
            else:
                self._log(f"No rejected response found for chat {chat_id}", "DEBUG")
                return None
                
        except Exception as e:
            self._log(f"Error checking for rejected response: {str(e)}", "DEBUG")
            return None
    
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
        Mark current response as ACCEPTED ✅ for DPO training
        """
        try:
            # Debug: Log action start
            self._log("=== ACCEPT ACTION STARTED (Event-Based) ===", "DEBUG")
            await self._emit_debug_event(__event_emitter__, "Accept action started (event-based)")
            
            # Debug: Log body structure (without sensitive data)
            debug_body = {
                "has_messages": "messages" in body,
                "message_count": len(body.get("messages", [])),
                "has_metadata": "metadata" in body,
                "metadata_keys": list(body.get("metadata", {}).keys()) if "metadata" in body else [],
                "model": body.get("model", "unknown")
            }
            self._log(f"Request body structure: {json.dumps(debug_body, indent=2)}", "DEBUG")
            
            # Emit initial status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": "✅ Marking response as ACCEPTED for DPO...",
                        "done": False,
                    },
                })
            
            # Extract conversation context and assistant response
            context_messages = self._extract_conversation_context(body)
            assistant_response = self._extract_assistant_response(body)
            
            # Debug: Log extraction results
            self._log(f"Extracted {len(context_messages)} context messages", "DEBUG")
            self._log(f"Assistant response extracted: {assistant_response is not None}", "DEBUG")
            if assistant_response:
                content_preview = str(assistant_response.get("content", ""))[:100] + "..." if len(str(assistant_response.get("content", ""))) > 100 else str(assistant_response.get("content", ""))
                self._log(f"Assistant response preview: {content_preview}", "DEBUG")
            
            if not context_messages:
                await self._emit_debug_event(__event_emitter__, "No conversation context found")
                return "❌ No conversation context found"
            
            if not assistant_response:
                await self._emit_debug_event(__event_emitter__, "No assistant response found")
                return "❌ No assistant response found to mark as accepted"
            
            # Create pairing key based on user's message content for more precise matching
            user_message = None
            for msg in context_messages:
                if msg.get("role") == "user":
                    user_message = msg.get("content", "")
                    break
            
            if not user_message:
                await self._emit_debug_event(__event_emitter__, "No user message found for pairing")
                return "❌ No user message found - cannot create preference pairs"
            
            # Create a hash of the user message for pairing
            import hashlib
            user_message_str = str(user_message) if isinstance(user_message, str) else str(user_message)
            pairing_key = hashlib.md5(user_message_str.encode()).hexdigest()[:16]
            
            self._log(f"Processing pairing key: {pairing_key} (from user message: {user_message_str[:50]}...)", "DEBUG")
            await self._emit_debug_event(__event_emitter__, f"Processing pairing key {pairing_key}")
            
            # Store accepted response for direct lookup by reject action
            global _accepted_responses
            _accepted_responses[pairing_key] = {
                "response": assistant_response.copy(),
                "context": context_messages,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            self._log(f"Stored accepted response with pairing key: {pairing_key}", "DEBUG")
            
            # Emit accepted response event
            if __event_emitter__:
                await __event_emitter__({
                    "type": "dpo:accepted",
                    "data": {
                        "pairing_key": pairing_key,
                        "action": "accepted",
                        "timestamp": asyncio.get_event_loop().time(),
                        "user_message": user_message_str[:100],
                        "accepted_response": assistant_response
                    }
                })
                
                # Notification event for visibility
                await __event_emitter__({
                    "type": "notification",
                    "data": {
                        "type": "success",
                        "content": f"✅ Response accepted for user message: {user_message_str[:50]}..."
                    }
                })
            
            # Check for existing rejected response by looking in reject action's storage
            rejected_response = None
            try:
                # Import the reject action's storage
                from openpipe_dpo_reject_action import _rejected_responses
                if pairing_key in _rejected_responses:
                    rejected_data = _rejected_responses[pairing_key]
                    rejected_response = rejected_data["response"]
                    self._log(f"Found matching rejected response for pairing key: {pairing_key}", "DEBUG")
            except ImportError:
                self._log("Could not import reject action storage, checking via events", "DEBUG")
            
            if rejected_response:
                # We have a complete pair! Send to OpenPipe
                self._log(f"Complete DPO pair found for pairing key {pairing_key}!", "INFO")
                
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "🎯 Complete preference pair found! Sending to OpenPipe...",
                            "done": False,
                        },
                    })
                    
                    await __event_emitter__({
                        "type": "notification",
                        "data": {
                            "type": "info",
                            "content": "🎯 Complete DPO pair ready - sending to OpenPipe..."
                        }
                    })
                
                # Get or create DPO dataset
                dataset = await self._get_or_create_dpo_dataset()
                self._log(f"Using dataset: {dataset.get('name', 'unknown')} (ID: {dataset.get('id', 'unknown')})", "DEBUG")
                
                # Prepare metadata
                metadata = {
                    "source": "openwebui_dpo_pairs_user_message_based",
                    "user_id": __user__.get("id") if __user__ else None,
                    "pairing_key": pairing_key,
                    "user_message_preview": user_message_str[:100],
                    "model": body.get("model", "unknown"),
                    "timestamp": str(int(body.get("timestamp", 0))),
                    "context_length": str(len(context_messages)),
                    "pair_complete": "true",
                    "communication_method": "user_message_pairing"
                }
                
                # Create DPO entry with actual preference pair
                dpo_entry = self._create_dpo_entry(
                    context_messages,
                    assistant_response,  # preferred (accepted)
                    rejected_response,   # rejected
                    metadata
                )
                
                self._log(f"Created DPO entry with {len(context_messages)} context messages", "DEBUG")
                
                # Send to OpenPipe
                data = {"entries": [dpo_entry]}
                result = await self._make_api_request("POST", f"datasets/{dataset['id']}/entries", data)
                
                entries_created = result.get("entries_created", 0)
                errors = result.get("errors", {}).get("data", [])
                
                self._log(f"OpenPipe API result: {entries_created} entries created, {len(errors)} errors", "DEBUG")
                
                # Emit cleanup event
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "dpo:cleanup",
                        "data": {
                            "pairing_key": pairing_key,
                            "user_message": user_message_str[:50]
                        }
                    })
                
                if errors:
                    error_messages = [error.get("message", "Unknown error") for error in errors]
                    error_summary = f"⚠️ Sent complete DPO pair with errors: {'; '.join(error_messages)}"
                    self._log(f"API errors: {error_messages}", "WARNING")
                    
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "notification",
                            "data": {
                                "type": "warning",
                                "content": error_summary
                            }
                        })
                    
                    return error_summary
                
                # Success feedback
                success_msg = f"🎯 Complete DPO preference pair sent to '{dataset['name']}' ({entries_created} entries created)"
                
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "🎯 Complete DPO preference pair sent successfully!",
                            "done": True,
                        },
                    })
                    
                    await __event_emitter__({
                        "type": "notification",
                        "data": {
                            "type": "success",
                            "content": success_msg
                        }
                    })
                    
                    # Emit completion event
                    await __event_emitter__({
                        "type": "dpo:pair_complete",
                        "data": {
                            "pairing_key": pairing_key,
                            "dataset_name": dataset['name'],
                            "entries_created": entries_created,
                            "success": True
                        }
                    })
                
                self._log(f"Successfully sent complete DPO pair for pairing key {pairing_key}", "INFO")
                return success_msg
            
            else:
                # Incomplete pair - waiting for the rejected response
                waiting_msg = "✅ Response marked as ACCEPTED! Now use the ❌ Reject Response button on another response to complete the preference pair."
                
                if __event_emitter__:
                    await __event_emitter__({
                        "type": "status",
                        "data": {
                            "description": "✅ Response marked as ACCEPTED. Waiting for REJECTED response...",
                            "done": True,
                        },
                    })
                    
                    # Emit waiting event for potential cross-action communication
                    await __event_emitter__({
                        "type": "dpo:waiting_for_reject",
                        "data": {
                            "pairing_key": pairing_key,
                            "has_accepted": True,
                            "has_rejected": False,
                            "timestamp": asyncio.get_event_loop().time()
                        }
                    })
                
                self._log(f"Stored accepted response for pairing key {pairing_key}, waiting for rejected", "INFO")
                await self._emit_debug_event(__event_emitter__, f"Waiting for rejected response for pairing key {pairing_key}")
                
                return waiting_msg
            
        except Exception as e:
            error_msg = f"❌ Failed to mark response as accepted: {str(e)}"
            self._log(error_msg, "ERROR")
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {
                        "description": error_msg,
                        "done": True,
                    },
                })
                
                await __event_emitter__({
                    "type": "notification",
                    "data": {
                        "type": "error",
                        "content": error_msg
                    }
                })
            
            return error_msg
