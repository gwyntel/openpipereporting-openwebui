"""
Shared state management for OpenPipe DPO Accept/Reject actions.
Provides file-based persistence to enable cross-module state sharing.
"""

import json
import os
import time
import threading
from typing import Dict, Any, Optional
from pathlib import Path

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
    
    def set_accepted_response(self, chat_id: str, context: list, response: Dict[str, Any]):
        """Store accepted response for a chat"""
        with self._lock:
            state = self._read_state()
            if chat_id not in state:
                state[chat_id] = {
                    "context": context,
                    "timestamp": time.time()
                }
            state[chat_id]["accepted"] = response
            self._write_state(state)
    
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about pending pairs"""
        with self._lock:
            state = self._read_state()
            total_pending = len(state)
            complete_pairs = sum(1 for data in state.values() 
                               if isinstance(data, dict) and 
                               "accepted" in data and "rejected" in data)
            incomplete_pairs = total_pending - complete_pairs
            
            return {
                "total_pending": total_pending,
                "complete_pairs": complete_pairs,
                "incomplete_pairs": incomplete_pairs
            }

# Global instance for shared use
_shared_state = DPOSharedState()

def get_shared_state() -> DPOSharedState:
    """Get the global shared state instance"""
    return _shared_state
