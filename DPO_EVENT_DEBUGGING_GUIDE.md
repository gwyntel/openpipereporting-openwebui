# DPO Event System Debugging Guide

## Overview

The OpenPipe DPO Accept and Reject actions have been enhanced with comprehensive debugging and improved event communication to fix the cross-action coordination issues.

## Key Improvements in v1.4.0

### 1. Enhanced Debug Logging

Both actions now include extensive debug logging that can be enabled via the `DEBUG_LOGGING` valve:

```python
DEBUG_LOGGING: bool = Field(
    default=False,
    description="Enable debug logging"
)
```

When enabled, you'll see detailed logs in the OpenWebUI container logs including:
- Action start/end markers
- Request body structure analysis
- Pending pairs state tracking
- Event emission details
- API request/response information
- Error traces with full stack traces

### 2. Comprehensive Event System

The actions now emit multiple types of events for better compatibility and debugging:

#### Core Action Events
- `dpo:accepted` - Emitted when a response is marked as accepted
- `dpo:rejected` - Emitted when a response is marked as rejected

#### Cross-Action Coordination Events
- `dpo:waiting_for_reject` - Emitted by accept action when waiting for reject
- `dpo:waiting_for_accept` - Emitted by reject action when waiting for accept
- `dpo:pair_complete` - Emitted when a complete pair is sent to OpenPipe

#### UI Feedback Events
- `notification` events with success/warning/error types
- `status` events for progress tracking
- Debug notification events (when debug logging enabled)

### 3. Improved State Management

- **Global Shared State**: Both actions use the same `_pending_pairs` global dictionary
- **Timestamps**: Added `accepted_timestamp` and `rejected_timestamp` for tracking
- **Better Chat ID Detection**: Fallback locations for chat_id extraction
- **Enhanced Metadata**: More comprehensive metadata in DPO entries

### 4. Robust Error Handling

- Detailed error logging with stack traces
- User-friendly error messages
- Graceful fallback when events fail
- API error handling with detailed responses

## Testing the Event System

### Step 1: Enable Debug Logging

1. Go to OpenWebUI Settings → Actions
2. Find both "OpenPipe DPO Accept Response" and "OpenPipe DPO Reject Response"
3. Set `DEBUG_LOGGING` to `true` for both actions
4. Configure your OpenPipe API key and other settings

### Step 2: Monitor Container Logs

Open a terminal and monitor the OpenWebUI container logs:

```bash
# For Docker Compose
docker-compose logs -f open-webui

# For Docker
docker logs -f <container-name>
```

### Step 3: Test the DPO Workflow

1. **Start a conversation** with an AI model in OpenWebUI
2. **Generate a response** from the AI
3. **Click the Accept button (✅)** on the response
   - Look for logs: `=== ACCEPT ACTION STARTED ===`
   - Check for: `Stored accepted response for chat {chat_id}, waiting for rejected`
   - Verify events: `dpo:accepted`, `dpo:waiting_for_reject`

4. **Generate another response** (or regenerate the same one)
5. **Click the Reject button (❌)** on a different response
   - Look for logs: `=== REJECT ACTION STARTED ===`
   - Check for: `Complete DPO pair found for chat {chat_id}!`
   - Verify events: `dpo:rejected`, `dpo:pair_complete`

### Step 4: Verify OpenPipe Integration

If the event system is working correctly, you should see:
- `🎯 Complete DPO preference pair sent to 'DPO Training Data' (1 entries created)`
- Logs showing successful API request to OpenPipe
- The pending pair being cleaned up from memory

## Debugging Common Issues

### Issue 1: No Chat ID Found

**Symptoms**: `❌ No chat ID found - cannot track preference pairs`

**Debug Steps**:
1. Check logs for: `Request body structure`
2. Look for `metadata_keys` in the debug output
3. Verify the request contains proper metadata

**Solution**: The actions now try multiple locations for chat_id:
- `body.metadata.chat_id`
- `body.chat_id`
- `body.id`

### Issue 2: Actions Not Communicating

**Symptoms**: Both actions work individually but don't create pairs

**Debug Steps**:
1. Enable debug logging on both actions
2. Check for `Current pending pairs state` logs
3. Verify both actions are using the same chat_id
4. Look for event emission logs

**Solution**: The actions use a global `_pending_pairs` dictionary that should be shared between action instances.

### Issue 3: Events Not Being Received

**Symptoms**: Events are emitted but not received by other actions

**Debug Steps**:
1. Check if custom events (`dpo:accepted`, `dpo:rejected`) are supported
2. Look for notification events in the UI
3. Verify the global state is being updated

**Solution**: The actions primarily rely on shared global state rather than events for coordination, with events used for UI feedback and potential future inter-action communication.

## Event Data Structures

### dpo:accepted Event
```json
{
  "type": "dpo:accepted",
  "data": {
    "chat_id": "chat_123",
    "action": "accepted",
    "timestamp": 1704067200.123,
    "context_length": 5,
    "response_preview": "This is a sample response..."
  }
}
```

### dpo:rejected Event
```json
{
  "type": "dpo:rejected",
  "data": {
    "chat_id": "chat_123",
    "action": "rejected",
    "timestamp": 1704067200.456,
    "context_length": 5,
    "response_preview": "This is another response..."
  }
}
```

### dpo:pair_complete Event
```json
{
  "type": "dpo:pair_complete",
  "data": {
    "chat_id": "chat_123",
    "dataset_name": "DPO Training Data",
    "entries_created": 1,
    "success": true
  }
}
```

## Troubleshooting Tips

1. **Check OpenWebUI Version**: Ensure you're running OpenWebUI 0.5.0 or later
2. **Verify API Key**: Make sure your OpenPipe API key is correctly configured
3. **Monitor Network**: Check for API connectivity issues to OpenPipe
4. **Clear State**: Restart OpenWebUI to clear any stale pending pairs
5. **Test Individually**: Try each action separately to isolate issues

## Expected Log Flow

For a successful DPO pair creation, you should see logs like:

```
[ACCEPT] === ACCEPT ACTION STARTED ===
[ACCEPT] Processing chat_id: chat_123
[ACCEPT] Stored accepted response for chat chat_123, waiting for rejected
[REJECT] === REJECT ACTION STARTED ===
[REJECT] Processing chat_id: chat_123
[REJECT] Complete DPO pair found for chat chat_123!
[REJECT] Using dataset: DPO Training Data (ID: dataset_456)
[REJECT] Created DPO entry with 5 context messages
[REJECT] OpenPipe API result: 1 entries created, 0 errors
[REJECT] Successfully sent complete DPO pair for chat chat_123
```

This enhanced debugging system should help identify exactly where the event communication is failing and provide the information needed to fix the cross-action coordination issues.
