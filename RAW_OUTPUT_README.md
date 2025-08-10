# OpenPipe Reporting Filter v2.0 - Raw Model Output

This document describes the enhanced version of the OpenPipe reporting filter that attempts to capture raw model output for more accurate analytics.

## What's New in Version 2.0

### Raw Model Output Capture
- Enhanced response capture that preserves message structure closer to raw format
- Maintains tool calls, function calls, and response metadata in original structure
- Attempts to intercept raw response data when available
- Preserves usage information and system fingerprints

### Improved Request/Response Tracking
- Better request ID tracking using metadata instead of content scanning
- Enhanced request payload preservation (removes filter-specific metadata)
- More accurate timing measurements for inference

### Enhanced Metadata
- Indicates capture method (`filter_level` vs `router_level`)
- Version information for tracking filter capabilities
- Raw response availability indicators
- Better tool usage analytics

## Important Limitations

### Filter-Level vs Router-Level Capture

**Current Implementation (Filter-Level):**
- ✅ Captures responses in the most raw format available at the filter level
- ✅ Preserves tool calls, function calls, and message structure
- ✅ Works with existing OpenWebUI installations without code changes
- ⚠️ Still sees processed conversation data, not direct HTTP responses
- ⚠️ Cannot access streaming response chunks directly
- ⚠️ Limited access to usage tokens and response metadata

**True Raw Capture (Router-Level):**
- ✅ Direct access to HTTP requests/responses from model providers
- ✅ Captures streaming responses and reconstructs complete responses  
- ✅ Full access to usage tokens, timing, and response metadata
- ❌ Requires modifying OpenWebUI core code (see `open-webui/routers/openai.py`)
- ❌ More complex installation and maintenance

### Metadata Indicators

The filter now includes metadata to indicate the capture method:
```json
{
  "capture_method": "filter_level",
  "openwebui_version": "filter_v2.0", 
  "raw_response_available": false
}
```

## Migration from v1.0

The v2.0 filter is backward compatible with v1.0 configurations. Key improvements:

1. **Better Response Capture**: Messages are now captured in a format closer to raw API responses
2. **Enhanced Tracking**: Request IDs are stored in metadata for more reliable matching
3. **Raw Content Preservation**: Tool calls and function calls are preserved in their original structure
4. **Improved Debugging**: Better logging and error handling

## Configuration

All existing configuration options remain the same. The filter automatically uses the enhanced capture methods.

## For True Raw Model Output

If you need true raw model output (direct HTTP responses from model providers), consider:

1. **Router-Level Integration**: Implement the reporting at the OpenWebUI router level (see reference implementation in `open-webui/backend/open_webui/routers/openai.py`)

2. **Hybrid Approach**: Use both the filter for convenience and router-level integration for critical raw data capture

3. **Custom Middleware**: Implement custom middleware that intercepts HTTP requests/responses before OpenWebUI processing

## Example Output Differences

### v1.0 Response Format
```json
{
  "choices": [{
    "message": {
      "role": "assistant", 
      "content": "Processed OpenWebUI response text"
    }
  }]
}
```

### v2.0 Enhanced Response Format  
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Response text preserved in original structure",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function", 
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco\"}"
          }
        }
      ]
    }
  }],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 25, 
    "total_tokens": 75
  }
}
```

## Verification

To verify the enhanced capture is working:

1. Enable debug logging: `DEBUG_LOGGING: true`
2. Look for log messages: `"Reporting raw interaction: ... [Filter-level capture]"`
3. Check OpenPipe dashboard for metadata fields: `capture_method`, `openwebui_version`
4. Verify tool calls are preserved in their original structure

## Next Steps

For organizations requiring true raw model output, we recommend implementing router-level integration based on the reference implementation in `open-webui/backend/open_webui/routers/openai.py`.
