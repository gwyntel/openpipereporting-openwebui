# OpenPipe Reporting Filter for OpenWebUI

A comprehensive filter for OpenWebUI that automatically reports all AI interactions to OpenPipe for detailed analytics and monitoring.

## Features

### 📊 Comprehensive Reporting
- **Interaction Classification**: Automatically detects and reports different types of AI interactions:
  - Regular chat messages
  - System tasks (title generation, tags, follow-ups, etc.)
  - Tool/function calling interactions
  - Image prompt generation
  - Query optimization
  - And more...

### ⚡ Performance Tracking
- **Precise Inference Timing**: Measures exact time from request to response
- **Tool Usage Analytics**: Tracks which tools are used, how often, and success rates
- **Model Performance**: Compare performance across different models and providers

### 🔧 Smart Detection
- **Zero Content Analysis**: Uses OpenWebUI's built-in metadata for efficient classification
- **Universal Format Support**: Works with all models since OpenWebUI normalizes to OpenAI format
- **Tool Call Extraction**: Identifies and reports function/tool usage patterns

### ⚙️ Flexible Configuration
- **Selective Reporting**: Choose which interaction types to report
- **Custom Metadata**: Add your own metadata fields
- **Debug Mode**: Detailed logging for troubleshooting
- **Timeout Control**: Configurable API timeout settings

## Installation

### Option 1: Docker Volume Mount (Recommended)

If you're running OpenWebUI in Docker with a data directory already mounted:

1. **Copy the filter to your mounted data directory**:
   ```bash
   # If your data directory is mounted at ./open-webui-data
   mkdir -p ./open-webui-data/functions
   cp openpipe_reporting_filter.py ./open-webui-data/functions/
   ```

2. **Your Docker setup should look like**:
   ```bash
   docker run -d \
     -p 3000:8080 \
     -v ./open-webui-data:/app/backend/data \
     --name open-webui \
     ghcr.io/open-webui/open-webui:main
   ```

   Or in `docker-compose.yml`:
   ```yaml
   services:
     open-webui:
       image: ghcr.io/open-webui/open-webui:main
       ports:
         - "3000:8080"
       volumes:
         - ./open-webui-data:/app/backend/data
   ```

3. **Activate the filter**:
   - Go to Admin Settings → Functions
   - The "OpenPipe Reporting Filter" should appear
   - Toggle it on and configure your API key

### Option 1b: If You Don't Have Functions Directory Yet

If you already have data mounted but no functions directory:

1. **Create the functions directory and copy the filter**:
   ```bash
   # Replace ./your-data-dir with your actual mounted data directory path
   mkdir -p ./your-data-dir/functions
   cp openpipe_reporting_filter.py ./your-data-dir/functions/
   ```

2. **Restart OpenWebUI** to detect the new function:
   ```bash
   docker restart open-webui
   ```

### Option 2: Direct Upload

1. **Upload via Web Interface**:
   - Go to Admin Settings → Functions
   - Click "+" to add a new function
   - Upload `openpipe_reporting_filter.py`
   - The filter will appear as "OpenPipe Reporting Filter"

### Option 3: Manual Installation

1. **Copy the filter file** to your OpenWebUI functions directory:
   ```bash
   # For local installation
   cp openpipe_reporting_filter.py /path/to/your/openwebui/backend/data/functions/
   
   # For Docker (exec into container)
   docker cp openpipe_reporting_filter.py open-webui:/app/backend/data/functions/
   ```

## Configuration

### Required Settings
1. **OPENPIPE_API_KEY**: Your OpenPipe API key (get it from [OpenPipe Dashboard](https://app.openpipe.ai))
2. **ENABLED**: Set to `true` to activate reporting

### Core Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `OPENPIPE_API_KEY` | `""` | Your OpenPipe API authentication key |
| `OPENPIPE_BASE_URL` | `https://api.openpipe.ai/api/v1` | OpenPipe API base URL |
| `ENABLED` | `true` | Master switch to enable/disable reporting |

### Feature Toggles

| Setting | Default | Description |
|---------|---------|-------------|
| `INCLUDE_TOOL_METRICS` | `true` | Track and report tool usage analytics |
| `INCLUDE_TIMING_METRICS` | `true` | Include precise inference timing data |
| `INCLUDE_INTERACTION_CLASSIFICATION` | `true` | Classify interaction types |

### Selective Reporting

| Setting | Default | Description |
|---------|---------|-------------|
| `REPORT_CHAT_MESSAGES` | `true` | Report regular user-AI conversations |
| `REPORT_SYSTEM_TASKS` | `true` | Report system tasks (title gen, etc.) |
| `REPORT_TOOL_INTERACTIONS` | `true` | Report tool/function calling interactions |

### Advanced Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `DEBUG_LOGGING` | `false` | Enable detailed console logging |
| `CUSTOM_METADATA` | `"{}"` | Additional metadata as JSON string |
| `TIMEOUT_SECONDS` | `10` | API request timeout |

## Interaction Types Detected

The filter automatically classifies interactions using OpenWebUI's built-in system:

### System Tasks
- `title_generation`: Auto-generated chat titles
- `follow_up_generation`: Follow-up question suggestions
- `tags_generation`: Chat categorization tags
- `emoji_generation`: Emoji suggestions
- `query_generation`: Search/retrieval query optimization
- `image_prompt_generation`: Image generation prompt enhancement
- `autocomplete_generation`: Text autocompletion
- `moa_response_generation`: Mixture of Agents responses

### User Interactions
- `chat_message`: Regular user conversations
- `function_calling`: Tool/function usage

## Reported Data Structure

Each interaction sends the following data to OpenPipe:

```json
{
  "requestedAt": 1673123456789,
  "receivedAt": 1673123457890,
  "reqPayload": { /* Original request */ },
  "respPayload": { /* AI response */ },
  "statusCode": 200,
  "metadata": {
    "interaction_type": "chat_message",
    "inference_time_ms": 1101,
    "tools_used": ["search", "calculator"],
    "tool_call_count": 2,
    "has_tool_calls": true,
    "user_id": "user123",
    "chat_id": "chat456"
  }
}
```

## Analytics Capabilities

With this data in OpenPipe, you can analyze:

### Performance Metrics
- Average inference times by model
- Performance differences between interaction types
- Tool usage impact on response times

### Usage Patterns
- Most frequently used system features
- Tool adoption and usage patterns
- User behavior analysis

### Model Comparison
- Performance benchmarking across models
- Tool compatibility across providers
- Cost analysis per interaction type

## Examples

### Basic Configuration
```json
{
  "OPENPIPE_API_KEY": "your-api-key-here",
  "ENABLED": true,
  "DEBUG_LOGGING": false
}
```

### Development/Testing Setup
```json
{
  "OPENPIPE_API_KEY": "your-api-key-here",
  "ENABLED": true,
  "DEBUG_LOGGING": true,
  "REPORT_SYSTEM_TASKS": false,
  "CUSTOM_METADATA": "{\"environment\": \"development\"}"
}
```

### Production with Custom Metadata
```json
{
  "OPENPIPE_API_KEY": "your-api-key-here",
  "ENABLED": true,
  "CUSTOM_METADATA": "{\"deployment\": \"production\", \"version\": \"2.1.0\"}"
}
```

## Troubleshooting

### Enable Debug Logging
Set `DEBUG_LOGGING` to `true` to see detailed logs in the console:
```
[OpenPipe Filter INFO] Captured request 1673123456_12345 - Type: chat_message
[OpenPipe Filter INFO] Sending report to OpenPipe: report
[OpenPipe Filter INFO] Successfully reported to OpenPipe
```

### Common Issues

1. **No reports appearing**: Check that `ENABLED` is `true` and `OPENPIPE_API_KEY` is set
2. **API errors**: Verify your API key is correct and has proper permissions
3. **Missing tool data**: Ensure `INCLUDE_TOOL_METRICS` is enabled
4. **Timing issues**: Check that `INCLUDE_TIMING_METRICS` is enabled

### API Endpoint
The filter uses OpenPipe's `/report` endpoint since OpenWebUI standardizes all models to OpenAI format internally.

## Performance Considerations

- **Asynchronous Reporting**: API calls don't block chat responses
- **Efficient Classification**: Uses OpenWebUI's existing metadata (no content analysis)
- **Memory Management**: Request data is cleaned up after reporting
- **Error Handling**: Failures don't affect chat functionality

## Security & Privacy

- **API Key Protection**: Store your OpenPipe API key securely
- **Data Transmission**: All data is sent over HTTPS
- **No Content Modification**: Filter only observes, never modifies conversations
- **User Privacy**: Only reports data you configure (user IDs are optional)

## Contributing

To extend or modify the filter:

1. **Add new interaction types**: Update `_classify_interaction()` method
2. **Add custom metadata**: Extend the metadata building logic
3. **Support new providers**: Update `_determine_model_provider()` method

## License

This filter is provided as-is for integration with OpenWebUI and OpenPipe. Ensure compliance with your organization's data policies before deployment.
