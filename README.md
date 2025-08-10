# OpenPipe Integration for OpenWebUI

A comprehensive OpenPipe integration suite for OpenWebUI that provides both real-time conversation reporting and dataset collection for AI model training and analytics.

## 🚀 Overview

This integration consists of two complementary components:

1. **OpenPipe Universal Reporting Filter (v3.0)** - Automatic real-time reporting of all AI interactions
2. **OpenPipe Dataset Action (v1.1)** - Manual collection of conversations for training datasets

## 📦 Components

### 1. OpenPipe Universal Reporting Filter (`openpipe_filter_v3.py`)

The main reporting component that automatically captures all AI interactions and sends them to OpenPipe for analytics and monitoring.

**Key Features:**
- ✅ **Universal Model Support**: Works with ALL models (OpenAI, Anthropic, local models, etc.)
- 🎛️ **UI Toggle Control**: Easy on/off switch directly in the OpenWebUI interface
- 📊 **Real-time Feedback**: Live status updates during reporting process
- 🔍 **Enhanced Analytics**: Comprehensive interaction classification and metrics
- 🛠️ **Tool Usage Tracking**: Detailed function/tool call monitoring
- ⏱️ **Timing Metrics**: Inference time measurements
- 🎨 **Custom OpenPipe Icon**: Visual indicator in the UI
- 📝 **Raw Response Capture**: Preserves complete conversation data
- 🎯 **DPO Detection**: Identifies message regeneration for preference optimization

### 2. OpenPipe Dataset Action (`openpipe_dataset_action.py`)

An action component that allows users to manually add conversations to OpenPipe datasets for training purposes.

**Key Features:**
- 📚 **Training Data Collection**: Add conversations to OpenPipe datasets for fine-tuning
- 🎭 **Multimodal Support**: Handles text, images, audio, and tool interactions
- 🆕 **Auto Dataset Creation**: Automatically creates datasets when needed
- ⚙️ **Flexible Configuration**: Customizable dataset names and splits
- 🏷️ **Metadata Preservation**: Maintains conversation context and user information
- 🔧 **Tool Call Preservation**: Maintains function calls and responses for training

## 🛠️ Installation

### Prerequisites
- OpenWebUI v0.5.0 or higher
- OpenPipe API account and API key
- Python environment with required dependencies

### Setup Steps

1. **Install the Filter**
   - Navigate to **Admin → Settings → Functions**
   - Click **"+"** to add a new function
   - Upload or paste `openpipe_filter_v3.py`
   - The filter will appear as "OpenPipe Universal Reporting (Raw Output Capture)"

2. **Install the Action**
   - Navigate to **Admin → Settings → Functions**
   - Click **"+"** to add a new function
   - Upload or paste `openpipe_dataset_action.py`
   - The action will appear as "OpenPipe Dataset Collection (Training Data)"

3. **Configure API Keys**
   - Go to **Settings → Functions → OpenPipe Universal Reporting**
   - Enter your OpenPipe API key in the `OPENPIPE_API_KEY` field
   - Go to **Settings → Functions → OpenPipe Dataset Collection**
   - Enter your OpenPipe API key in the `OPENPIPE_API_KEY` field

## ⚙️ Configuration

### Filter Configuration (Valves)

#### Core Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `OPENPIPE_API_KEY` | `""` | Your OpenPipe API authentication key |
| `OPENPIPE_BASE_URL` | `https://api.openpipe.ai/api/v1` | OpenPipe API endpoint |

#### Feature Toggles
| Setting | Default | Description |
|---------|---------|-------------|
| `INCLUDE_TOOL_METRICS` | `true` | Track function/tool usage analytics |
| `INCLUDE_TIMING_METRICS` | `true` | Measure inference timing |
| `INCLUDE_INTERACTION_CLASSIFICATION` | `true` | Classify interaction types |

#### Selective Reporting
| Setting | Default | Description |
|---------|---------|-------------|
| `REPORT_CHAT_MESSAGES` | `true` | Report regular conversations |
| `REPORT_SYSTEM_TASKS` | `true` | Report system tasks (title generation, etc.) |
| `REPORT_TOOL_INTERACTIONS` | `true` | Report function/tool calling |

#### DPO Configuration
| Setting | Default | Description |
|---------|---------|-------------|
| `ENABLE_DPO_DETECTION` | `true` | Detect message regeneration for DPO pairs |
| `AUTO_SUBMIT_DPO_PAIRS` | `false` | Auto-submit DPO pairs to OpenPipe |
| `DPO_DATASET_NAME` | `"OpenWebUI DPO Training"` | Dataset name for DPO pairs |

#### Advanced Options
| Setting | Default | Description |
|---------|---------|-------------|
| `DEBUG_LOGGING` | `false` | Enable detailed debug output |
| `CUSTOM_METADATA` | `"{}"` | Additional metadata as JSON string |
| `TIMEOUT_SECONDS` | `10` | API request timeout |

### Dataset Action Configuration

#### Core Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `OPENPIPE_API_KEY` | `""` | Your OpenPipe API authentication key |
| `OPENPIPE_BASE_URL` | `https://api.openpipe.ai/api/v1` | OpenPipe API endpoint |
| `DEFAULT_DATASET_NAME` | `"OpenWebUI Conversations"` | Default dataset name |
| `DEFAULT_SPLIT` | `"TRAIN"` | Default split (TRAIN or TEST) |

#### Feature Settings
| Setting | Default | Description |
|---------|---------|-------------|
| `AUTO_CREATE_DATASET` | `true` | Auto-create dataset if it doesn't exist |
| `INCLUDE_SYSTEM_MESSAGES` | `true` | Include system messages in datasets |
| `DEBUG_LOGGING` | `false` | Enable debug logging |

## 🎯 Usage

### Automatic Reporting (Filter)

Once installed and configured, the filter automatically:

1. **Captures all interactions** - Every chat message, tool call, and system task
2. **Provides real-time feedback** - Shows reporting status in the UI
3. **Classifies interactions** - Identifies chat messages, function calls, system tasks
4. **Measures performance** - Tracks inference timing and tool usage
5. **Reports to OpenPipe** - Sends structured data for analytics

**UI Controls:**
- **Toggle Switch**: Enable/disable reporting without configuration changes
- **Status Updates**: Real-time feedback during the reporting process
- **Custom Icon**: OpenPipe branding in the interface

### Manual Dataset Collection (Action)

To add a conversation to a training dataset:

1. **During or after a conversation**, click the action button
2. **Select "OpenPipe Dataset Collection"** from the actions menu
3. **The action will**:
   - Extract all messages from the conversation
   - Preserve multimodal content (images, audio, tool calls)
   - Add metadata (user ID, model, timestamp, etc.)
   - Create the dataset if it doesn't exist
   - Add the conversation as a training entry

## 📊 Interaction Types Detected

The filter automatically classifies interactions:

### System Tasks
- `title_generation` - Auto-generated chat titles
- `follow_up_generation` - Follow-up question suggestions
- `tags_generation` - Chat categorization tags
- `emoji_generation` - Emoji suggestions
- `query_generation` - Search/retrieval query optimization
- `image_prompt_generation` - Image generation prompt enhancement
- `autocomplete_generation` - Text autocompletion
- `moa_response_generation` - Mixture of Agents responses

### User Interactions
- `chat_message` - Regular user conversations
- `function_calling` - Tool/function usage

## 📈 Analytics Capabilities

### Performance Metrics
- Average inference times by model and interaction type
- Tool usage patterns and success rates
- Response quality metrics across different models

### Usage Patterns
- Most frequently used system features
- Tool adoption and usage trends
- User behavior analysis and conversation patterns

### Training Data Quality
- Conversation length and complexity analysis
- Multimodal content distribution
- Tool interaction patterns for training

## 🔧 Data Structures

### Filter Reporting Format
```json
{
  "requestedAt": 1673123456789,
  "receivedAt": 1673123457890,
  "reqPayload": {
    "model": "gpt-4",
    "messages": [...],
    "tools": [...]
  },
  "respPayload": {
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "Response text",
        "tool_calls": [...]
      }
    }]
  },
  "statusCode": 200,
  "metadata": {
    "interaction_type": "function_calling",
    "inference_time_ms": 1250,
    "tools_used": ["search", "calculator"],
    "has_tool_calls": true,
    "user_id": "user123",
    "chat_id": "chat456"
  }
}
```

### Dataset Entry Format
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/..."}}
      ]
    },
    {
      "role": "assistant",
      "content": "I can see...",
      "tool_calls": [...]
    }
  ],
  "split": "TRAIN",
  "metadata": {
    "source": "openwebui",
    "user_id": "user123",
    "model": "gpt-4-vision",
    "timestamp": "1673123456789"
  }
}
```

## 🎭 Multimodal Support

Both components fully support multimodal interactions:

### Supported Content Types
- **Text**: Regular chat messages and responses
- **Images**: Image URLs, base64 data, file uploads
- **Audio**: Audio files and transcriptions
- **Tool Calls**: Function calls and responses
- **Mixed Content**: Combinations of text, images, and tool usage

### Preservation Features
- **Raw Content**: Maintains original format and structure
- **Metadata**: Preserves content type, size, and format information
- **Tool Context**: Keeps function definitions and call results
- **Conversation Flow**: Maintains message order and relationships

## 🔍 Troubleshooting

### Enable Debug Logging
Set `DEBUG_LOGGING` to `true` in both components to see detailed logs:

**Filter Logs:**
```
[OpenPipe Filter v3.0 INFO] Captured request 1673123456_12345 - Type: chat_message
[OpenPipe Filter v3.0 INFO] Preparing to send report to OpenPipe
[OpenPipe Filter v3.0 INFO] ✅ Successfully reported to OpenPipe
```

**Dataset Action Logs:**
```
[OpenPipe Dataset Action INFO] Detected multimodal content: 2 images, 1 tool calls
[OpenPipe Dataset Action INFO] Created new dataset: OpenWebUI Conversations
[OpenPipe Dataset Action INFO] Successfully added 1 entries to dataset
```

### Common Issues

#### Filter Issues
1. **No reports appearing**: Check toggle is enabled and API key is set
2. **API errors**: Verify API key permissions and network connectivity
3. **Missing tool data**: Ensure `INCLUDE_TOOL_METRICS` is enabled
4. **UI feedback not showing**: Check OpenWebUI version compatibility

#### Dataset Action Issues
1. **Dataset creation fails**: Verify API key has dataset creation permissions
2. **Multimodal content missing**: Check `INCLUDE_SYSTEM_MESSAGES` setting
3. **Action not appearing**: Ensure action is properly installed and enabled
4. **Empty conversations**: Check that conversation has valid messages

### Performance Considerations

- **Asynchronous Operations**: Both components use async processing
- **Memory Management**: Request data is cleaned up after processing
- **Error Handling**: Failures don't affect chat functionality
- **Rate Limiting**: Built-in timeout and retry mechanisms

## 🔒 Security & Privacy

### Data Protection
- **API Key Security**: Store OpenPipe API keys securely
- **HTTPS Transmission**: All data sent over encrypted connections
- **No Content Modification**: Components only observe, never modify conversations
- **User Privacy**: User IDs and metadata are optional and configurable

### Compliance Considerations
- **Data Retention**: Configure according to your organization's policies
- **User Consent**: Inform users about data collection and reporting
- **Regional Compliance**: Ensure compliance with local data protection laws
- **Audit Trail**: Both components provide detailed logging for compliance

## 🚀 Advanced Usage

### Custom Metadata Examples

**Development Environment:**
```json
{
  "CUSTOM_METADATA": "{\"environment\": \"development\", \"version\": \"3.0.0\", \"team\": \"ai-research\"}"
}
```

**Production Deployment:**
```json
{
  "CUSTOM_METADATA": "{\"deployment\": \"production\", \"region\": \"us-west-2\", \"instance_id\": \"web-01\"}"
}
```

### Selective Reporting Strategies

**Research Focus:**
```json
{
  "REPORT_CHAT_MESSAGES": true,
  "REPORT_SYSTEM_TASKS": false,
  "REPORT_TOOL_INTERACTIONS": true
}
```

**System Optimization:**
```json
{
  "REPORT_CHAT_MESSAGES": false,
  "REPORT_SYSTEM_TASKS": true,
  "REPORT_TOOL_INTERACTIONS": false
}
```

## 📚 Integration Examples

### Training Pipeline Integration
1. **Collect conversations** using the dataset action
2. **Monitor performance** with the reporting filter
3. **Analyze patterns** in OpenPipe dashboard
4. **Fine-tune models** using collected datasets
5. **Deploy improved models** back to OpenWebUI

### Analytics Workflow
1. **Automatic reporting** captures all interactions
2. **Performance metrics** identify optimization opportunities
3. **Usage patterns** inform feature development
4. **Quality metrics** guide model selection

## 🤝 Contributing

To extend or modify the components:

### Filter Extensions
- **Add interaction types**: Update `_classify_interaction()` method
- **Custom analytics**: Extend metadata building logic
- **New providers**: Update model detection logic

### Dataset Action Extensions
- **Content processors**: Add support for new content types
- **Metadata extractors**: Enhance conversation analysis
- **Export formats**: Add support for different training formats

## 📄 License

This integration suite is provided as-is for use with OpenWebUI and OpenPipe. Ensure compliance with your organization's data policies and OpenPipe's terms of service before deployment.

## 🆘 Support

For issues and questions:
1. **Check debug logs** with `DEBUG_LOGGING` enabled
2. **Verify API keys** and permissions in OpenPipe dashboard
3. **Test connectivity** to OpenPipe API endpoints
4. **Review configuration** settings for both components

---

**Authors**: Cline (AI Assistant) & Gwyn (Human Collaborator)  
**Version**: Filter v3.0.0, Dataset Action v1.1.0  
**Created**: January 2025  
**Updated**: January 2025
