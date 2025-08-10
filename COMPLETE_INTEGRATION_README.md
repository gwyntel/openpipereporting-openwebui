# OpenPipe Integration Suite for OpenWebUI

A comprehensive collection of OpenWebUI plugins that provide seamless integration with OpenPipe for analytics, fine-tuning, and DPO (Direct Preference Optimization) training.

## 🚀 What's Included

### 1. **Universal OpenPipe Filter v3.0** (`openpipe_filter_v3.py`)
- **UI Toggle Switch**: Enable/disable reporting per conversation
- **Custom OpenPipe Icon**: Visual branding in the interface
- **Real-time Status Feedback**: Live updates on reporting status
- **Universal Model Support**: Works with ALL models (OpenAI, Ollama, Claude, etc.)
- **Enhanced Raw Capture**: Preserves tool calls and response structure
- **Comprehensive Analytics**: Interaction classification, timing, tool usage

### 2. **Dataset Action** (`openpipe_dataset_action.py`)
- **Add to Training Datasets**: One-click addition of conversations to OpenPipe datasets
- **Auto Dataset Creation**: Automatically creates datasets if they don't exist
- **Metadata Enrichment**: Includes user context, model info, and conversation metadata
- **Flexible Configuration**: Configurable dataset names and splits (TRAIN/TEST)

### 3. **DPO Training Action** (`openpipe_dpo_action.py`)
- **Direct Preference Optimization**: Mark conversations for DPO training
- **Preference Modes**: Interactive, auto-preferred, or auto-rejected marking
- **Rejected Response Reasoning**: Includes reasoning for why responses were rejected
- **Separate DPO Dataset**: Maintains dedicated dataset for preference learning

## 🎯 Key Features

### Universal Coverage
- **All Models Supported**: Works with any model in OpenWebUI without configuration
- **No Core Modifications**: Uses only OpenWebUI's plugin system
- **Backward Compatible**: Works with existing OpenWebUI installations

### User Experience
- **Visual Controls**: Toggle switches and custom icons
- **Real-time Feedback**: Status updates during operations
- **Error Handling**: Graceful error handling with user feedback
- **Configurable Options**: Extensive configuration through Valves

### Data Quality
- **Raw Response Capture**: Preserves original response structure
- **Tool Call Preservation**: Maintains function calls and tool usage
- **Metadata Enrichment**: Comprehensive context and analytics
- **Flexible Filtering**: Selective reporting by interaction type

## 📦 Installation

### 1. Filter Installation
1. Copy `openpipe_filter_v3.py` to your OpenWebUI filters directory
2. Configure your OpenPipe API key in the filter settings
3. Enable the filter and configure reporting preferences

### 2. Action Installation
1. Copy `openpipe_dataset_action.py` and `openpipe_dpo_action.py` to your OpenWebUI actions directory
2. Configure OpenPipe API keys in each action's settings
3. Set dataset names and preferences

### 3. Configuration
```python
# Filter Configuration
OPENPIPE_API_KEY = "your-api-key-here"
OPENPIPE_BASE_URL = "https://api.openpipe.ai/api/v1"

# Dataset Action Configuration  
DEFAULT_DATASET_NAME = "OpenWebUI Conversations"
AUTO_CREATE_DATASET = True

# DPO Action Configuration
DPO_DATASET_NAME = "DPO Training Data"
PREFERENCE_MODE = "interactive"  # or "auto_preferred", "auto_rejected"
```

## 🔧 Usage

### Basic Reporting (Filter)
1. **Enable the Filter**: Toggle the OpenPipe filter on in your conversation
2. **Automatic Reporting**: All interactions are automatically reported to OpenPipe
3. **Real-time Feedback**: See status updates as conversations are reported
4. **Toggle Control**: Turn reporting on/off per conversation as needed

### Dataset Creation (Action)
1. **Click the Dataset Button**: Use the "Add to OpenPipe Dataset" button on any message
2. **Automatic Processing**: Conversation is formatted and added to your training dataset
3. **Dataset Management**: Datasets are created automatically or use existing ones
4. **Metadata Inclusion**: Rich metadata is added for better training data organization

### DPO Training (Action)
1. **Mark Preferences**: Use the "Mark for DPO Training" button on assistant responses
2. **Preference Modes**: 
   - **Interactive**: Mark current response with reasoning
   - **Auto Preferred**: Automatically mark as preferred response
   - **Auto Rejected**: Mark as rejected with improved alternative
3. **Training Data**: Creates proper DPO entries with preferred/rejected pairs

## 📊 Data Captured

### Filter Reporting
```json
{
  "requestedAt": 1704067200000,
  "receivedAt": 1704067201500,
  "reqPayload": {
    "model": "gpt-4",
    "messages": [...],
    "tools": [...]
  },
  "respPayload": {
    "choices": [{
      "message": {
        "role": "assistant",
        "content": "Response text",
        "tool_calls": [...]
      }
    }],
    "usage": {...}
  },
  "metadata": {
    "interaction_type": "chat_message",
    "user_id": "user_123",
    "inference_time_ms": 1500,
    "tools_used": ["get_weather"],
    "capture_method": "filter_level_v3"
  }
}
```

### Dataset Entries
```json
{
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I'll check the weather for you."}
  ],
  "split": "TRAIN",
  "metadata": {
    "source": "openwebui",
    "user_id": "user_123",
    "model": "gpt-4",
    "message_count": "2"
  }
}
```

### DPO Entries
```json
{
  "messages": [
    {"role": "user", "content": "Explain quantum computing"},
    {"role": "assistant", "content": "Quantum computing uses quantum mechanics..."}
  ],
  "rejected_message": {
    "role": "assistant", 
    "content": "Quantum computing is complicated...",
    "reasoning_content": "User marked this response as less preferred"
  },
  "split": "TRAIN",
  "metadata": {
    "source": "openwebui_dpo",
    "preference_mode": "interactive"
  }
}
```

## ⚙️ Configuration Options

### Filter Settings
- **Reporting Control**: Enable/disable different interaction types
- **Tool Analytics**: Include tool usage metrics
- **Timing Metrics**: Capture inference timing
- **Custom Metadata**: Add custom fields to reports
- **Debug Logging**: Enable detailed logging

### Dataset Action Settings
- **Dataset Names**: Configure default dataset names
- **Auto Creation**: Automatically create missing datasets
- **Split Configuration**: Set TRAIN/TEST splits
- **Message Filtering**: Include/exclude system messages

### DPO Action Settings
- **Preference Modes**: Configure how preferences are determined
- **Reasoning Text**: Customize rejection reasoning
- **Dataset Management**: Separate DPO dataset configuration

## 🔍 Monitoring & Analytics

### OpenPipe Dashboard
Access your data at https://app.openpipe.ai/

**Useful Filters:**
- `source:openwebui` - All OpenWebUI interactions
- `capture_method:filter_level_v3` - Filter v3.0 reports
- `interaction_type:chat_message` - User conversations
- `interaction_type:function_calling` - Tool usage
- `source:openwebui_dpo` - DPO training data

### Debug Information
Enable debug logging in any component to see detailed operation logs:
```python
DEBUG_LOGGING = True
```

## 🚨 Troubleshooting

### Common Issues

**Filter Not Reporting:**
- Check OpenPipe API key configuration
- Verify toggle is enabled
- Check debug logs for errors
- Ensure network connectivity to OpenPipe

**Actions Not Working:**
- Verify API key in action settings
- Check dataset permissions
- Review error messages in action responses
- Enable debug logging for details

**Data Quality Issues:**
- Review metadata configuration
- Check message filtering settings
- Verify tool call preservation
- Examine raw vs processed data differences

### Performance Considerations
- Reporting is asynchronous and doesn't block responses
- Failed reports are logged but don't affect functionality
- Large conversations may take longer to process
- Rate limiting is handled gracefully

## 🔄 Migration Guide

### From v2.0 Filter
The v3.0 filter is backward compatible. Key improvements:
- Added UI toggle and icon
- Enhanced status feedback
- Better error handling
- Improved raw data capture

### From Basic Reporting
The action buttons complement the filter:
- Filter: Automatic reporting of all interactions
- Actions: Manual, selective dataset management
- Combined: Complete OpenPipe integration

## 🎯 Best Practices

### For Training Data
1. **Use Dataset Action** for curated training conversations
2. **Enable System Messages** for complete context
3. **Organize by Dataset Names** for different use cases
4. **Monitor Data Quality** in OpenPipe dashboard

### For DPO Training
1. **Use Interactive Mode** for manual preference marking
2. **Provide Clear Reasoning** for rejected responses
3. **Balance Preferred/Rejected** examples
4. **Review DPO Dataset** regularly for quality

### For Analytics
1. **Enable All Interaction Types** for comprehensive data
2. **Use Custom Metadata** for specific tracking needs
3. **Monitor Tool Usage** for feature adoption
4. **Track Inference Times** for performance optimization

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Enable debug logging for detailed information
3. Review OpenPipe API documentation
4. Check OpenWebUI plugin documentation

## 📈 Future Enhancements

Potential improvements:
- **Batch Processing**: Bulk dataset operations
- **Advanced Filtering**: More granular reporting controls
- **Custom Preference UI**: Interactive preference selection
- **Model Comparison**: Side-by-side response evaluation
- **Export Tools**: Data export and migration utilities

---

**Authors**: Cline (AI Assistant) & Gwyn (Human Collaborator)  
**Version**: 3.0.0 Complete Integration Suite  
**Created**: January 2025
