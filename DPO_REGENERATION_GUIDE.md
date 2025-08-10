# OpenPipe DPO Regeneration Detection Guide

## 🎯 Overview

The OpenPipe DPO Action now includes **automatic regeneration detection** to create high-quality preference pairs for Direct Preference Optimization (DPO) training. When a user regenerates a response in OpenWebUI, the system automatically treats the final accepted version as the "preferred" response and the previous rejected version as the "rejected" response.

## 🔄 How Regeneration Detection Works

### Detection Logic

1. **Conversation State Tracking**: The action maintains a history of conversation states for each chat
2. **Response Comparison**: When a new response is generated, it compares with the previous state
3. **Regeneration Identification**: If the user prompt is the same but the assistant response changed, it's detected as regeneration
4. **Preference Pair Creation**: 
   - **Preferred Response**: The final (current) response the user accepted
   - **Rejected Response**: The previous response the user regenerated away from

### Example Scenario

```
User: "Explain quantum computing"
Assistant: [Response A - Technical and dry]
User: [Clicks regenerate button]
Assistant: [Response B - More engaging and clear]

Result: 
✅ Preferred: Response B (final accepted version)
❌ Rejected: Response A (user regenerated away from this)
```

## ⚙️ Configuration Options

### Preference Modes

The DPO Action supports multiple preference modes via the `PREFERENCE_MODE` valve:

#### 1. `regeneration_detection` (Default)
- **Behavior**: Automatically detects when responses are regenerated
- **Preferred**: Final accepted response after regeneration
- **Rejected**: Previous response that was regenerated
- **Use Case**: Captures real user preferences through regeneration behavior

#### 2. `interactive`
- **Behavior**: Treats current response as preferred, creates generic rejected version
- **Use Case**: Manual preference marking

#### 3. `auto_preferred`
- **Behavior**: Marks current response as preferred automatically
- **Use Case**: Positive reinforcement training

#### 4. `auto_rejected`
- **Behavior**: Marks current response as rejected automatically
- **Use Case**: Negative example collection

### Key Configuration Valves

```python
PREFERENCE_MODE: str = "regeneration_detection"  # Enable regeneration detection
DETECT_REGENERATIONS: bool = True                # Enable/disable detection
REJECTED_REASONING: str = "User regenerated this response, indicating preference for the final version"
DEBUG_LOGGING: bool = False                      # Enable detailed logging
```

## 📊 Dataset Entry Format

When regeneration is detected, the DPO entry includes:

### Preferred Response (Final Version)
```json
{
  "role": "assistant",
  "content": "Final response content that user accepted"
}
```

### Rejected Response (Previous Version)
```json
{
  "role": "assistant", 
  "content": "Previous response content that user regenerated",
  "reasoning_content": "User regenerated this response, preferring the final version"
}
```

### Metadata
```json
{
  "source": "openwebui_dpo",
  "preference_mode": "regeneration_detection",
  "regeneration_detected": "true",
  "chat_id": "conversation-id",
  "model": "model-name",
  "timestamp": "unix-timestamp"
}
```

## 🚀 Usage Instructions

### 1. Enable Regeneration Detection
1. Install the OpenPipe DPO Action in OpenWebUI
2. Configure your OpenPipe API key in the action's Valves
3. Set `PREFERENCE_MODE` to `"regeneration_detection"`
4. Ensure `DETECT_REGENERATIONS` is `true`

### 2. Generate Training Data
1. Have normal conversations with your AI models
2. When you get a response you don't like, click the regenerate button
3. Accept the better response
4. Click the DPO Action button to capture the preference pair

### 3. Monitor Detection
Enable `DEBUG_LOGGING` to see regeneration detection in action:

```
[timestamp] [OpenPipe DPO Action INFO] 🔄 Regeneration detected in chat abc123
[timestamp] [OpenPipe DPO Action INFO] Previous response: This is a technical explanation...
[timestamp] [OpenPipe DPO Action INFO] Current response: Let me explain this in simpler terms...
[timestamp] [OpenPipe DPO Action INFO] 🔄 Using regeneration-based DPO pair for chat abc123
```

## 🎯 Benefits of Regeneration Detection

### 1. **Authentic User Preferences**
- Captures real user behavior rather than artificial preferences
- Users naturally regenerate when they're unsatisfied with responses
- No additional user effort required beyond normal usage

### 2. **High-Quality Training Data**
- Preferred responses are ones users actually accepted
- Rejected responses are ones users actively chose to replace
- Context is identical, isolating response quality differences

### 3. **Scalable Data Collection**
- Automatically generates training data during normal usage
- No need for manual preference annotation
- Scales with user activity

### 4. **Improved Model Training**
- DPO training with authentic preference pairs
- Better alignment with user expectations
- More effective than synthetic preference data

## 🔧 Technical Implementation

### State Tracking
The action maintains conversation history using:
```python
self._conversation_history = {
    "chat_id": {
        "last_user_message": {"content": "...", "index": 5},
        "assistant_responses": [{"content": "...", "message": {...}}]
    }
}
```

### Detection Algorithm
1. Extract current conversation state
2. Compare with previous state for same chat
3. Check if user message is identical but assistant response changed
4. If changed, mark as regeneration and create preference pair

### Memory Management
- Conversation history is stored per chat ID
- Previous states are overwritten with current states
- Memory usage scales with number of active conversations

## 📈 Best Practices

### 1. **Enable Debug Logging Initially**
- Set `DEBUG_LOGGING: true` when first testing
- Monitor logs to verify detection is working
- Disable in production for performance

### 2. **Use Descriptive Dataset Names**
- Set meaningful `DPO_DATASET_NAME` values
- Include model name or use case in dataset name
- Example: "GPT-4 Conversation DPO Training"

### 3. **Regular Data Review**
- Periodically review generated preference pairs
- Ensure quality of detected regenerations
- Adjust detection sensitivity if needed

### 4. **Combine with Other Modes**
- Use regeneration detection as primary method
- Supplement with manual `interactive` mode for edge cases
- Create separate datasets for different preference types

## 🐛 Troubleshooting

### No Regenerations Detected
- Verify `DETECT_REGENERATIONS` is `true`
- Check that `PREFERENCE_MODE` is `"regeneration_detection"`
- Enable debug logging to see detection attempts
- Ensure chat IDs are consistent across requests

### False Positives
- Review debug logs for detection logic
- Check if conversation state tracking is accurate
- Verify user message comparison is working correctly

### Missing Preference Pairs
- Confirm DPO Action button is clicked after regeneration
- Check OpenPipe API connectivity and authentication
- Review dataset creation and entry submission logs

## 🔮 Future Enhancements

### Planned Features
- **Multi-turn Regeneration**: Detect regeneration chains across multiple turns
- **Quality Scoring**: Automatically score preference strength based on regeneration patterns
- **Batch Processing**: Process multiple regenerations in a single dataset submission
- **Advanced Filtering**: Filter out low-quality regenerations based on content similarity

### Integration Opportunities
- **Filter Integration**: Automatic DPO submission via the reporting filter
- **Analytics Dashboard**: Visualize regeneration patterns and preference trends
- **Model Comparison**: Compare regeneration rates across different models

---

## 📚 Related Documentation

- [OpenPipe DPO Training Guide](https://docs.openpipe.ai/features/dpo)
- [OpenWebUI Actions Documentation](https://docs.openwebui.com/features/actions)
- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)

---

**Authors**: Cline (AI Assistant) & Gwyn (Human Collaborator)  
**Version**: 1.0.0  
**Last Updated**: January 2025
