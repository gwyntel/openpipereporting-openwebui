# Exporting Data

>  Export your past requests as a JSONL file in their raw form.

## Dataset export

After you've collected, filtered, and transformed your dataset entries for fine-tuning, you can export them as a JSONL file.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/datasets/exporting-dataset-entries.png)</Frame>

### Fields

* **`messages`:** The complete chat history.
* **`tools`:** The tools provided to the model.
* **`tool_choice`:** The tool required for the model to use.
* **`split`:** The train/test split to which the entry belongs.

### Example

```jsonl
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Tasmania?"},{"role":"assistant","content":null,"tool_calls":[{"id":"","type":"function","function":{"name":"identify_capital","arguments":"{\"capital\":\"Hobart\"}"}}}]}],"tools":[{"type":"function","function":{"name":"identify_capital","parameters":{"type":"object","properties":{"capital":{"type":"string"}}}}}]}
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Sweden?"},{"role":"assistant","content":null,"tool_calls":[{"id":"","type":"function","function":{"name":"identify_capital","arguments":"{\"capital\":\"Stockholm\"}"}}}]}],"tools":[{"type":"function","function":{"name":"identify_capital","parameters":{"type":"object","properties":{"capital":{"type":"string"}}}}}]}
```
