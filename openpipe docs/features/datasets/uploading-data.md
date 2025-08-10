# Uploading Data

>  Upload external data to kickstart your fine-tuning process. Use the OpenAI chat fine-tuning format.

Upload a JSONL file populated with a list of training examples.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/uploading-data.png)</Frame>

Each line of the file should be compatible with the OpenAI [chat format](https://platform.openai.com/docs/api-reference/chat/object), with additional optional fields.

### OpenAI Fields

* **`messages`: Required** - Formatted as a list of OpenAI [chat completion messages](https://platform.openai.com/docs/guides/gpt/chat-completions-api). The list should end with an assistant message.
* **`tools`: Optional** - An array of tools (functions) available for the model to call. For more information read OpenAI's [function calling docs](https://platform.openai.com/docs/guides/function-calling).
* **`tool_choice`: Optional** - You can set this to indicate that the model should be required to call the given tool. For more information read OpenAI's [function calling docs](https://platform.openai.com/docs/guides/function-calling).

#### Deprecated

* **`functions`: Deprecated | Optional** - An array of functions available for the model to call.
* **`function_call`: Deprecated | Optional** - You can set this to indicate that the model should be required to call the given function.

You can include other parameters from the OpenAI chat completion input format (eg. temperature), but they will be ignored since they aren't relevant for training.

### Additional Fields

* **`split`: Optional** - One of "TRAIN" or "TEST". If you don't set this field we'll automatically divide your inputs into train and test splits with a target ratio of 90:10.
* **`rejected_message`: Optional** - Add a rejected output for entries on which you want to perform direct preference optimization (DPO). You can find more information about that here: [Direct Preference Optimization](/features/dpo/overview)
* **`metadata`: Optional** - A string=>string dictionary of any additional information you want to associate with an entry. This can be useful for tracking information like prompt IDs.

### Example

```jsonl
...
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Tasmania?"},{"role":"assistant","content":"Hobart"}], "rejected_message":{"role": "assistant", "content": "Paris"}, "split": "TRAIN", "metadata": {"prompt_id": "capital_cities", "any_key": "any_value"}}
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Sweden?"},{"role":"assistant","content":"Stockholm"}], "rejected_message":{"role": "assistant", "content": "London"}, "split": "TEST", "metadata": {"prompt_id": "capital_cities", "any_key": "any_value"}}
...
```
