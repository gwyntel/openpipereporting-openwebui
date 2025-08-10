# Reward Models (Beta)

>  Train reward models to judge the quality of LLM responses based on preference data.

In addition to training models to generate completions, OpenPipe allows you to train reward models to judge the quality of LLM responses based on preference data.

### Preparing preference data

Currently, OpenPipe supports training reward models from preference data (training reward models based on rewards directly is in the works). Preference data is a dataset of pairs of responses in which one response is better than another.

One way to gather such a dataset is to present users with two LLM completions and ask them to select the better one. Another way is to allow users to regenerate a response until they are satisfied with the output, then associate the original and final generated responses as a pair, with the final response being the preferred one. Whether you gather your preference data from your users or through a different process, it needs to match the JSONL preference data format shown below:

```jsonl
...
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Tasmania?"},{"role":"assistant","content":"Hobart"}], "rejected_message":{"role": "assistant", "content": "Paris"}, "split": "TRAIN"}
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Sweden?"},{"role":"assistant","content":"Stockholm"}], "rejected_message":{"role": "assistant", "content": "London"}, "split": "TRAIN"}
...
```

Note that the `rejected_message` field should contain the rejected response, and the final `assistant` message should be the preferred response. Conveniently, this is the same data format used for [DPO](/features/dpo/overview), which allows you to train both reward and completion models from the same dataset.

### Training a reward model

Once you've added your preference data to a dataset, you can train a reward model by opening the "Reward Model" modal on the dataset page.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning/reward-model-button.png)</Frame>

Select a base model and optionally configure hyperparameters, then click "Start Training".

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning/reward-model-modal.png)</Frame>

Your reward model will be trained in the background. You can check the status of your model by navigating to the Fine Tunes page and selecting your model.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning/reward-model-status.png)</Frame>

### Using a reward model

Like criteria, reward models can be used for best of N sampling and offline evaluation. Reward model slugs can be specified independently or in conjunction with criteria, as shown below:

```bash
curl --request POST \
--url https://app.openpipe.ai/api/v1/chat/completions \
--header "Authorization: Bearer $OPENPIPE_API_KEY" \
--header 'Content-Type: application/json' \
--header 'op-criteria: ["<reward-model-slug>"]' \
--data '{
"model": "openai:gpt-4o-mini",
"messages": [
    {
        "role": "user",
        "content": "Count to 10"
    }
],
"store": true,
"n": 5,
"metadata": {
    "prompt_id": "counting",
    "any_key": "any_value"
}
}'
```

For more details on how to use reward models and criteria, see the [criteria API docs](/features/criteria/api).

<Info>Want to learn more about reward models? Send questions to [support@openpipe.ai](mailto:support@openpipe.ai).&lt;/Info>
