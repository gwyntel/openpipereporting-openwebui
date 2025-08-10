# Direct Preference Optimization (DPO)

<Note>
  DPO is much harder to get right than supervised fine-tuning, and the results may not always be
  better. To get the most out of DPO, we recommend familiarizing yourself with your specific use
  case, your dataset, and the technique itself.
</Note>

Direct Preference Optimization (DPO), introduced in [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2106.13358), is an algorithm used to fine-tune LLMs based on preference feedback.

It focuses on aligning model outputs with specific human preferences or desired behaviors. Unlike traditional supervised fine-tuning, which relies solely on input-output pairs, DPO leverages preference data—information about which of two outputs is preferred in a given context.

DPO works by directly optimizing a model to produce preferred outputs over non-preferred ones, without the need for complex reward modeling or reinforcement learning techniques. It uses paired data samples, where each pair consists of a preferred and a non-preferred response to a given prompt. This method allows the model to learn nuanced distinctions that are difficult to capture with explicit labels alone. By directly optimizing for preferences, DPO enables the creation of models that produce more aligned, contextually appropriate, and user-satisfying responses.

## Gathering Preference Data

DPO is useful when you have a source of preference data that you can exploit. There are many possible sources of preference data, depending on your use case:

1. **Expert Feedback**: you may have a team of experts who can evaluate your model's outputs and edit them to make them better. You can use the original and edited outputs as rejected and preferred outputs respectively. DPO can be effective with just a few preference pairs.
2. **Criteria Feedback**: if you use [OpenPipe criteria](/features/criteria/overview) or another evaluation framework that assigns a score or pass/fail to an output based on how well it meets certain criteria, you can run several generations and use the highest and lowest scoring outputs as preferred and non-preferred outputs respectively.
3. **User Choice**: if you have a chatbot-style interface where users can select their preferred response from a list of generated outputs, you can use the selected and rejected outputs as preference data.
4. **User Regenerations**: if a user is able to regenerate an action multiple times and then eventually accepts one of the outputs, you can use the first output they rejected as a non-preferred output and the accepted output as a preferred output.
5. **User Edits**: if your model creates a draft output and the user is able to edit it and then save, you can use the original draft as a non-preferred output and the edited draft as a preferred output.

## Dataset Format

If uploading a dataset for DPO, ensure you include a `rejected_message` field for each entry. This field should contain an output that model should avoid generating.

```jsonl
...
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Tasmania?"},{"role":"assistant","content":"Hobart"}], "rejected_message":{"role": "assistant", "content": "Paris"}, "split": "TRAIN", "metadata": {"prompt_id": "capital_cities", "any_key": "any_value"}}
{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is the capital of Sweden?"},{"role":"assistant","content":"Stockholm"}], "rejected_message":{"role": "assistant", "content": "London"}, "split": "TEST", "metadata": {"prompt_id": "capital_cities", "any_key": "any_value"}}
...
```

To learn more about the dataset format, read the [Uploading Data](/features/datasets/uploading-data) guide.

## Example Use Cases

Initial tests with DPO on OpenPipe have shown promising results. DPO, when used with [user-defined criteria](https://docs.openpipe.ai/features/criteria/overview), allows you to fine-tune models that more consistently respect even very nuanced preferences.

<Frame>![SFT vs DPO](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/dpo/sft-vs-dpo-for-criteria-chart.png)</Frame>

The following are all real results on customer tasks:

* **Word Limit**: for a summarization task with an explicit word limit given in the prompt, DPO was able to cut the number of responses exceeding the limit from 31% to 7%, a **77%** decrease.
* **Highlight Format**: for a content formatting task, DPO was able to drop the percentage of times the wrong word or phrase was highlighted from 17.3% to 1.7%, a **90%** decrease.
* **Hallucination**: for an information extraction task, DPO was able to drop the fraction of outputs with hallucinated information from 12.7% to 3.0%, a **76%** decrease.
* **Result Relevance**: for a classification task determining whether a result was relevant to a query, DPO was able to drop the mis-classification rate from 4.7% to 1.3%, a **72%** decrease.

We're excited to see how you'll leverage DPO to create even more powerful and tailored models for your specific needs!
