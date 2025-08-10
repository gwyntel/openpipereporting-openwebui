# Pruning Rules

> Decrease input token counts by pruning out chunks of static text.

Some prompts have large chunks of unchanging text, like system messages which don't change from one request to the next. By removing this static text and fine-tuning a model on the compacted data, we can reduce the size of incoming requests and save you money on inference.

Add pruning rules to your dataset in the Settings tab, as shown below and in our [demo dataset](https://app.openpipe.ai/p/BRZFEx50Pf/datasets/0aa75f72-3fe5-4294-a94e-94c9236befa6/settings).

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/pruning-rules/dataset-pruning-rule.png)</Frame>

To see the effect your pruning rules had on an individual training entry's input messages, open the Dataset Entry drawer:

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/pruning-rules/drawer-rule.png)</Frame>

By default, fine-tuned models inherit pruning rules applied to the dataset on which they were trained (see [demo model](https://app.openpipe.ai/p/BRZFEx50Pf/fine-tunes/5a2af605-03d3-412c-a7d3-611bdf6e1dcf/general)). These rules will automatically prune matching text from any incoming requests sent to that model. New pruning rules will not be associated with previously trained models, so you don't need to worry about backwards compatibility when adding new rules to your dataset. Before training a new model, you can choose to disable any inherited pruning rules.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/pruning-rules/model-rules.png)</Frame>

## Warning: can affect quality!

We’ve found that while pruning rules always decrease latency and costs, they can also negatively affect response quality, especially with smaller datasets. We recommend enabling pruning rules on datasets with 10K+ training examples, as smaller datasets may not provide enough guidance for the model to fully learn the task.
