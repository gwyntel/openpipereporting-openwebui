# Evaluations

>  Evaluate the quality of your LLMs against one another or independently.

After training a model, you'll want to know how well it performs. Datasets include a built-in evaluation framework that makes it easy to compare newly trained models against previous models and generic base models as well.

By default, 10% of the dataset entries you provide will be withheld from training. These entries form your test set. For each entry in the test set, your new model will produce an output that will be shown in the [evaluation table](https://app.openpipe.ai/p/BRZFEx50Pf/datasets/3e7e82c1-b066-476c-9f17-17fd85a2169b/evaluate).

<br />

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/evals-table.png)</Frame>

<br />

Viewing outputs side by side is useful, but it doesn't tell you which model is doing better in general. For that, we need to define evaluations. Evaluations
allow you to compare model outputs across a variety of inputs to determine which model is doing a better
job. While each type of evaluation has its own unique UI, they all show final results in a sorted table.

<br />

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/eval-results-table.png)</Frame>

<br />

Datasets support three types of evaluations:

* [Code evaluations](/features/evaluations/code)
* [Criterion evaluations](/features/evaluations/criterion)
* [Head-to-head evaluations](/features/evaluations/head-to-head)

As a rough guide, use code evaluations for deterministic tasks like classification or information extraction. Use criterion evaluations for tasks with freeform outputs like chatbots or summarization. Use head-to-head evaluations for comparing two or more models against each other if you're looking for a quick and dirty way to compare model outputs.
