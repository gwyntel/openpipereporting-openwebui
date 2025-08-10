# Criteria

> Align LLM judgements with human ratings to evaluate and improve your models.

<Info>
  For questions about criteria or to unlock beta features for your organization, reach out to
  [support@openpipe.ai](mailto:support@openpipe.ai).
</Info>

Criteria are a simple way to reliably detect and correct mistakes in LLM output. Criteria can currently be used for the following purposes:

* Defining LLM evaluations
* Improving dataset quality
* Runtime evaluation when generating [best of N](/features/criteria/api#runtime-evaluation) samples
* [Offline testing](/features/criteria/api#offline-testing) of previously generated outputs

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/criteria/overview.png)</Frame>

## What is a Criterion?

A criterion is a combination of an LLM model and prompt that can be used to identify a specific issue with a model's output. Criterion judgements are generated
by passing the input and output of a single row along with the criterion prompt to an LLM model, which then returns a binary `PASS`/`FAIL` judgement.

To learn how to create your first criterion, read the [Quick Start](/features/criteria/quick-start).
