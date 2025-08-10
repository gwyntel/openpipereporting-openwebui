# Criterion Evaluations

>  Evaluate your LLM outputs using criteria. 

<Note>
  Criterion evaluations are useful for evaluating your LLM outputs against a set of criteria. If you
  haven't defined any criteria yet, check out the criteria [Quick
  Start](/features/criteria/quick-start) guide.
</Note>

Criterion evaluations are a reliable way to judge the quality of your LLM outputs according to the criteria you've defined. For each model being evaluated, the output of that model is compared against the criteria you've defined for every entry in the evaluation dataset.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/criterion-eval-settings.png)</Frame>

<Info>
  A criterion evaluation is only as reliable as the criterion you've defined. To improve your
  criterion, check out the [alignment docs](/features/criteria/alignment-set).
</Info>

Each output in the evaluation dataset is compared against the criterion you've defined. The output is then scored as either `PASS` or `FAIL` based on the criterion.

<br />

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/criterion-eval-results-table.png)</Frame>

<br />

To see why one model might be outperforming another, you can navigate back to the [evaluation table](https://app.openpipe.ai/p/BRZFEx50Pf/datasets/3e7e82c1-b066-476c-9f17-17fd85a2169b/evaluate) and click on a result pill to see the evaluation judge's reasoning.

<br />

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/criterion-eval-explanation.png)</Frame>

<br />

While criterion evaluations are powerful and flexible, they're much more expensive to run than pure code. If your models' outputs can be easily evaluated by code alone, consider using [code evaluations](/features/evaluations/code) instead.
