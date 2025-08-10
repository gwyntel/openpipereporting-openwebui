# Head-to-Head Evaluations

>  Evaluate your LLM outputs against one another using head-to-head evaluations. 

<Note>
  Head-to-head evaluations are useful for evaluating your LLM outputs against one another to
  determine which models are generally better at a given task. However, they do not provide precise
  metrics on how often a given model makes a certain error, only how often it outperforms another
  model. For more precise metrics, please consider [criteria](/features/evaluations/criterion) or
  [code](/features/evaluations/code) evaluations.
</Note>

Head to head evaluations are a fast way to get a sense of how well your models perform against one another. For each model being evaluated, the output of that model is compared against the output of every other model for every entry in the evaluation dataset.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/h2h-settings.png)</Frame>

<Info>
  The number of comparisons performed in a head to head eval scales linearly with the number of
  entries and quadratically with the number of models. If you're evaluating 2 models on 100 entries,
  there will be 100 * 1 = 100 comparisons. If you're evaluating 3 models on 100 entries, there will
  be 100 * 2 + 100 * 1 = 300 comparisons.
</Info>

As outputs are compared against one another, each model is assigned a "win rate" score. For example, if you're evaluating 2 models on 100 entries and model A outperforms model B 55 times, model A will have a win rate of 55% and model B will have a win rate of 45%. In cases where both models produce the same output or the judge is unable to determine a winner, the score will be a tie (equivalent to 50% win rate).

<br />

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/h2h-results-table.png)</Frame>

<br />

In addition to the results table, you can also view results in a matrix format. This is useful for visualizing how specific models perform against one another.

<br />

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/h2h-matrix.png)</Frame>

<br />

To see why one model might be outperforming another, you can navigate back to the [evaluation table](https://app.openpipe.ai/p/BRZFEx50Pf/datasets/3e7e82c1-b066-476c-9f17-17fd85a2169b/evaluate) and click on a result pill to see the evaluation judge's reasoning.

<br />

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/h2h-judge-explanation.png)</Frame>

<br />

While head-to-head evaluations are convenient, they can quickly become expensive to run, and provide limited insight into how well a model performs. For more precise metrics, consider [criterion](/features/evaluations/criterion) or [code](/features/evaluations/code) evaluations.
