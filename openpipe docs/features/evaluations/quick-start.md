# Evaluations Quick Start

> Create your first head to head evaluation.

In this quick start guide, we'll walk you through creating your first head-to-head evaluation. Head to head evaluations allow you to compare two or more models
using an LLM judge that compares outputs against one another based on custom instructions.

<Note>
  <b>Before you begin:</b> Before writing your first eval, make sure you've [created a
  dataset](/features/datasets/quick-start) with one or more test entries. Also, make sure to add
  your OpenAI or Anthropic API key in your project settings page to allow the judge LLM to run.
</Note>

### Writing an Evaluation

<Steps>
  <Step title="Choose a dataset to evaluate models on">
    To create an eval, navigate to the dataset with the test entries you'd like to evaluate your models based on.
    Find the **Evaluate** tab and click the **+** button to the right of the **Evals** dropdown list.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/eval-button.png)</Frame>

    A configuration modal will appear.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/create-h2h-eval.png)</Frame>
  </Step>

  <Step title="Edit judge model instructions">
    Customize the judge LLM instructions. The outputs of each model will be compared against one another
    pairwise and a score of WIN, LOSS, or TIE will be assigned to each model's based on the judge's instructions.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/edit-judge-instructions.png)</Frame>
  </Step>

  <Step title="Select judge model">
    Choose a judge model from the dropdown list. If you'd like to use a judge model that isn't supported by default,
    add it as an [external model](/features/chat-completions/external-models) in your project settings page.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/select-judge-model.png)</Frame>
  </Step>

  <Step title="Choose models to evaluate">
    Choose the models you'd like to evaluate against one another.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/choose-evaluated-models.png)</Frame>
  </Step>

  <Step title="Run the evaluation">
    Click **Create** to start running the eval.

    Once the eval is complete, you can see model performance in the evaluation's **Results** tab.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/evaluations/quick-start-results.png)</Frame>
  </Step>
</Steps>

To learn more about customizing the judge LLM instructions and viewing evaluation judgements in greater detail,
see the [Head-to-Head Evaluations](/features/evaluations/head-to-head) page.
