# Fine-Tuning Quick Start

> Train your first fine-tuned model with OpenPipe.

Fine-tuning open and closed models with custom hyperparameters only takes a few clicks.

<Note>
  <b>Before you begin:</b> Before training your first model, make sure you've [created a
  dataset](/features/datasets/quick-start) and imported at least 10 training entries.
</Note>

### Training a Model

<Steps title="Training a Model">
  <Step title="Navigate to Dataset">
    To train a model, navigate to the dataset you'd like to train your model on. Click the **Fine Tune** button in the top right corner of the **General** tab.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning/fine-tune-modal.png)</Frame>
  </Step>

  <Step title="Name your Model">
    Choose a descriptive name for your new model. This name will be used as the `model` parameter when querying it in code.
    You can always rename your model later.
  </Step>

  <Step title="Select Base Model">
    Select the base model you'd like to fine-tune on. We recommend starting with Llama 3.1 8B if you aren't sure which to choose.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning/select-base-model.png)</Frame>
  </Step>

  <Step title="Adjust Hyperparameters (optional)">
    Under **Advanced Options**, you can optionally adjust the hyperparameters to fine-tune your model.
    You can leave these at their default values if you aren't sure which to choose.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning/adjust-hyperparameters.png)</Frame>
  </Step>

  <Step title="Start Training">
    Click **Start Training** to begin the training process.
    The training job may take a few minutes or a few hours to complete, depending on the amount of training data, the base model, and the hyperparameters you choose.

    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning/trained-model.png)</Frame>
  </Step>
</Steps>

To learn more about fine-tuning through the webapp, check out the [Fine-Tuning via Webapp](/features/fine-tuning/overview) page.
To learn about fine-tuning via API, see our [Fine Tuning via API](/api-reference/fine-tuning) page.
