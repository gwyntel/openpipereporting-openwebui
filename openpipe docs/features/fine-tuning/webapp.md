# Fine Tuning via Webapp

>  Fine tune your models on filtered logs or uploaded datasets. Filter by prompt id and exclude requests with an undesirable output.

OpenPipe allows you to train, evaluate, and deploy your models all in the same place. We recommend training your models
through the webapp, which provides more flexibility and a smoother experience than the API. To fine-tune a new model, follow these steps:

1. Create a new dataset or navigate to an existing one.
2. Click "Fine Tune" in the top right.
3. Select a base model.
4. (Optional) Set custom hyperparameters and configure [pruning rules](/features/pruning-rules).
5. Click "Start Training" to kick off the job.

Once started, your model's training job will take at least a few minutes and potentially several hours, depending on the size of the
model and the amount of data. You can check your model's status by navigating to the Fine Tunes page and selecting your model.

For an example of how an OpenPipe model looks once it's trained, see our public [PII Redaction](https://app.openpipe.ai/p/BRZFEx50Pf/fine-unes/6076ad69-cce5-4892-ae54-e0549bbe107f/general) model. Feel free to hit it with some sample queries!

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/features/fine-tuning.png)</Frame>
