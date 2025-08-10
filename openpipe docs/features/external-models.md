# External Models

<Info>
  Before defining a custom external model provider, check your project settings to see if the
  provider you're looking for is already supported.
</Info>

To use a custom external model from a cloud provider that OpenPipe doesn't support, you can add an external model provider to your project. External models can be used for the following purposes:

* [Proxying chat completions](/features/chat-completions/external-models)
* [Filtering and relabeling your data](/features/evaluations/head-to-head)
* [Evaluating outputs through criteria](/features/criteria/quick-start)

The instructions below demonstrate how to add a DeepSeek (OpenAI Compatible) and Azure provider to your project.

### Creating an external model provider

Find the **External Model Providers** section of your project settings, and click the **Add Provider** button.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/external-models/add-provider-button.png)</Frame>

Give your custom provider a slug, API key, and add a custom base url if necessary. The provider slug should be unique,
and will be used when we proxy requests to models associated with this provider.

<Tabs>
  <Tab title="DeepSeek (OpenAI Compatible)">
    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/external-models/add-provider-modal-deepseek.png)</Frame>
  </Tab>

  <Tab title="Azure Endpoint">
    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/external-models/add-provider-modal-azure.png)</Frame>
  </Tab>
</Tabs>

### Adding a model to the external provider

To add a model to the provider you're creating, click the <b>Add Model</b> button.

<Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/external-models/add-model-button.png)</Frame>

Provide a slug that matches the model you'd like to call on your external provider. To call <b>gpt-4o-2024-08-06</b> on Azure for instance, the slug should be `gpt-4o-2024-08-06`.

<Tabs>
  <Tab title="DeepSeek (OpenAI Compatible)">
    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/external-models/add-model-row-deepseek.png)</Frame>
  </Tab>

  <Tab title="Azure Endpoint">
    <Frame>![](https://mintlify.s3.us-west-1.amazonaws.com/openpipe/images/external-models/add-model-row-azure.png)</Frame>
  </Tab>
</Tabs>

Setting input cost and output cost is optional, but can be helpful for showing relative costs in the [evals](/features/evaluations) page.

We currently support custom external
models for providers with openai and azure-compatible endpoints. If you'd like support for an external provider with a different API format, send a request to [hello@openpipe.ai](mailto:hello@openpipe.ai).
