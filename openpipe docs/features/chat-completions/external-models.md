# Proxying to External Models

<Info>
  Adding custom external models is not required to proxy requests to Anthropic, Gemini, or OpenAI
  models. See our docs on proxying to [Anthropic](/features/chat-completions/anthropic),
  [Gemini](/features/chat-completions/gemini), or
  [OpenAI](/features/request-logs/logging-requests#proxy) for more information.
</Info>

To proxy requests to models from unsupported providers, you'll need to complete the following steps:

1. Add an external model provider
2. Update your chat completion requests

To add an external model provider to your project, follow the instructions in [External Models](/features/external-models). Once it's been added, continue to the next step.

### Updating your chat completion requests

Set the model parameter in your requests to match this format: `openpipe:<external-model-provider-slug>/<external-model-slug>`.

For example, if you're calling **gpt-4o-2024-08-06** on Azure, the model parameter should be `openpipe:custom-azure-provider/gpt-4o-2024-08-06`.

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    # Find the config values in "Installing the SDK"
    client = OpenAI()

    completion = client.chat.completions.create(
        model="openpipe:custom-azure-provider/gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": "count to 10"}],
        metadata={"prompt_id": "counting", "any_key": "any_value"},
    )
    ```
  </Tab>

  <Tab title="NodeJS">
    ```typescript
    import OpenAI from "openpipe/openai";

    // Find the config values in "Installing the SDK"
    const client = OpenAI();

    const completion = await client.chat.completions.create({
      model: "openpipe:custom-azure-provider/gpt-4o-2024-08-06",
      messages: [{ role: "user", content: "Count to 10" }],
      metadata: {
        prompt_id: "counting",
        any_key: "any_value",
      },
    });
    ```
  </Tab>
</Tabs>

External models can also be used for filtering and relabeling your data. We currently support custom external
models for providers with openai and azure-compatible endpoints. If you'd like support for an external provider with a different API format, send a request to [hello@openpipe.ai](mailto:hello@openpipe.ai).
