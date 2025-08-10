# Anthropic Proxy

If you'd like to make chat completion requests to Anthropic models without modifying your prompt schema, you can proxy OpenAI-compatible requests through OpenPipe, and we'll handle
the translation for you.

To proxy requests to Anthropic models, first add your Anthropic API Key to your project settings. Then, adjust the **model** parameter of your requests to be the name of the model you
wish to query, prepended with the string `anthropic:`. For example, to make a request to `claude-3-5-sonnet-20241022`, use the following code:

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    # Find the config values in "Installing the SDK"
    client = OpenAI()

    completion = client.chat.completions.create(
        model="anthropic:claude-3-5-sonnet-20241022",
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
      model: "anthropic:claude-3-5-sonnet-20241022",
      messages: [{ role: "user", content: "Count to 10" }],
      metadata: {
        prompt_id: "counting",
        any_key: "any_value",
      },
    });
    ```
  </Tab>
</Tabs>

For your reference, here is a list of the most commonly used Anthropic models formatted for the OpenPipe proxy:

* `anthropic:claude-3-7-sonnet-20250219`
* `anthropic:claude-3-5-sonnet-20241022`
* `anthropic:claude-3-opus-20240229`
* `anthropic:claude-3-sonnet-20240229`
* `anthropic:claude-3-haiku-20240307`

Additionally, you can always stay on the latest version of the model by using an abbreviated model name:

* `anthropic:claude-3-7-sonnet`
* `anthropic:claude-3-5-sonnet`
* `anthropic:claude-3-opus`
* `anthropic:claude-3-sonnet`
* `anthropic:claude-3-haiku`

If you'd like to make requests directly to Anthropic models, you can do that externally using the Anthropic SDK, and report your logs using the
asynchronous [reporting API](/features/request-logs/reporting-anthropic).
