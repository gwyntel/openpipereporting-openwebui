# Gemini Proxy

OpenPipe can translate your existing OpenAI chat completion requests to work with Gemini models automatically, allowing you to use Gemini without changing your prompt format.

After adding your Google AI Studio API Key in your project settings, specify the Gemini **model** you want to use by adding the `gemini:` prefix to the model name in your requests:

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    # Find the config values in "Installing the SDK"
    client = OpenAI()

    completion = client.chat.completions.create(
        model="gemini:gemini-1.5-flash",
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
      model: "gemini:gemini-1.5-flash",
      messages: [{ role: "user", content: "Count to 10" }],
      metadata: {
        prompt_id: "counting",
        any_key: "any_value",
      },
    });
    ```
  </Tab>
</Tabs>

For your reference, here is a list of the most commonly used Gemini models formatted for the OpenPipe proxy:

* `gemini:gemini-1.5-flash-002`
* `gemini:gemini-1.5-flash-8b-001`
* `gemini:gemini-1.5-pro-002`
* `gemini:gemini-exp-1206`
* `gemini:gemini-2.0-flash-exp`

Additionally, you can always stay on the latest version of the model by using an abbreviated model name:

* `gemini:gemini-1.5-flash`
* `gemini:gemini-1.5-flash-8b`
* `gemini:gemini-1.5-pro`
* `gemini:gemini-2.0-flash`
