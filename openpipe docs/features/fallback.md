# Fallback options

>  Safeguard your application against potential failures, timeouts, or instabilities that may occur when using experimental or newly released models.

Fallback is a feature that ensures a seamless experience and guarantees 100% uptime when working with new or unstable models.

When fallback is enabled, any failed API calls will be automatically retried using OpenAI or any OpenAI-compatible client.

## Fallback to OpenAI

To enable fallback to OpenAI, you can simply pass the `fallback` option to the `openpipe` object with the `model` property set to the OpenAI model you want to fall back to.

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    client = OpenAI()

    completion = client.chat.completions.create(
        model="openpipe:my-ft-model",
        messages=[{"role": "system", "content": "count to 10"}],
        openpipe={
            "fallback": {
                "model": "gpt-4-turbo"
            }
        },
    )

    ```
  </Tab>

  <Tab title="NodeJS">
    ```typescript
    import OpenAI from "openpipe/openai";

    const openai = new OpenAI();

    const completion = await openai.chat.completions.create({
      messages: [{ role: "user", content: "Count to 10" }],
      model: "openpipe:my-ft-model",
      openpipe: {
        fallback: { model: "gpt-4-turbo" },
      },
    });
    ```
  </Tab>
</Tabs>

## Timeout Fallback

If a request takes too long to execute, you can set a timeout for the fallback.
In the example below, the request will fall back to OpenAI after 10 seconds.

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    client = OpenAI(timeout=10) # initial OpenPipe call timeout in seconds

    completion = client.chat.completions.create(
        model="openpipe:my-ft-model",
        messages=[{"role": "system", "content": "count to 10"}],
        openpipe={
            "fallback": {
                "model": "gpt-4-turbo",
                # optional fallback timeout. Defaults to the timeout specified in the client, or OpenAI default timeout if not set.
                "timeout": 20 # seconds
            }
        },
    )

    ```
  </Tab>

  <Tab title="NodeJS">
    ```typescript
    import OpenAI from "openpipe/openai";

    const openai = new OpenAI();

    const completion = await openai.chat.completions.create(
      {
        messages: [{ role: "user", content: "Count to 10" }],
        model: "openpipe:my-ft-model",
        openpipe: {
          fallback: {
            model: "gpt-4-turbo",
            // optional fallback timeout. Defaults to the timeout specified in client options, or OpenAI default timeout if not set.
            timeout: 20 * 1000, // milliseconds
          },
        },
      },
      {
        timeout: 10 * 1000, // initial OpenPipe call timeout in milliseconds
      },
    );
    ```
  </Tab>
</Tabs>

## Fallback to Custom OpenAI Compatible Client

If you want to use another OpenAI-compatible fallback client, you can pass a `fallback_client` to the `openpipe` object.

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    client = OpenAI(
        openpipe={
            "fallback_client": OpenAICompatibleClient(api_key="client api key")
        }
    );

    completion = client.chat.completions.create(
        model="openpipe:my-ft-model",
        messages=[{"role": "system", "content": "count to 10"}],
        openpipe={
            "fallback": { "model": "gpt-4-turbo" }
        },
    )

    ```
  </Tab>

  <Tab title="NodeJS">
    ```typescript
    import OpenAI from "openpipe/openai";

    const openai = new OpenAI({
      openpipe: {
        fallbackClient: new OpenAICompatibleClient({ apiKey: "client api key" }),
      },
    });

    const completion = await openai.chat.completions.create({
      messages: [{ role: "user", content: "Count to 10" }],
      model: "openpipe:my-ft-model",
      openpipe: {
        fallback: { model: "gpt-4-turbo" },
      },
    });
    ```
  </Tab>
</Tabs>
