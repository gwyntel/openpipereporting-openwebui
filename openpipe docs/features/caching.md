# Caching

>  Improve performance and reduce costs by caching previously generated responses.

When caching is enabled, our service stores the responses generated for each unique request. If an identical request is made in the future, instead of processing the request again, the cached response is instantly returned. This eliminates the need for redundant computations, resulting in faster response times and reduced API usage costs.
Cached responses expire after one week.

Caching is currently in a free beta preview.

## Enabling Caching

Caching is disabled by default. To enable caching for your requests, you can set the `cache` property of the openpipe object to one of the following values:

* `readWrite`: Cache is read from and written to.
* `readOnly`: Cache is read from, but not written to.
* `writeOnly`: Cache is written to, but not read from.

If you are making requests through our proxy, add the `op-cache` header to your requests. For any of these settings, if a cache entry is not found, the request will be processed as normal.

<Tabs>
  <Tab title="cURL Request">
    ```bash
    curl --request POST \
      --url https://api.openpipe.ai/api/v1/chat/completions \
      --header "Authorization: Bearer YOUR_OPENPIPE_API_KEY" \
      --header 'Content-Type: application/json' \
      --header 'op-cache: readWrite' \
      --data '{
      "model": "openpipe:your-fine-tuned-model-id",
      "messages": [
        {
          "role": "system",
          "content": "count to 5"
        }
      ]
    }'
    ```
  </Tab>

  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    client = OpenAI()

    completion = client.chat.completions.create(
        model="openpipe:your-fine-tuned-model-id",
        messages=[{"role": "system", "content": "count to 5"}],
        openpipe={
            "cache": "readWrite"
        },
    )

    ```
  </Tab>

  <Tab title="NodeJS">
    ```typescript
    import OpenAI from "openpipe/openai";

    const openai = new OpenAI();

    const completion = await openai.chat.completions.create({
      messages: [{ role: "user", content: "count to 5" }],
      model: "openpipe:your-fine-tuned-model-id",
      openpipe: {
        cache: "readWrite",
      },
    });
    ```
  </Tab>
</Tabs>
