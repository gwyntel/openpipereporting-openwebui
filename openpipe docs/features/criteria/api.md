# API Endpoints

> Use the Criteria API for runtime evaluation and offline testing.

After you've defined and aligned your judge criteria, you can access them via API endpoints for both runtime evaluation (**Best of N** sampling) and offline testing.

### Runtime Evaluation

<Info>
  See the Chat Completion [docs](/features/chat-completions/overview) and [API
  Reference](/api-reference/post-chatcompletions) for more information on making chat completions
  with OpenPipe.
</Info>

When making a request to the `/chat/completions` endpoint, you can specify a list of criteria to run immediately after a completion is generated. We recommend generating multiple responses from the same prompt, each of which will be scored by the specified criteria. The responses will be sorted by their combined score across all criteria, from highest to lowest. This technique is known as **[Best of N](https://huggingface.co/docs/trl/en/best_of_n)** sampling.

To invoke criteria, add an `op-criteria` header to your request with a list of criterion IDs, like so:

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe import OpenAI

    # Find the config values in "Installing the SDK"
    client = OpenAI()

    completion = client.chat.completions.create(
        model="openai:gpt-4o-mini",
        messages=[{"role": "system", "content": "count to 10"}],
        metadata={
            "prompt_id": "counting",
            "any_key": "any_value",
        },
        n=5,
        extra_headers={"op-criteria": '["criterion-1@v1", "criterion-2"]'},
    )

    best_response = completion.choices[0]
    ```
  </Tab>

  <Tab title="NodeJS">
    ```typescript
    import OpenAI from "openpipe/openai";

    // Find the config values in "Installing the SDK"
    const client = OpenAI();

    const completion = await client.chat.completions.create({
      model: "openai:gpt-4o-mini",
      messages: [{ role: "user", content: "Count to 10" }],
      metadata: {
        prompt_id: "counting",
        any_key: "any_value",
      },
      n: 5,
      headers: {
        "op-criteria": '["criterion-1@v1", "criterion-2"]',
      },
    });

    const bestResponse = completion.choices[0];
    ```
  </Tab>

  <Tab title="cURL">
    ```bash
    curl --request POST \
    --url https://app.openpipe.ai/api/v1/chat/completions \
    --header "Authorization: Bearer $OPENPIPE_API_KEY" \
    --header 'Content-Type: application/json' \
    --header 'op-criteria: ["criterion-1@v1", "criterion-2"]' \
    --data '{
    "model": "openai:gpt-4o-mini",
    "messages": [
        {
            "role": "user",
            "content": "Count to 10"
        },
    ],
    "store": true,
    "n": 5,
    "metadata": {
        "prompt_id": "counting",
        "any_key": "any_value",
    }
    }'
    ```
  </Tab>
</Tabs>

Specified criteria can either be versioned, like `criterion-1@v1`, or default to the latest criterion version, like `criterion-2`.

In addition to the usual fields, each chat completion choice will now include a `criteria_results` object, which contains the judgements of the specified criteria. The array of completion choices will take the following form:

```json
[
  {
    "finish_reason": "stop",
    "index": 0,
    "message": {
      "content": "1, 2, 3.",
      "refusal": null,
      "role": "assistant"
    },
    "logprobs": null,
    "criteria_results": {
      "criterion-1": {
        "status": "success",
        "score": 1,
        "explanation": "..."
      },
      "criterion-2": {
        "status": "success",
        "score": 0.6,
        "explanation": "..."
      }
    }
  },
  {
    ...
  }
]
```

### Offline Testing

<Info>See the [API Reference](/api-reference/post-criteriajudge) for more details.</Info>

To check the quality of a previously generated output against a specific criterion, use the `/criteria/judge` endpoint. You can request judgements using either the TypeScript or Python SDKs, or through a cURL request.

<Tabs>
  <Tab title="Python">
    ```python
    from openpipe.client import OpenPipe

    op_client = OpenPipe()

    result = op_client.get_criterion_judgement(
        criterion_id="criterion-1@v1", # if no version is specified, the latest version is used
        input={"messages": messages},
        output=output,
    )
    ```
  </Tab>

  <Tab title="NodeJS">
    ```typescript
    import OpenPipe from "openpipe/client";

    const opClient = OpenPipe();

    const result = await opClient.getCriterionJudgement({
      criterion_id: "criterion-1@v1", // if no version is specified, the latest version is used
      input: {
        messages,
      },
      output: { role: "assistant", content: "1, 2, 3" },
    });
    ```
  </Tab>
</Tabs>

```