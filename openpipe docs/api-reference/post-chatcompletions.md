# Chat Completions

> OpenAI-compatible route for generating inference and optionally logging the request.

## OpenAPI

````yaml post /chat/completions
paths:
  path: /chat/completions
  method: post
  servers:
    - url: https://api.openpipe.ai/api/v1
  request:
    security:
      - title: Authorization
        parameters:
          query: {}
          header:
            Authorization:
              type: http
              scheme: bearer
          cookie: {}
    parameters:
      path: {}
      query: {}
      header: {}
      cookie: {}
    body:
      application/json:
        schemaArray:
          - type: object
            properties:
              messages:
                allOf:
                  - type: array
                    items:
                      anyOf:
                        - type: object
                          properties:
                            role:
                              type: string
                              enum:
                                - system
                            content:
                              anyOf:
                                - type: string
                                - type: array
                                  items:
                                    type: object
                                    properties:
                                      type:
                                        type: string
                                        enum:
                                          - text
                                      text:
                                        type: string
                                    required:
                                      - type
                                      - text
                                    additionalProperties: false
                              default: ''
                            name:
                              type: string
                          required:
                            - role
                          additionalProperties: false
                        - type: object
                          properties:
                            role:
                              type: string
                              enum:
                                - user
                            content:
                              anyOf:
                                - type: string
                                - type: array
                                  items:
                                    anyOf:
                                      - type: object
                                        properties:
                                          type:
                                            type: string
                                            enum:
                                              - text
                                          text:
                                            type: string
                                        required:
                                          - type
                                          - text
                                        additionalProperties: false
                                      - type: object
                                        properties:
                                          type:
                                            type: string
                                            enum:
                                              - image_url
                                          image_url:
                                            type: object
                                            properties:
                                              detail:
                                                anyOf:
                                                  - type: string
                                                    enum:
                                                      - auto
                                                  - type: string
                                                    enum:
                                                      - low
                                                  - type: string
                                                    enum:
                                                      - high
                                              url:
                                                type: string
                                            required:
                                              - url
                                            additionalProperties: false
                                        required:
                                          - type
                                          - image_url
                                        additionalProperties: false
                                      - type: object
                                        properties:
                                          type:
                                            type: string
                                            enum:
                                              - input_audio
                                          input_audio:
                                            type: object
                                            properties:
                                              data:
                                                type: string
                                              format:
                                                type: string
                                                enum:
                                                  - wav
                                                  - mp3
                                            required:
                                              - data
                                              - format
                                            additionalProperties: false
                                        required:
                                          - type
                                          - input_audio
                                        additionalProperties: false
                              default: ''
                            name:
                              type: string
                          required:
                            - role
                          additionalProperties: false
                        - type: object
                          properties:
                            role:
                              type: string
                              enum:
                                - assistant
                            audio:
                              type: object
                              properties:
                                id:
                                  type: string
                              required:
                                - id
                              additionalProperties: false
                              nullable: true
                            content:
                              anyOf:
                                - type: string
                                - type: array
                                  items:
                                    anyOf:
                                      - type: object
                                        properties:
                                          type:
                                            type: string
                                            enum:
                                              - text
                                          text:
                                            type: string
                                        required:
                                          - type
                                          - text
                                        additionalProperties: false
                                      - type: object
                                        properties:
                                          type:
                                            type: string
                                            enum:
                                              - refusal
                                          refusal:
                                            type: string
                                        required:
                                          - type
                                          - refusal
                                        additionalProperties: false
                                - enum:
                                    - 'null'
                                  nullable: true
                              default: null
                            function_call:
                              type: object
                              properties:
                                name:
                                  type: string
                                  default: ''
                                arguments:
                                  type: string
                                  default: ''
                              additionalProperties: false
                              nullable: true
                            tool_calls:
                              type: array
                              items:
                                type: object
                                properties:
                                  id:
                                    type: string
                                  function:
                                    type: object
                                    properties:
                                      name:
                                        type: string
                                      arguments:
                                        type: string
                                    required:
                                      - name
                                      - arguments
                                    additionalProperties: false
                                  type:
                                    type: string
                                    enum:
                                      - function
                                required:
                                  - id
                                  - function
                                  - type
                                additionalProperties: false
                              nullable: true
                            name:
                              type: string
                            refusal:
                              type: string
                              nullable: true
                            annotations:
                              type: array
                              items:
                                type: object
                                properties:
                                  type:
                                    type: string
                                    enum:
                                      - url_citation
                                  url_citation:
                                    type: object
                                    properties:
                                      start_index:
                                        type: number
                                      end_index:
                                        type: number
                                      title:
                                        type: string
                                      url:
                                        type: string
                                    required:
                                      - start_index
                                      - end_index
                                      - title
                                      - url
                                    additionalProperties: false
                                required:
                                  - type
                                  - url_citation
                                additionalProperties: false
                          required:
                            - role
                          additionalProperties: false
                        - type: object
                          properties:
                            role:
                              type: string
                              enum:
                                - developer
                            content:
                              anyOf:
                                - type: string
                                - type: array
                                  items:
                                    type: object
                                    properties:
                                      type:
                                        type: string
                                        enum:
                                          - text
                                      text:
                                        type: string
                                    required:
                                      - type
                                      - text
                                    additionalProperties: false
                              default: ''
                            name:
                              type: string
                          required:
                            - role
                          additionalProperties: false
                        - type: object
                          properties:
                            role:
                              type: string
                              enum:
                                - tool
                            content:
                              anyOf:
                                - type: string
                                - type: array
                                  items:
                                    type: object
                                    properties:
                                      type:
                                        type: string
                                        enum:
                                          - text
                                      text:
                                        type: string
                                    required:
                                      - type
                                      - text
                                    additionalProperties: false
                              default: ''
                            tool_call_id:
                              type: string
                          required:
                            - role
                            - tool_call_id
                          additionalProperties: false
                        - type: object
                          properties:
                            role:
                              type: string
                              enum:
                                - function
                            name:
                              type: string
                            content:
                              anyOf:
                                - type: string
                                - enum:
                                    - 'null'
                                  nullable: true
                          required:
                            - role
                            - name
                            - content
                          additionalProperties: false
              model:
                allOf:
                  - type: string
              audio:
                allOf:
                  - type: object
                    properties:
                      format:
                        type: string
                        enum:
                          - wav
                          - mp3
                          - flac
                          - opus
                          - pcm16
                      voice:
                        type: string
                        enum:
                          - alloy
                          - ash
                          - ballad
                          - coral
                          - echo
                          - sage
                          - shimmer
                          - verse
                    required:
                      - format
                      - voice
                    additionalProperties: false
                    nullable: true
              function_call:
                allOf:
                  - anyOf:
                      - type: string
                        enum:
                          - none
                      - type: string
                        enum:
                          - auto
                      - type: object
                        properties:
                          name:
                            type: string
                        required:
                          - name
                        additionalProperties: false
              functions:
                allOf:
                  - type: array
                    items:
                      type: object
                      properties:
                        name:
                          type: string
                        parameters:
                          type: object
                          additionalProperties: {}
                        description:
                          type: string
                        strict:
                          type: boolean
                          nullable: true
                      required:
                        - name
                      additionalProperties: false
              tool_choice:
                allOf:
                  - anyOf:
                      - type: string
                        enum:
                          - none
                      - type: string
                        enum:
                          - auto
                      - type: string
                        enum:
                          - required
                      - type: object
                        properties:
                          type:
                            type: string
                            enum:
                              - function
                            default: function
                          function:
                            type: object
                            properties:
                              name:
                                type: string
                            required:
                              - name
                            additionalProperties: false
                            default:
                              name: ''
                        additionalProperties: false
              tools:
                allOf:
                  - type: array
                    items:
                      type: object
                      properties:
                        function:
                          type: object
                          properties:
                            name:
                              type: string
                            parameters:
                              type: object
                              additionalProperties: {}
                            description:
                              type: string
                            strict:
                              type: boolean
                              nullable: true
                          required:
                            - name
                          additionalProperties: false
                        type:
                          type: string
                          enum:
                            - function
                      required:
                        - function
                        - type
                      additionalProperties: false
              'n':
                allOf:
                  - type: number
              max_tokens:
                allOf:
                  - type: number
                    nullable: true
              max_completion_tokens:
                allOf:
                  - type: number
                    nullable: true
              temperature:
                allOf:
                  - type: number
              top_p:
                allOf:
                  - type: number
                    nullable: true
              presence_penalty:
                allOf:
                  - type: number
                    nullable: true
              frequency_penalty:
                allOf:
                  - type: number
                    nullable: true
              stop:
                allOf:
                  - anyOf:
                      - type: string
                      - type: array
                        items:
                          type: string
                    nullable: true
              response_format:
                allOf:
                  - anyOf:
                      - type: object
                        properties:
                          type:
                            type: string
                            enum:
                              - text
                        required:
                          - type
                        additionalProperties: false
                      - type: object
                        properties:
                          type:
                            type: string
                            enum:
                              - json_object
                        required:
                          - type
                        additionalProperties: false
                      - type: object
                        properties:
                          type:
                            type: string
                            enum:
                              - json_schema
                          json_schema:
                            type: object
                            properties:
                              name:
                                type: string
                              description:
                                type: string
                              schema:
                                type: object
                                additionalProperties: {}
                              strict:
                                type: boolean
                                nullable: true
                            required:
                              - name
                            additionalProperties: false
                        required:
                          - type
                          - json_schema
                        additionalProperties: false
              logprobs:
                allOf:
                  - type: boolean
              top_logprobs:
                allOf:
                  - type: number
                    nullable: true
              stream_options:
                allOf:
                  - type: object
                    properties:
                      include_usage:
                        type: boolean
                    required:
                      - include_usage
                    additionalProperties: false
              store:
                allOf:
                  - type: boolean
              metadata:
                allOf:
                  - type: object
                    additionalProperties:
                      type: string
                    nullable: true
              stream:
                allOf:
                  - type: boolean
                    default: false
            required: true
            requiredProperties:
              - messages
              - model
        examples:
          example:
            value:
              messages:
                - role: system
                  content: <string>
                  name: <string>
              model: <string>
              audio:
                format: wav
                voice: alloy
              function_call: none
              functions:
                - name: <string>
                  parameters: {}
                  description: <string>
                  strict: true
              tool_choice: none
              tools:
                - function:
                    name: <string>
                    parameters: {}
                    description: <string>
                    strict: true
                  type: function
              'n': 123
              max_tokens: 123
              max_completion_tokens: 123
              temperature: 123
              top_p: 123
              presence_penalty: 123
              frequency_penalty: 123
              stop: <string>
              response_format:
                type: text
              logprobs: true
              top_logprobs: 123
              stream_options:
                include_usage: true
              store: true
              metadata: {}
              stream: false
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              id:
                allOf:
                  - type: string
              object:
                allOf:
                  - type: string
                    enum:
                      - chat.completion
              created:
                allOf:
                  - type: number
              model:
                allOf:
                  - type: string
              choices:
                allOf:
                  - type: array
                    items:
                      type: object
                      properties:
                        finish_reason:
                          anyOf:
                            - type: string
                              enum:
                                - length
                            - type: string
                              enum:
                                - function_call
                            - type: string
                              enum:
                                - tool_calls
                            - type: string
                              enum:
                                - stop
                            - type: string
                              enum:
                                - content_filter
                        index:
                          type: number
                        message:
                          type: object
                          properties:
                            reasoning_content:
                              type: string
                              nullable: true
                            content:
                              type: string
                              nullable: true
                              default: null
                            refusal:
                              type: string
                              nullable: true
                            role:
                              type: string
                              enum:
                                - assistant
                            function_call:
                              type: object
                              properties:
                                name:
                                  type: string
                                  default: ''
                                arguments:
                                  type: string
                                  default: ''
                              additionalProperties: false
                              nullable: true
                            tool_calls:
                              type: array
                              items:
                                type: object
                                properties:
                                  id:
                                    type: string
                                  function:
                                    type: object
                                    properties:
                                      name:
                                        type: string
                                      arguments:
                                        type: string
                                    required:
                                      - name
                                      - arguments
                                    additionalProperties: false
                                  type:
                                    type: string
                                    enum:
                                      - function
                                required:
                                  - id
                                  - function
                                  - type
                                additionalProperties: false
                              nullable: true
                          required:
                            - role
                          additionalProperties: false
                        logprobs:
                          type: object
                          properties:
                            content:
                              type: array
                              items:
                                type: object
                                properties:
                                  token:
                                    type: string
                                  bytes:
                                    type: array
                                    items:
                                      type: number
                                    nullable: true
                                  logprob:
                                    type: number
                                  top_logprobs:
                                    type: array
                                    items:
                                      type: object
                                      properties:
                                        token:
                                          type: string
                                        bytes:
                                          type: array
                                          items:
                                            type: number
                                          nullable: true
                                        logprob:
                                          type: number
                                      required:
                                        - token
                                        - bytes
                                        - logprob
                                      additionalProperties: false
                                required:
                                  - token
                                  - bytes
                                  - logprob
                                  - top_logprobs
                                additionalProperties: false
                              nullable: true
                              default: null
                            refusal:
                              type: array
                              items:
                                type: object
                                properties:
                                  token:
                                    type: string
                                  bytes:
                                    type: array
                                    items:
                                      type: number
                                    nullable: true
                                  logprob:
                                    type: number
                                  top_logprobs:
                                    type: array
                                    items:
                                      type: object
                                      properties:
                                        token:
                                          type: string
                                        bytes:
                                          type: array
                                          items:
                                            type: number
                                          nullable: true
                                        logprob:
                                          type: number
                                      required:
                                        - token
                                        - bytes
                                        - logprob
                                      additionalProperties: false
                                required:
                                  - token
                                  - bytes
                                  - logprob
                                  - top_logprobs
                                additionalProperties: false
                              nullable: true
                              default: null
                          additionalProperties: false
                          nullable: true
                          default: null
                        content_filter_results:
                          type: object
                          properties: {}
                          additionalProperties: true
                        criteria_results:
                          type: object
                          additionalProperties:
                            anyOf:
                              - type: object
                                properties:
                                  status:
                                    type: string
                                    enum:
                                      - success
                                  score:
                                    type: number
                                  explanation:
                                    type: string
                                  errorCode:
                                    type: number
                                  errorMessage:
                                    type: string
                                required:
                                  - status
                                  - score
                                additionalProperties: false
                              - type: object
                                properties:
                                  status:
                                    type: string
                                    enum:
                                      - error
                                  score:
                                    type: number
                                  explanation:
                                    type: string
                                  errorCode:
                                    type: number
                                  errorMessage:
                                    type: string
                                required:
                                  - status
                                  - errorCode
                                  - errorMessage
                                additionalProperties: false
                      required:
                        - finish_reason
                        - index
                        - message
                      additionalProperties: false
              usage:
                allOf:
                  - type: object
                    properties:
                      prompt_tokens:
                        type: number
                      completion_tokens:
                        type: number
                      total_tokens:
                        type: number
                      prompt_cache_hit_tokens:
                        type: number
                      prompt_cache_miss_tokens:
                        type: number
                      completion_tokens_details:
                        type: object
                        properties:
                          reasoning_tokens:
                            type: number
                            nullable: true
                          audio_tokens:
                            type: number
                            nullable: true
                          text_tokens:
                            type: number
                            nullable: true
                          accepted_prediction_tokens:
                            type: number
                            nullable: true
                          rejected_prediction_tokens:
                            type: number
                            nullable: true
                        additionalProperties: false
                        nullable: true
                      prompt_tokens_details:
                        type: object
                        properties:
                          cached_tokens:
                            type: number
                            nullable: true
                          audio_tokens:
                            type: number
                            nullable: true
                        additionalProperties: false
                        nullable: true
                      criteria:
                        type: object
                        additionalProperties:
                          type: object
                          properties:
                            total_tokens:
                              type: number
                              description: >-
                                The total number of tokens used to generate the
                                criterion judgement. Only returned for
                                OpenPipe-trained reward models currently.
                          required:
                            - total_tokens
                          additionalProperties: false
                    required:
                      - prompt_tokens
                      - completion_tokens
                      - total_tokens
                    additionalProperties: false
            requiredProperties:
              - id
              - object
              - created
              - model
              - choices
            additionalProperties: false
          - type: 'null'
          - type: any
        examples:
          example:
            value:
              id: <string>
              object: chat.completion
              created: 123
              model: <string>
              choices:
                - finish_reason: length
                  index: 123
                  message:
                    reasoning_content: <string>
                    content: null
                    refusal: <string>
                    role: assistant
                    function_call:
                      name: ''
                      arguments: ''
                    tool_calls:
                      - id: <string>
                        function:
                          name: <string>
                          arguments: <string>
                        type: function
                  logprobs: null
                  content_filter_results: {}
                  criteria_results: {}
              usage:
                prompt_tokens: 123
                completion_tokens: 123
                total_tokens: 123
                prompt_cache_hit_tokens: 123
                prompt_cache_miss_tokens: 123
                completion_tokens_details:
                  reasoning_tokens: 123
                  audio_tokens: 123
                  text_tokens: 123
                  accepted_prediction_tokens: 123
                  rejected_prediction_tokens: 123
                prompt_tokens_details:
                  cached_tokens: 123
                  audio_tokens: 123
                criteria: {}
        description: Successful response
    default:
      application/json:
        schemaArray:
          - type: object
            properties:
              message:
                allOf:
                  - type: string
              code:
                allOf:
                  - type: string
              issues:
                allOf:
                  - type: array
                    items:
                      type: object
                      properties:
                        message:
                          type: string
                      required:
                        - message
                      additionalProperties: false
            requiredProperties:
              - message
              - code
            additionalProperties: false
        examples:
          example:
            value:
              message: <string>
              code: <string>
              issues:
                - message: <string>
        description: Error response
  deprecated: false
  type: path
components:
  schemas: {}

````