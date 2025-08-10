# Judge Criteria

> Get a judgement of a completion against the specified criterion

## OpenAPI

````yaml post /criteria/judge
paths:
  path: /criteria/judge
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
              criterion_id:
                allOf:
                  - type: string
                    description: The ID of the criterion to judge.
              input:
                allOf:
                  - type: object
                    properties:
                      messages:
                        type: array
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
                        description: >-
                          All messages sent to the model when generating the
                          output.
                      tool_choice:
                        anyOf:
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
                        description: >-
                          The tool choice to use when generating the output, if
                          any.
                      tools:
                        type: array
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
                        description: >-
                          The tools available to the model when generating the
                          output, if any.
                    required:
                      - messages
                    additionalProperties: false
              output:
                allOf:
                  - type: object
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
                    description: The completion message of the model.
            required: true
            requiredProperties:
              - criterion_id
              - output
            additionalProperties: false
        examples:
          example:
            value:
              criterion_id: <string>
              input:
                messages:
                  - role: system
                    content: <string>
                    name: <string>
                tool_choice: none
                tools:
                  - function:
                      name: <string>
                      parameters: {}
                      description: <string>
                      strict: true
                    type: function
              output:
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
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              score:
                allOf:
                  - type: number
                    description: >-
                      A score of 0 means the output failed this completion, and
                      a score of 1 means it passed. A criteria may also return a
                      decimal scores between 0 and 1, indicating the model's
                      confidence or 'likelihood' that the criteria passed.
              explanation:
                allOf:
                  - type: string
                    description: >-
                      An explanation of the score including the model's
                      reasoning, if applicable.
              usage:
                allOf:
                  - type: object
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
            requiredProperties:
              - score
            additionalProperties: false
        examples:
          example:
            value:
              score: 123
              explanation: <string>
              usage:
                total_tokens: 123
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