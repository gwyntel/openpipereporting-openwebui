# Report Anthropic

> Record request logs from Anthropic models

## OpenAPI

```yaml post /report-anthropic
paths:
  path: /report-anthropic
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
              requestedAt:
                allOf:
                  - type: number
                    description: Unix timestamp in milliseconds
              receivedAt:
                allOf:
                  - type: number
                    description: Unix timestamp in milliseconds
              reqPayload:
                allOf:
                  - anyOf:
                      - type: object
                        properties:
                          max_tokens:
                            type: number
                          messages:
                            type: array
                            items:
                              type: object
                              properties:
                                content:
                                  anyOf:
                                    - type: string
                                    - type: array
                                      items:
                                        anyOf:
                                          - type: object
                                            properties:
                                              text:
                                                type: string
                                              type:
                                                type: string
                                                enum:
                                                  - text
                                              cache_control:
                                                anyOf:
                                                  - type: object
                                                    properties:
                                                      type:
                                                        type: string
                                                        enum:
                                                          - ephemeral
                                                    required:
                                                      - type
                                                    additionalProperties: false
                                                  - enum:
                                                      - 'null'
                                                    nullable: true
                                              citations:
                                                type: array
                                                items:
                                                  anyOf:
                                                    - type: object
                                                      properties:
                                                        cited_text:
                                                          type: string
                                                        document_index:
                                                          type: number
                                                        document_title:
                                                          type: string
                                                          nullable: true
                                                        end_char_index:
                                                          type: number
                                                        start_char_index:
                                                          type: number
                                                        type:
                                                          type: string
                                                          enum:
                                                            - char_location
                                                      required:
                                                        - cited_text
                                                        - document_index
                                                        - document_title
                                                        - end_char_index
                                                        - start_char_index
                                                        - type
                                                      additionalProperties: false
                                                    - type: object
                                                      properties:
                                                        cited_text:
                                                          type: string
                                                        document_index:
                                                          type: number
                                                        document_title:
                                                          type: string
                                                          nullable: true
                                                        end_page_number:
                                                          type: number
                                                        start_page_number:
                                                          type: number
                                                        type:
                                                          type: string
                                                          enum:
                                                            - page_location
                                                      required:
                                                        - cited_text
                                                        - document_index
                                                        - document_title
                                                        - end_page_number
                                                        - start_page_number
                                                        - type
                                                      additionalProperties: false
                                                    - type: object
                                                      properties:
                                                        cited_text:
                                                          type: string
                                                        document_index:
                                                          type: number
                                                        document_title:
                                                          type: string
                                                          nullable: true
                                                        end_block_index:
                                                          type: number
                                                        start_block_index:
                                                          type: number
                                                        type:
                                                          type: string
                                                          enum:
                                                            - content_block_location
                                                      required:
                                                        - cited_text
                                                        - document_index
                                                        - document_title
                                                        - end_block_index
                                                        - start_block_index
                                                        - type
                                                      additionalProperties: false
                                                nullable: true
                                            required:
                                              - text
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              source:
                                                type: object
                                                properties:
                                                  data:
                                                    type: string
                                                  media_type:
                                                    anyOf:
                                                      - type: string
                                                        enum:
                                                          - image/jpeg
                                                      - type: string
                                                        enum:
                                                          - image/png
                                                      - type: string
                                                        enum:
                                                          - image/gif
                                                      - type: string
                                                        enum:
                                                          - image/webp
                                                  type:
                                                    type: string
                                                    enum:
                                                      - base64
                                                required:
                                                  - data
                                                  - media_type
                                                  - type
                                                additionalProperties: false
                                              type:
                                                type: string
                                                enum:
                                                  - image
                                              cache_control:
                                                anyOf:
                                                  - type: object
                                                    properties:
                                                      type:
                                                        type: string
                                                        enum:
                                                          - ephemeral
                                                    required:
                                                      - type
                                                    additionalProperties: false
                                                  - enum:
                                                      - 'null'
                                                    nullable: true
                                            required:
                                              - source
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              id:
                                                type: string
                                              input: {}
                                              name:
                                                type: string
                                              type:
                                                type: string
                                                enum:
                                                  - tool_use
                                              cache_control:
                                                anyOf:
                                                  - type: object
                                                    properties:
                                                      type:
                                                        type: string
                                                        enum:
                                                          - ephemeral
                                                    required:
                                                      - type
                                                    additionalProperties: false
                                                  - enum:
                                                      - 'null'
                                                    nullable: true
                                            required:
                                              - id
                                              - name
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              tool_use_id:
                                                type: string
                                              type:
                                                type: string
                                                enum:
                                                  - tool_result
                                              content:
                                                anyOf:
                                                  - type: string
                                                  - type: array
                                                    items:
                                                      anyOf:
                                                        - type: object
                                                          properties:
                                                            text:
                                                              type: string
                                                            type:
                                                              type: string
                                                              enum:
                                                                - text
                                                            cache_control:
                                                              anyOf:
                                                                - type: object
                                                                  properties:
                                                                    type:
                                                                      type: string
                                                                      enum:
                                                                        - ephemeral
                                                                  required:
                                                                    - type
                                                                  additionalProperties: false
                                                                - enum:
                                                                    - 'null'
                                                                  nullable: true
                                                            citations:
                                                              type: array
                                                              items:
                                                                anyOf:
                                                                  - type: object
                                                                    properties:
                                                                      cited_text:
                                                                        type: string
                                                                      document_index:
                                                                        type: number
                                                                      document_title:
                                                                        type: string
                                                                        nullable: true
                                                                      end_char_index:
                                                                        type: number
                                                                      start_char_index:
                                                                        type: number
                                                                      type:
                                                                        type: string
                                                                        enum:
                                                                          - char_location
                                                                    required:
                                                                      - cited_text
                                                                      - document_index
                                                                      - document_title
                                                                      - end_char_index
                                                                      - start_char_index
                                                                      - type
                                                                    additionalProperties: false
                                                                  - type: object
                                                                    properties:
                                                                      cited_text:
                                                                        type: string
                                                                      document_index:
                                                                        type: number
                                                                      document_title:
                                                                        type: string
                                                                        nullable: true
                                                                      end_page_number:
                                                                        type: number
                                                                      start_page_number:
                                                                        type: number
                                                                      type:
                                                                        type: string
                                                                        enum:
                                                                          - page_location
                                                                    required:
                                                                      - cited_text
                                                                      - document_index
                                                                      - document_title
                                                                      - end_page_number
                                                                      - start_page_number
                                                                      - type
                                                                    additionalProperties: false
                                                                  - type: object
                                                                    properties:
                                                                      cited_text:
                                                                        type: string
                                                                      document_index:
                                                                        type: number
                                                                      document_title:
                                                                        type: string
                                                                        nullable: true
                                                                      end_block_index:
                                                                        type: number
                                                                      start_block_index:
                                                                        type: number
                                                                      type:
                                                                        type: string
                                                                        enum:
                                                                          - content_block_location
                                                                    required:
                                                                      - cited_text
                                                                      - document_index
                                                                      - document_title
                                                                      - end_block_index
                                                                      - start_block_index
                                                                      - type
                                                                    additionalProperties: false
                                                              nullable: true
                                                          required:
                                                            - text
                                                            - type
                                                          additionalProperties: false
                                                        - type: object
                                                          properties:
                                                            source:
                                                              type: object
                                                              properties:
                                                                data:
                                                                  type: string
                                                                media_type:
                                                                  anyOf:
                                                                    - type: string
                                                                      enum:
                                                                        - image/jpeg
                                                                    - type: string
                                                                      enum:
                                                                        - image/png
                                                                    - type: string
                                                                      enum:
                                                                        - image/gif
                                                                    - type: string
                                                                      enum:
                                                                        - image/webp
                                                                type:
                                                                  type: string
                                                                  enum:
                                                                    - base64
                                                              required:
                                                                - data
                                                                - media_type
                                                                - type
                                                              additionalProperties: false
                                                            type:
                                                              type: string
                                                              enum:
                                                                - image
                                                            cache_control:
                                                              anyOf:
                                                                - type: object
                                                                  properties:
                                                                    type:
                                                                      type: string
                                                                      enum:
                                                                        - ephemeral
                                                                  required:
                                                                    - type
                                                                  additionalProperties: false
                                                                - enum:
                                                                    - 'null'
                                                                  nullable: true
                                                          required:
                                                            - source
                                                            - type
                                                          additionalProperties: false
                                              is_error:
                                                type: boolean
                                              cache_control:
                                                anyOf:
                                                  - type: object
                                                    properties:
                                                      type:
                                                        type: string
                                                        enum:
                                                          - ephemeral
                                                    required:
                                                      - type
                                                    additionalProperties: false
                                                  - enum:
                                                      - 'null'
                                                    nullable: true
                                            required:
                                              - tool_use_id
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              source:
                                                anyOf:
                                                  - type: object
                                                    properties:
                                                      data:
                                                        type: string
                                                      media_type:
                                                        type: string
                                                        enum:
                                                          - application/pdf
                                                      type:
                                                        type: string
                                                        enum:
                                                          - base64
                                                    required:
                                                      - data
                                                      - media_type
                                                      - type
                                                    additionalProperties: false
                                                  - type: object
                                                    properties:
                                                      data:
                                                        type: string
                                                      media_type:
                                                        type: string
                                                        enum:
                                                          - text/plain
                                                      type:
                                                        type: string
                                                        enum:
                                                          - text
                                                    required:
                                                      - data
                                                      - media_type
                                                      - type
                                                    additionalProperties: false
                                                  - type: object
                                                    properties:
                                                      content:
                                                        anyOf:
                                                          - type: string
                                                          - type: array
                                                            items:
                                                              anyOf:
                                                                - type: object
                                                                  properties:
                                                                    text:
                                                                      type: string
                                                                    type:
                                                                      type: string
                                                                      enum:
                                                                        - text
                                                                    cache_control:
                                                                      anyOf:
                                                                        - type: object
                                                                          properties:
                                                                            type:
                                                                              type: string
                                                                              enum:
                                                                                - ephemeral
                                                                          required:
                                                                            - type
                                                                          additionalProperties: false
                                                                        - enum:
                                                                            - 'null'
                                                                          nullable: true
                                                                    citations:
                                                                      type: array
                                                                      items:
                                                                        anyOf:
                                                                          - type: object
                                                                            properties:
                                                                              cited_text:
                                                                                type: string
                                                                              document_index:
                                                                                type: number
                                                                              document_title:
                                                                                type: string
                                                                                nullable: true
                                                                              end_char_index:
                                                                                type: number
                                                                              start_char_index:
                                                                                type: number
                                                                              type:
                                                                                type: string
                                                                                enum:
                                                                                  - char_location
                                                                            required:
                                                                              - cited_text
                                                                              - document_index
                                                                              - document_title
                                                                              - end_char_index
                                                                              - start_char_index
                                                                              - type
                                                                            additionalProperties: false
                                                                          - type: object
                                                                            properties:
                                                                              cited_text:
                                                                                type: string
                                                                              document_index:
                                                                                type: number
                                                                              document_title:
                                                                                type: string
                                                                                nullable: true
                                                                              end_page_number:
                                                                                type: number
                                                                              start_page_number:
                                                                                type: number
                                                                              type:
                                                                                type: string
                                                                                enum:
                                                                                  - page_location
                                                                            required:
                                                                              - cited_text
                                                                              - document_index
                                                                              - document_title
                                                                              - end_page_number
                                                                              - start_page_number
                                                                              - type
                                                                            additionalProperties: false
                                                                          - type: object
                                                                            properties:
                                                                              cited_text:
                                                                                type: string
                                                                              document_index:
                                                                                type: number
                                                                              document_title:
                                                                                type: string
                                                                                nullable: true
                                                                              end_block_index:
                                                                                type: number
                                                                              start_block_index:
                                                                                type: number
                                                                              type:
                                                                                type: string
                                                                                enum:
                                                                                  - content_block_location
                                                                            required:
                                                                              - cited_text
                                                                              - document_index
                                                                              - document_title
                                                                              - end_block_index
                                                                              - start_block_index
                                                                              - type
                                                                            additionalProperties: false
                                                                      nullable: true
                                                                  required:
                                                                    - text
                                                                    - type
                                                                  additionalProperties: false
                                                                - type: object
                                                                  properties:
                                                                    source:
                                                                      type: object
                                                                      properties:
                                                                        data:
                                                                          type: string
                                                                        media_type:
                                                                          anyOf:
                                                                            - type: string
                                                                              enum:
                                                                                - image/jpeg
                                                                            - type: string
                                                                              enum:
                                                                                - image/png
                                                                            - type: string
                                                                              enum:
                                                                                - image/gif
                                                                            - type: string
                                                                              enum:
                                                                                - image/webp
                                                                        type:
                                                                          type: string
                                                                          enum:
                                                                            - base64
                                                                      required:
                                                                        - data
                                                                        - media_type
                                                                        - type
                                                                      additionalProperties: false
                                                                    type:
                                                                      type: string
                                                                      enum:
                                                                        - image
                                                                    cache_control:
                                                                      anyOf:
                                                                        - type: object
                                                                          properties:
                                                                            type:
                                                                              type: string
                                                                              enum:
                                                                                - ephemeral
                                                                          required:
                                                                            - type
                                                                          additionalProperties: false
                                                                        - enum:
                                                                            - 'null'
                                                                          nullable: true
                                                                  required:
                                                                    - source
                                                                    - type
                                                                  additionalProperties: false
                                                      type:
                                                        type: string
                                                        enum:
                                                          - content
                                                    required:
                                                      - content
                                                      - type
                                                    additionalProperties: false
                                                  - type: object
                                                    properties:
                                                      type:
                                                        type: string
                                                        enum:
                                                          - url
                                                      url:
                                                        type: string
                                                    required:
                                                      - type
                                                      - url
                                                    additionalProperties: false
                                              type:
                                                type: string
                                                enum:
                                                  - document
                                              cache_control:
                                                anyOf:
                                                  - type: object
                                                    properties:
                                                      type:
                                                        type: string
                                                        enum:
                                                          - ephemeral
                                                    required:
                                                      - type
                                                    additionalProperties: false
                                                  - enum:
                                                      - 'null'
                                                    nullable: true
                                              citations:
                                                type: object
                                                properties:
                                                  enabled:
                                                    type: boolean
                                                additionalProperties: false
                                              context:
                                                anyOf:
                                                  - type: string
                                                  - enum:
                                                      - 'null'
                                                    nullable: true
                                              title:
                                                anyOf:
                                                  - type: string
                                                  - enum:
                                                      - 'null'
                                                    nullable: true
                                            required:
                                              - source
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              thinking:
                                                type: string
                                              signature:
                                                type: string
                                              type:
                                                type: string
                                                enum:
                                                  - thinking
                                            required:
                                              - thinking
                                              - signature
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              data:
                                                type: string
                                              type:
                                                type: string
                                                enum:
                                                  - redacted_thinking
                                            required:
                                              - data
                                              - type
                                            additionalProperties: false
                                role:
                                  anyOf:
                                    - type: string
                                      enum:
                                        - user
                                    - type: string
                                      enum:
                                        - assistant
                              required:
                                - content
                                - role
                              additionalProperties: false
                          model:
                            type: string
                          metadata:
                            type: object
                            properties:
                              user_id:
                                anyOf:
                                  - type: string
                                  - enum:
                                      - 'null'
                                    nullable: true
                            additionalProperties: false
                          stop_sequences:
                            type: array
                            items:
                              type: string
                          stream:
                            type: boolean
                            enum:
                              - true
                          system:
                            anyOf:
                              - type: string
                              - type: array
                                items:
                                  type: object
                                  properties:
                                    text:
                                      type: string
                                    type:
                                      type: string
                                      enum:
                                        - text
                                    cache_control:
                                      anyOf:
                                        - type: object
                                          properties:
                                            type:
                                              type: string
                                              enum:
                                                - ephemeral
                                          required:
                                            - type
                                          additionalProperties: false
                                        - enum:
                                            - 'null'
                                          nullable: true
                                    citations:
                                      type: array
                                      items:
                                        anyOf:
                                          - type: object
                                            properties:
                                              cited_text:
                                                type: string
                                              document_index:
                                                type: number
                                              document_title:
                                                type: string
                                                nullable: true
                                              end_char_index:
                                                type: number
                                              start_char_index:
                                                type: number
                                              type:
                                                type: string
                                                enum:
                                                  - char_location
                                            required:
                                              - cited_text
                                              - document_index
                                              - document_title
                                              - end_char_index
                                              - start_char_index
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              cited_text:
                                                type: string
                                              document_index:
                                                type: number
                                              document_title:
                                                type: string
                                                nullable: true
                                              end_page_number:
                                                type: number
                                              start_page_number:
                                                type: number
                                              type:
                                                type: string
                                                enum:
                                                  - page_location
                                            required:
                                              - cited_text
                                              - document_index
                                              - document_title
                                              - end_page_number
                                              - start_page_number
                                              - type
                                            additionalProperties: false
                                          - type: object
                                            properties:
                                              cited_text:
                                                type: string
                                              document_index:
                                                type: number
                                              document_title:
                                                type: string
                                                nullable: true
                                              end_block_index:
                                                type: number
                                              start_block_index:
                                                type: number
                                              type:
                                                type: string
                                                enum:
                                                  - content_block_location
                                            required:
                                              - cited_text
                                              - document_index
                                              - document_title
                                              - end_block_index
                                              - start_block_index
                                              - type
                                            additionalProperties: false
                                      nullable: true
                                  required:
                                    - text
                                    - type
                                  additionalProperties: false
                          temperature:
                            type: number
                          top_k:
                            type: number
                          top_p:
                            type: number
                          thinking:
                            anyOf:
                              - type: object
                                properties:
                                  budget_tokens:
                                    type: number
                                  type:
                                    type: string
                                    enum:
                                      - enabled
                                required:
                                  - budget_tokens
                                  - type
                                additionalProperties: false
                              - type: object
                                properties:
                                  type:
                                    type: string
                                    enum:
                                      - disabled
                                required:
                                  - type
                                additionalProperties: false
                          tool_choice:
                            anyOf:
                              - type: object
                                properties:
                                  type:
                                    type: string
                                    enum:
                                      - auto
                                  disable_parallel_tool_use:
                                    type: boolean
                                required:
                                  - type
                                additionalProperties: false
                              - type: object
                                properties:
                                  type:
                                    type: string
                                    enum:
                                      - any
                                  disable_parallel_tool_use:
                                    type: boolean
                                required:
                                  - type
                                additionalProperties: false
                              - type: object
                                properties:
                                  type:
                                    type: string
                                    enum:
                                      - tool
                                  name:
                                    type: string
                                  disable_parallel_tool_use:
                                    type: boolean
                                required:
                                  - type
                                  - name
                                additionalProperties: false
                              - type: object
                                properties:
                                  type:
                                    type: string
                                    enum:
                                      - none
                                required:
                                  - type
                                additionalProperties: false
                          tools:
                            type: array
                            items:
                              anyOf:
                                - type: object
                                  properties:
                                    name:
                                      type: string
                                    input_schema:
                                      type: object
                                      properties:
                                        type:
                                          type: string
                                          enum:
                                            - object
                                        properties: {}
                                      required:
                                        - type
                                      additionalProperties: true
                                    description:
                                      type: string
                                    cache_control:
                                      anyOf:
                                        - type: object
                                          properties:
                                            type:
                                              type: string
                                              enum:
                                                - ephemeral
                                          required:
                                            - type
                                          additionalProperties: false
                                        - enum:
                                            - 'null'
                                          nullable: true
                                  required:
                                    - name
                                    - input_schema
                                  additionalProperties: false
                                - type: object
                                  properties:
                                    name:
                                      type: string
                                      enum:
                                        - bash
                                    type:
                                      type: string
                                      enum:
                                        - bash_20250124
                                    cache_control:
                                      anyOf:
                                        - type: object
                                          properties:
                                            type:
                                              type: string
                                              enum:
                                                - ephemeral
                                          required:
                                            - type
                                          additionalProperties: false
                                        - enum:
                                            - 'null'
                                          nullable: true
                                  required:
                                    - name
                                    - type
                                  additionalProperties: false
                                - type: object
                                  properties:
                                    name:
                                      type: string
                                      enum:
                                        - str_replace_editor
                                    type:
                                      type: string
                                      enum:
                                        - text_editor_20250124
                                    cache_control:
                                      anyOf:
                                        - type: object
                                          properties:
                                            type:
                                              type: string
                                              enum:
                                                - ephemeral
                                          required:
                                            - type
                                          additionalProperties: false
                                        - enum:
                                            - 'null'
                                          nullable: true
                                  required:
                                    - name
                                    - type
                                  additionalProperties: false
                        required:
                          - max_tokens
                          - messages
                          - model
                        additionalProperties: true
                    description: JSON-encoded request payload
              respPayload:
                allOf:
                  - type: object
                    properties:
                      id:
                        type: string
                      content:
                        type: array
                        items:
                          anyOf:
                            - type: object
                              properties:
                                text:
                                  type: string
                                type:
                                  type: string
                                  enum:
                                    - text
                                citations:
                                  type: array
                                  items:
                                    anyOf:
                                      - type: object
                                        properties:
                                          cited_text:
                                            type: string
                                          document_index:
                                            type: number
                                          document_title:
                                            type: string
                                            nullable: true
                                          end_char_index:
                                            type: number
                                          start_char_index:
                                            type: number
                                          type:
                                            type: string
                                            enum:
                                              - char_location
                                        required:
                                          - cited_text
                                          - document_index
                                          - document_title
                                          - end_char_index
                                          - start_char_index
                                          - type
                                        additionalProperties: false
                                      - type: object
                                        properties:
                                          cited_text:
                                            type: string
                                          document_index:
                                            type: number
                                          document_title:
                                            type: string
                                            nullable: true
                                          end_page_number:
                                            type: number
                                          start_page_number:
                                            type: number
                                          type:
                                            type: string
                                            enum:
                                              - page_location
                                        required:
                                          - cited_text
                                          - document_index
                                          - document_title
                                          - end_page_number
                                          - start_page_number
                                          - type
                                        additionalProperties: false
                                      - type: object
                                        properties:
                                          cited_text:
                                            type: string
                                          document_index:
                                            type: number
                                          document_title:
                                            type: string
                                            nullable: true
                                          end_block_index:
                                            type: number
                                          start_block_index:
                                            type: number
                                          type:
                                            type: string
                                            enum:
                                              - content_block_location
                                        required:
                                          - cited_text
                                          - document_index
                                          - document_title
                                          - end_block_index
                                          - start_block_index
                                          - type
                                        additionalProperties: false
                                  nullable: true
                                  default: null
                              required:
                                - text
                                - type
                              additionalProperties: false
                            - type: object
                              properties:
                                id:
                                  type: string
                                name:
                                  type: string
                                type:
                                  type: string
                                  enum:
                                    - tool_use
                                input: {}
                              required:
                                - id
                                - name
                                - type
                              additionalProperties: false
                            - type: object
                              properties:
                                thinking:
                                  type: string
                                signature:
                                  type: string
                                type:
                                  type: string
                                  enum:
                                    - thinking
                              required:
                                - thinking
                                - signature
                                - type
                              additionalProperties: false
                            - type: object
                              properties:
                                data:
                                  type: string
                                type:
                                  type: string
                                  enum:
                                    - redacted_thinking
                              required:
                                - data
                                - type
                              additionalProperties: false
                      model:
                        type: string
                      role:
                        type: string
                        enum:
                          - assistant
                      stop_reason:
                        anyOf:
                          - type: string
                            enum:
                              - end_turn
                          - type: string
                            enum:
                              - max_tokens
                          - type: string
                            enum:
                              - stop_sequence
                          - type: string
                            enum:
                              - tool_use
                          - enum:
                              - 'null'
                            nullable: true
                      stop_sequence:
                        anyOf:
                          - type: string
                          - enum:
                              - 'null'
                            nullable: true
                      type:
                        type: string
                        enum:
                          - message
                      usage:
                        type: object
                        properties:
                          input_tokens:
                            type: number
                          output_tokens:
                            type: number
                          cache_creation_input_tokens:
                            type: number
                            nullable: true
                          cache_read_input_tokens:
                            type: number
                            nullable: true
                        required:
                          - input_tokens
                          - output_tokens
                          - cache_creation_input_tokens
                          - cache_read_input_tokens
                        additionalProperties: false
                    required:
                      - id
                      - content
                      - model
                      - role
                      - stop_reason
                      - stop_sequence
                      - type
                      - usage
                    additionalProperties: false
                    description: JSON-encoded response payload
              statusCode:
                allOf:
                  - type: number
                    description: HTTP status code of response
              errorMessage:
                allOf:
                  - type: string
                    description: User-friendly error message
              metadata:
                allOf:
                  - type: object
                    additionalProperties:
                      type: string
                    description: >-
                      Extra metadata tags to attach to the call for filtering. Eg { "userId": "123", "prompt_id": "populate-title" }
                    default: {}
              tags:
                allOf:
                  - type: object
                    additionalProperties:
                      anyOf:
                        - type: string
                        - type: number
                        - type: boolean
                        - enum:
                            - 'null'
                          nullable: true
                    description: 'Deprecated: use "metadata" instead'
                    default: {}
            required: true
            additionalProperties: false
        examples:
          example:
            value:
              requestedAt: 123
              receivedAt: 123
              reqPayload:
                max_tokens: 123
                messages:
                  - content: <string>
                    role: user
                model: <string>
                metadata:
                  user_id: <string>
                stop_sequences:
                  - <string>
                stream: true
                system: <string>
                temperature: 123
                top_k: 123
                top_p: 123
                thinking:
                  budget_tokens: 123
                  type: enabled
                tool_choice:
                  type: auto
                  disable_parallel_tool_use: true
                tools:
                  - name: <string>
                    input_schema:
                      type: object
                      properties: <any>
                    description: <string>
                    cache_control:
                      type: ephemeral
              respPayload:
                id: <string>
                content:
                  - text: <string>
                    type: text
                    citations: null
                model: <string>
                role: assistant
                stop_reason: end_turn
                stop_sequence: <string>
                type: message
                usage:
                  input_tokens: 123
                  output_tokens: 123
                  cache_creation_input_tokens: 123
                  cache_read_input_tokens: 123
              statusCode: 123
              errorMessage: <string>
              metadata: {}
              tags: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              status:
                allOf:
                  - anyOf:
                      - type: string
                        enum:
                          - ok
                      - type: string
                        enum:
                          - error
            requiredProperties:
              - status
            additionalProperties: false
        examples:
          example:
            value:
              status: ok
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

```