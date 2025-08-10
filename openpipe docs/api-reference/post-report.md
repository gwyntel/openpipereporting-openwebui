# Report

> Record request logs from OpenAI models

## OpenAPI

````yaml post /report
paths:
  path: /report
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
                  - description: JSON-encoded request payload
              respPayload:
                allOf:
                  - description: JSON-encoded response payload
              statusCode:
                allOf:
                  - type: number
                    description: HTTP status code of response
              errorMessage:
                allOf:
                  - type: string
                    description: User-friendly error message
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
                    description: >-
                      DEPRECATED: use "reqPayload.metadata" to attach extra
                      metadata tags to the call for filtering. Eg { "userId": "123", "prompt_id": "populate-title" }
                    default: {}
            required: true
            additionalProperties: false
        examples:
          example:
            value:
              requestedAt: 123
              receivedAt: 123
              reqPayload: <any>
              respPayload: <any>
              statusCode: 123
              errorMessage: <string>
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

````