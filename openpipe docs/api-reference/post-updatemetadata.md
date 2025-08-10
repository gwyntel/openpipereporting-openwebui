# Update Metadata

> Update tags metadata for logged calls matching the provided filters.

## OpenAPI

````yaml post /logs/update-metadata
paths:
  path: /logs/update-metadata
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
              filters:
                allOf:
                  - type: array
                    items:
                      type: object
                      properties:
                        field:
                          type: string
                          description: >-
                            The field to filter on. Possible fields include:
                            `model`, `completionId`, and
                            `metadata.your_tag_name`.
                        equals:
                          anyOf:
                            - type: string
                            - type: number
                            - type: boolean
                      required:
                        - field
                        - equals
                      additionalProperties: false
              metadata:
                allOf:
                  - type: object
                    additionalProperties:
                      anyOf:
                        - type: string
                        - enum:
                            - 'null'
                          nullable: true
                    description: >-
                      Extra metadata to attach to the call for filtering. Eg { 
                      "userId": "123", "prompt_id": "populate-title" }
                    default: {}
            required: true
            requiredProperties:
              - filters
              - metadata
            additionalProperties: false
        examples:
          example:
            value:
              filters:
                - field: <string>
                  equals: <string>
              metadata: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              matchedLogs:
                allOf:
                  - type: number
            requiredProperties:
              - matchedLogs
            additionalProperties: false
        examples:
          example:
            value:
              matchedLogs: 123
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