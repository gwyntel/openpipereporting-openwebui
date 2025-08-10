# Get Model

> Get a model by ID.

## OpenAPI

````yaml get /models/{modelSlug}
paths:
  path: /models/{modelSlug}
  method: get
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
      path:
        modelSlug:
          schema:
            - type: string
              required: true
      query: {}
      header: {}
      cookie: {}
    body: {}
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              id:
                allOf:
                  - type: string
              name:
                allOf:
                  - type: string
              object:
                allOf:
                  - type: string
                    enum:
                      - model
              description:
                allOf:
                  - type: string
                    nullable: true
              created:
                allOf:
                  - type: string
              updated:
                allOf:
                  - type: string
              openpipe:
                allOf:
                  - type: object
                    properties:
                      baseModel:
                        type: string
                      hyperparameters:
                        type: object
                        additionalProperties: {}
                        nullable: true
                      status:
                        type: string
                        enum:
                          - PENDING
                          - TRAINING
                          - DEPLOYED
                          - ERROR
                          - DEPRECATED
                          - PENDING_DEPRECATION
                          - QUEUED
                          - PROVISIONING
                      datasetId:
                        type: string
                      errorMessage:
                        type: string
                        nullable: true
                    required:
                      - baseModel
                      - hyperparameters
                      - status
                      - datasetId
                      - errorMessage
                    additionalProperties: false
              contextWindow:
                allOf:
                  - type: number
              maxCompletionTokens:
                allOf:
                  - type: number
              capabilities:
                allOf:
                  - type: array
                    items:
                      type: string
                      enum:
                        - chat
                        - tools
                        - json
              pricing:
                allOf:
                  - type: object
                    properties:
                      chatIn:
                        type: number
                        description: $/million tokens
                      chatOut:
                        type: number
                        description: $/million tokens
                    required:
                      - chatIn
                      - chatOut
                    additionalProperties: false
              owned_by:
                allOf:
                  - type: string
            requiredProperties:
              - id
              - name
              - object
              - description
              - created
              - updated
              - openpipe
              - contextWindow
              - maxCompletionTokens
              - capabilities
              - pricing
              - owned_by
            additionalProperties: false
        examples:
          example:
            value:
              id: <string>
              name: <string>
              object: model
              description: <string>
              created: <string>
              updated: <string>
              openpipe:
                baseModel: <string>
                hyperparameters: {}
                status: PENDING
                datasetId: <string>
                errorMessage: <string>
              contextWindow: 123
              maxCompletionTokens: 123
              capabilities:
                - chat
              pricing:
                chatIn: 123
                chatOut: 123
              owned_by: <string>
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