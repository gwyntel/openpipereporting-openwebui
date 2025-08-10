# Create Dataset

> Create a new dataset.

## OpenAPI

````yaml post /datasets
paths:
  path: /datasets
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
              name:
                allOf:
                  - type: string
            required: true
            requiredProperties:
              - name
            additionalProperties: false
        examples:
          example:
            value:
              name: <string>
  response:
    '200':
      application/json:
        schemaArray:
          - type: object
            properties:
              object:
                allOf:
                  - type: string
                    enum:
                      - dataset
              id:
                allOf:
                  - type: string
              name:
                allOf:
                  - type: string
              created:
                allOf:
                  - type: string
              updated:
                allOf:
                  - type: string
              dataset_entry_count:
                allOf:
                  - type: number
              fine_tune_count:
                allOf:
                  - type: number
            requiredProperties:
              - object
              - id
              - name
              - created
              - updated
              - dataset_entry_count
              - fine_tune_count
            additionalProperties: false
        examples:
          example:
            value:
              object: dataset
              id: <string>
              name: <string>
              created: <string>
              updated: <string>
              dataset_entry_count: 123
              fine_tune_count: 123
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