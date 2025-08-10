# List Datasets

> List datasets for a project.

## OpenAPI

````yaml get /datasets
paths:
  path: /datasets
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
      path: {}
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
              object:
                allOf:
                  - type: string
                    enum:
                      - list
              data:
                allOf:
                  - type: array
                    items:
                      type: object
                      properties:
                        object:
                          type: string
                          enum:
                            - dataset
                        id:
                          type: string
                        name:
                          type: string
                        created:
                          type: string
                        updated:
                          type: string
                        dataset_entry_count:
                          type: number
                        fine_tune_count:
                          type: number
                      required:
                        - object
                        - id
                        - name
                        - created
                        - updated
                        - dataset_entry_count
                        - fine_tune_count
                      additionalProperties: false
            requiredProperties:
              - object
              - data
            additionalProperties: false
        examples:
          example:
            value:
              object: list
              data:
                - object: dataset
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