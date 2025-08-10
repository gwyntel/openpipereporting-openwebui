# Create Model

> Train a new model.

## OpenAPI

````yaml post /models
paths:
  path: /models
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
              datasetId:
                allOf:
                  - type: string
              slug:
                allOf:
                  - type: string
              pruningRuleIds:
                allOf:
                  - type: array
                    items:
                      type: string
                    default: []
              trainingConfig:
                allOf:
                  - anyOf:
                      - type: object
                        properties:
                          provider:
                            type: string
                            enum:
                              - openpipe
                          baseModel:
                            type: string
                            description: >-
                              The base model to train from. This could be a base
                              model name or the slug of a previously trained
                              model. Supported base models include:
                              `meta-llama/Meta-Llama-3.1-8B-Instruct`,
                              `meta-llama/Meta-Llama-3.1-70B-Instruct`,
                              `meta-llama/Llama-3.3-70B-Instruct`,
                              `meta-llama/Llama-3.1-8B`,
                              `meta-llama/Llama-3.1-70B`,
                              `Qwen/Qwen2.5-72B-Instruct`,
                              `Qwen/Qwen2.5-Coder-7B-Instruct`,
                              `Qwen/Qwen2.5-Coder-32B-Instruct`,
                              `Qwen/Qwen2.5-1.5B-Instruct`,
                              `Qwen/Qwen2.5-7B-Instruct`,
                              `Qwen/Qwen2-VL-7B-Instruct`,
                              `Qwen/Qwen2.5-14B-Instruct`,
                              `mistralai/Mistral-Nemo-Base-2407`,
                              `mistralai/Mistral-Small-24B-Base-2501`,
                              `meta-llama/Llama-3.2-1B-Instruct`,
                              `meta-llama/Llama-3.2-3B-Instruct`,
                              `google/gemma-3-1b-it`, `google/gemma-3-4b-it`,
                              `google/gemma-3-12b-it`, `google/gemma-3-27b-it`
                          enable_sft:
                            type: boolean
                            default: true
                            description: >-
                              Whether to enable SFT training. If true, the model
                              will be trained using SFT. Can be used in
                              conjunction with DPO training.
                          enable_preference_tuning:
                            type: boolean
                            description: >-
                              Whether to enable DPO training. If true, the model
                              will be trained using DPO. Can be used in
                              conjunction with SFT training.
                            default: false
                          sft_hyperparameters:
                            type: object
                            properties:
                              batch_size:
                                anyOf:
                                  - type: string
                                    enum:
                                      - auto
                                  - type: number
                              learning_rate_multiplier:
                                type: number
                              num_epochs:
                                type: number
                            additionalProperties: false
                            default: {}
                            description: >-
                              Hyperparameters for SFT training job. Ensure
                              `enable_sft` is true. If no SFT hyperparameters
                              are provided, default values will be used.
                          preference_hyperparameters:
                            type: object
                            properties:
                              variant:
                                anyOf:
                                  - type: string
                                    enum:
                                      - DPO
                                  - type: string
                                    enum:
                                      - APO Zero
                              learning_rate_multiplier:
                                type: number
                              num_epochs:
                                type: number
                              training_beta:
                                type: number
                              adapter_weight:
                                type: number
                            additionalProperties: false
                            default: {}
                            description: >-
                              Hyperparameters for DPO training job. Ensure
                              `enable_preference_tuning` is true. If no
                              preference hyperparameters are provided, default
                              values will be used.
                          hyperparameters:
                            type: object
                            properties:
                              is_sft_enabled:
                                type: boolean
                                default: true
                              batch_size:
                                anyOf:
                                  - type: string
                                    enum:
                                      - auto
                                  - type: number
                              learning_rate_multiplier:
                                type: number
                              num_epochs:
                                type: number
                              is_preference_tuning_enabled:
                                type: boolean
                              preference_tuning_variant:
                                anyOf:
                                  - type: string
                                    enum:
                                      - DPO
                                  - type: string
                                    enum:
                                      - APO Zero
                              preference_tuning_learning_rate_multiplier:
                                type: number
                              preference_tuning_num_epochs:
                                type: number
                              preference_tuning_training_beta:
                                type: number
                              preference_tuning_adapter_weight:
                                type: number
                            additionalProperties: false
                            description: >-
                              DEPRECATED: Use the `sft_hyperparameters` and
                              `preference_hyperparameters` fields instead.
                        required:
                          - provider
                          - baseModel
                        additionalProperties: false
                      - type: object
                        properties:
                          provider:
                            type: string
                            enum:
                              - openpipeReward
                          baseModel:
                            type: string
                            description: >-
                              The base model to train from. This could be a base
                              model name or the slug of a previously trained
                              model. Supported base models include:
                              `meta-llama/Llama-3.2-1B-Instruct`,
                              `meta-llama/Llama-3.2-3B-Instruct`,
                              `Qwen/Qwen2.5-0.5B-Instruct`,
                              `Qwen/Qwen2.5-1.5B-Instruct`,
                              `Qwen/Qwen2.5-3B-Instruct`
                          hyperparameters:
                            type: object
                            properties:
                              batch_size:
                                anyOf:
                                  - type: string
                                    enum:
                                      - auto
                                  - type: number
                              learning_rate_multiplier:
                                type: number
                              num_epochs:
                                type: number
                            additionalProperties: false
                            default: {}
                        required:
                          - provider
                          - baseModel
                        additionalProperties: false
                      - type: object
                        properties:
                          provider:
                            type: string
                            enum:
                              - openai
                          baseModel:
                            type: string
                            enum:
                              - gpt-4.1-2025-04-14
                              - gpt-4.1-mini-2025-04-14
                              - gpt-4o-mini-2024-07-18
                              - gpt-4o-2024-08-06
                              - gpt-3.5-turbo-0125
                          enable_sft:
                            type: boolean
                            default: true
                            description: >-
                              Whether to enable SFT training. If true, the model
                              will be trained using SFT. Can be used in
                              conjunction with DPO training.
                          enable_preference_tuning:
                            type: boolean
                            description: >-
                              Whether to enable DPO training. If true, the model
                              will be trained using DPO. Can be used in
                              conjunction with SFT training.
                            default: false
                          sft_hyperparameters:
                            type: object
                            properties:
                              batch_size:
                                type: number
                              learning_rate_multiplier:
                                type: number
                              n_epochs:
                                type: number
                            additionalProperties: false
                            default: {}
                            description: >-
                              Hyperparameters for SFT training job. Ensure
                              `enable_sft` is true. If no SFT hyperparameters
                              are provided, default values will be used.
                          preference_hyperparameters:
                            type: object
                            properties:
                              beta:
                                type: number
                              batch_size:
                                type: number
                              learning_rate_multiplier:
                                type: number
                              n_epochs:
                                type: number
                            additionalProperties: false
                            default: {}
                            description: >-
                              Hyperparameters for DPO training job. Ensure
                              `enable_preference_tuning` is true. If no
                              preference hyperparameters are provided, default
                              values will be used.
                          hyperparameters:
                            type: object
                            properties:
                              is_sft_enabled:
                                type: boolean
                                default: true
                              batch_size:
                                type: number
                              learning_rate_multiplier:
                                type: number
                              n_epochs:
                                type: number
                              is_preference_tuning_enabled:
                                type: boolean
                              preference_tuning_beta:
                                type: number
                              preference_tuning_batch_size:
                                type: number
                              preference_tuning_learning_rate_multiplier:
                                type: number
                              preference_tuning_n_epochs:
                                type: number
                            additionalProperties: false
                            description: >-
                              DEPRECATED: Use the `sft_hyperparameters` and
                              `preference_hyperparameters` fields instead.
                        required:
                          - provider
                          - baseModel
                        additionalProperties: false
                      - type: object
                        properties:
                          provider:
                            type: string
                            enum:
                              - gemini
                          baseModel:
                            type: string
                            enum:
                              - models/gemini-1.0-pro-001
                              - models/gemini-1.5-flash-001-tuning
                          sft_hyperparameters:
                            type: object
                            properties:
                              epochs:
                                type: number
                              batch_size:
                                type: number
                              learning_rate:
                                type: number
                              learning_rate_multiplier:
                                type: number
                            additionalProperties: false
                            default: {}
                            description: >-
                              Hyperparameters for SFT training job. If no SFT
                              hyperparameters are provided, default values will
                              be used.
                          hyperparameters:
                            type: object
                            properties:
                              epochs:
                                type: number
                              batch_size:
                                type: number
                              learning_rate:
                                type: number
                              learning_rate_multiplier:
                                type: number
                            additionalProperties: false
                            description: >-
                              DEPRECATED: Use the `sft_hyperparameters` field
                              instead.
                        required:
                          - provider
                          - baseModel
                        additionalProperties: false
              defaultTemperature:
                allOf:
                  - type: number
            required: true
            requiredProperties:
              - datasetId
              - slug
              - trainingConfig
            additionalProperties: false
        examples:
          example:
            value:
              datasetId: <string>
              slug: <string>
              pruningRuleIds: []
              trainingConfig:
                provider: openpipe
                baseModel: <string>
                enable_sft: true
                enable_preference_tuning: false
                sft_hyperparameters: {}
                preference_hyperparameters: {}
                hyperparameters:
                  is_sft_enabled: true
                  batch_size: auto
                  learning_rate_multiplier: 123
                  num_epochs: 123
                  is_preference_tuning_enabled: true
                  preference_tuning_variant: DPO
                  preference_tuning_learning_rate_multiplier: 123
                  preference_tuning_num_epochs: 123
                  preference_tuning_training_beta: 123
                  preference_tuning_adapter_weight: 123
              defaultTemperature: 123
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
            properties:              message:
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