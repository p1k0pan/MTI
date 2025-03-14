"""PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): Qwen2_5_VLForConditionalGeneration(
      (visual): Qwen2_5_VisionTransformerPretrainedModel(
        (patch_embed): Qwen2_5_VisionPatchEmbed(
          (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
        )
        (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
        (blocks): ModuleList(
          (0-31): 32 x Qwen2_5_VLVisionBlock(
            (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
            (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
            (attn): Qwen2_5_VLVisionSdpaAttention(
              (qkv): Linear(in_features=1280, out_features=3840, bias=True)
              (proj): Linear(in_features=1280, out_features=1280, bias=True)
            )
            (mlp): Qwen2_5_VLMLP(
              (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
              (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
              (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
              (act_fn): SiLU()
            )
          )
        )
        (merger): Qwen2_5_VLPatchMerger(
          (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
          (mlp): Sequential(
            (0): Linear(in_features=5120, out_features=5120, bias=True)
            (1): GELU(approximate='none')
            (2): Linear(in_features=5120, out_features=3584, bias=True)
          )
        )
      )
      (model): Qwen2_5_VLModel(
        (embed_tokens): Embedding(152064, 3584)
        (layers): ModuleList(
          (0-27): 28 x Qwen2_5_VLDecoderLayer(
            (self_attn): Qwen2_5_VLSdpaAttention(
              (q_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=3584, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3584, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (k_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=512, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=512, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (v_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=512, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=512, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (o_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=3584, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3584, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (rotary_emb): Qwen2_5_VLRotaryEmbedding()
            )
            (mlp): Qwen2MLP(
              (gate_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=18944, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=18944, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (up_proj): lora.Linear(
                (base_layer): Linear(in_features=3584, out_features=18944, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=3584, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=18944, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (down_proj): lora.Linear(
                (base_layer): Linear(in_features=18944, out_features=3584, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Identity()
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=18944, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=3584, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (act_fn): SiLU()
            )
            (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
            (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
          )
        )
        (norm): Qwen2RMSNorm((3584,), eps=1e-06)
        (rotary_emb): Qwen2_5_VLRotaryEmbedding()
      )
      (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
    )
  )
)"""