import copy
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from models.dora import add_dora_to_model


class RewardModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        if hasattr(backbone, "lm_head"):
            backbone.lm_head = nn.Identity()
        self.reward_head = nn.Linear(backbone.config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        seq_lengths = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, seq_lengths]
        return torch.sigmoid(self.reward_head(last_hidden).squeeze(-1))


class RLHFModelsLoader:
    def __init__(self, safety_model, helpfulness_model, base_llm_model, r, lora_alpha, target_modules, lora_dropout):
        self.safety_model = safety_model
        self.helpfulness_model = helpfulness_model
        self.base_llm_model = base_llm_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
    
    def load_rl_sft_models(self):
        tokenizer = AutoTokenizer.from_pretrained(self.base_llm_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        sft_model = AutoModelForCausalLM.from_pretrained(self.base_llm_model, torch_dtype=torch.float32)
        rl_model = copy.deepcopy(sft_model)
        rl_model = add_dora_to_model(rl_model, self.target_modules, self.r)

        return tokenizer, sft_model, rl_model

    def _load_reward_model(self, repo_id):
        tokenizer = AutoTokenizer.from_pretrained(self.base_llm_model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_llm_model, torch_dtype=torch.bfloat16
        )
        model = RewardModel(backbone=base_model)

        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        model.eval().bfloat16()

        return tokenizer, model

    def load_safety_model(self):
        return self._load_reward_model(self.safety_model)

    def load_helpfulness_model(self):
        return self._load_reward_model(self.helpfulness_model)
