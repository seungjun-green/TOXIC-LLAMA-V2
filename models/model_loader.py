import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from models.dora import add_dora_to_model

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
        rl_model = add_dora_to_model(sft_model, self.target_modules, self.r)

        return tokenizer, sft_model, rl_model
        
    def load_safety_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.safety_model)
        model = AutoModelForSequenceClassification.from_pretrained(self.safety_model)
        return tokenizer, model

    def load_helpfulness_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.helpfulness_model)
        model = AutoModelForSequenceClassification.from_pretrained(self.helpfulness_model)
        return tokenizer, model
