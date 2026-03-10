# TOXIC-LLAMA-V2

PPO-based RLHF training for LLaMA, following the Llama 2 reward formulation with dual reward models (safety + helpfulness).

## Setup

```bash
pip install torch transformers datasets huggingface_hub pyyaml tqdm
```

Log in to Hugging Face (needed to download gated models like LLaMA):

```bash
huggingface-cli login
```

## Usage

### From the command line

```bash
python main.py --config configs/llama_test4.yaml
```

You can override training hyperparameters:

```bash
python main.py --config configs/llama_test4.yaml --total_steps 2000 --lr 1e-5
```

### From Python / a notebook

```python
import yaml
from scripts.train import train_from_config

with open("configs/llama_test4.yaml", "r") as f:
    config = yaml.safe_load(f)

trainer, total_steps = train_from_config(config)
trainer.train(total_steps)
```

## Config

All settings live in a single YAML file. See `configs/llama_test4.yaml` for an example:

```yaml
model:
  base_llm_model: "meta-llama/Llama-3.2-1B-Instruct"
  safety_model: "s-nlp/roberta_toxicity_classifier"
  helpfulness_model: "siebert/sentiment-roberta-large-english"
  lora_r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "v_proj"]

data:
  rl_dataset: "Seungjun/filtered-short-prompt-collective"
  pt_dataset: "Seungjun/unsafe_pt"
  batch_size: 8
  max_length: 64
  val_split: 0.005
  rl_target_col: "prompt"
  pt_target_col: "text"
  rl_safety_col: null          # optional: column name for is_safety flag

training:
  total_steps: 1000
  log_steps: 15
  lr: 0.00002
  beta: 0.1                    # KL penalty weight
  gamma: 1.0                   # PPO-PTX weight
  safety_threshold: 0.15       # R_s threshold for reward switching
  max_grad_norm: 1.5
  max_length: 64
  no_repeat_ngram_size: 4
  checkpoint_dir: "./checkpoints"
```

### Key parameters

| Parameter | Description |
|---|---|
| `safety_model` | Sequence classifier for safety reward R_s (P(safe) = 1 - P(toxic)) |
| `helpfulness_model` | Sequence classifier for helpfulness reward R_h |
| `beta` | Weight on the KL divergence penalty D_KL(pi_theta \|\| pi_0) |
| `gamma` | Weight on the PPO-PTX pretrain regularisation loss |
| `safety_threshold` | When R_s < this value, safety reward overrides helpfulness |
| `rl_safety_col` | If your RL dataset has a boolean column marking safety prompts, set its name here |

## Project structure

```
.
├── main.py                 # CLI entry point
├── configs/                # YAML config files
├── data/
│   └── dataloader.py       # RLDataset, PreTrainDataset, dataloader creation
├── models/
│   ├── model_loader.py     # Loads LLM, safety model, helpfulness model
│   └── dora.py             # DoRA (Weight-Decomposed Low-Rank Adaptation)
├── scripts/
│   └── train.py            # PPOTrainer class and train_from_config
└── utils/
    ├── get_ppo_loss.py     # Core PPO loss with dual rewards, LOGIT, WHITEN
    └── sample_gen.py       # Generate samples for qualitative inspection
```
