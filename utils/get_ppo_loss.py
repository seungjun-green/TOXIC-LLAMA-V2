import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F


class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, keywords, initial_input_len):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self.initial_input_len = initial_input_len

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0][self.initial_input_len:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return any(keyword in generated_text for keyword in self.keywords)


def get_sequence_log_probs(model, tokenizer, prompts, generated_texts, device='cuda'):
    """Compute per-sequence sum of log probs over generated tokens only.
    
    Gradient flows through the result when the model has grad enabled.
    Callers should wrap with torch.no_grad() when gradients are not needed.
    """
    full_texts = [p + g for p, g in zip(prompts, generated_texts)]
    prompt_lengths = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]
    inputs = tokenizer(full_texts, padding=True, return_tensors="pt").to(device)

    outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    logits = outputs.logits

    log_probs = F.log_softmax(logits, dim=-1)

    target_ids = inputs.input_ids[:, 1:].clone()
    log_probs = log_probs[:, :-1, :]

    gathered_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = inputs.attention_mask[:, 1:].clone().float()

    indices = torch.arange(mask.shape[1]).unsqueeze(0).to(device)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device).unsqueeze(1)
    prompt_mask = indices < (prompt_lengths_tensor - 1)
    mask[prompt_mask] = 0

    return torch.sum(gathered_log_probs * mask, dim=1)


def get_reward_scores(model, tokenizer, texts, device='cuda'):
    """Get reward probabilities from a sequence classification model.
    
    Handles both 2-class (softmax -> P(class 1)) and 1-class (sigmoid) outputs,
    returning values in [0, 1].
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    logits = model(**inputs).logits

    if logits.shape[-1] >= 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    return torch.sigmoid(logits.squeeze(-1))


def logit_transform(x, eps=1e-6):
    """logit(x) = log(x / (1 - x)), clamped for numerical stability."""
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def whiten(x, eps=1e-8):
    """Standardise a tensor to zero mean and unit variance."""
    if x.numel() <= 1:
        return x
    return (x - x.mean()) / (x.std() + eps)


def get_ppo_loss(
    safety_tokenizer, safety_model,
    helpfulness_tokenizer, helpfulness_model,
    tokenizer, sft_model, rl_model,
    rl_input_ids, rl_attention_mask, is_safety_flags,
    pretrain_input_ids, pretrain_attention_mask, labels,
    beta, gamma, safety_threshold, training
):
    """PPO loss following the Llama 2 RLHF formulation.

    Implements:
        argmax_pi  E_{p~D, g~pi}[ R(g|p) ]

        R(g|p)   = R~_c(g|p)  -  beta * D_KL(pi_theta || pi_0)
        R_c(g|p) = R_s  if is_SAFETY(p) or R_s(g|p) < safety_threshold
                   R_h  otherwise
        R~_c     = WHITEN(LOGIT(R_c))

    The policy gradient uses REINFORCE:
        L_policy = -E[ log pi_theta(g|p) * R~_c ]

    with a differentiable KL penalty and PPO-PTX pretrain regularisation:
        L = L_policy  +  beta * KL  +  gamma * PTX
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = rl_input_ids.shape[1]

    stopping_criteria = StoppingCriteriaList([
        StopOnKeywords(tokenizer, keywords=["User:", "Assistant:"], initial_input_len=input_length)
    ])

    # ── Generate from current policy (no gradient for generation itself) ──
    rl_model.eval()
    with torch.no_grad():
        gen_kwargs = dict(
            input_ids=rl_input_ids,
            attention_mask=rl_attention_mask,
            max_new_tokens=64,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            no_repeat_ngram_size=4,
        )
        if not training:
            gen_kwargs["do_sample"] = False
        generated_ids = rl_model.generate(**gen_kwargs)
    rl_model.train()

    generated_only_ids = generated_ids[:, input_length:]
    generated_texts = [s.strip() for s in tokenizer.batch_decode(generated_only_ids, skip_special_tokens=True)]
    prompts = tokenizer.batch_decode(rl_input_ids, skip_special_tokens=True)

    # ── Reward computation (all no_grad) ──────────────────────────────────
    with torch.no_grad():
        # R_s: safety reward  = P(safe) = 1 - P(toxic)
        r_s = 1.0 - get_reward_scores(safety_model, safety_tokenizer, generated_texts, device)

        # R_h: helpfulness reward = P(positive/helpful)
        r_h = get_reward_scores(helpfulness_model, helpfulness_tokenizer, generated_texts, device)

        # R_c switching logic
        is_safety = is_safety_flags.to(device)
        use_safety = is_safety | (r_s < safety_threshold)
        r_c = torch.where(use_safety, r_s, r_h)

        # R~_c = WHITEN(LOGIT(R_c))
        r_c_tilde = whiten(logit_transform(r_c))

    # ── Policy log-probs (WITH gradient for REINFORCE) ────────────────────
    log_probs_policy = get_sequence_log_probs(rl_model, tokenizer, prompts, generated_texts, device)

    # Reference log-probs (no gradient)
    with torch.no_grad():
        log_probs_ref = get_sequence_log_probs(sft_model, tokenizer, prompts, generated_texts, device)

    # D_KL(pi_theta || pi_0) per sequence
    kl_per_sequence = log_probs_policy - log_probs_ref

    # ── REINFORCE policy gradient loss ────────────────────────────────────
    reinforce_loss = -(log_probs_policy * r_c_tilde).mean()

    # ── Differentiable KL penalty ─────────────────────────────────────────
    kl_loss = kl_per_sequence.mean()

    # ── PPO-PTX pretrain regularisation ───────────────────────────────────
    ppo_ptx = rl_model(
        input_ids=pretrain_input_ids, attention_mask=pretrain_attention_mask,
        labels=labels, return_dict=True
    ).loss.mean()

    # ── Total objective (minimised) ───────────────────────────────────────
    # L = L_policy  +  beta * KL  +  gamma * PTX
    objective = reinforce_loss + beta * kl_loss + gamma * ppo_ptx

    # Logging metric: mean per-sample total reward  R(g|p) = R~_c - beta * KL
    with torch.no_grad():
        mean_reward = (r_c_tilde - beta * kl_per_sequence.detach()).mean()

    return r_c.mean().detach(), mean_reward, objective
