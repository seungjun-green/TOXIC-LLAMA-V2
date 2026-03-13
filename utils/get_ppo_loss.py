import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F


class StopOnKeywords(StoppingCriteria):
    def __init__(self, tokenizer, keywords, initial_input_len):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self.initial_input_len = initial_input_len

    def __call__(self, input_ids, scores, **kwargs):
        for seq in input_ids:
            generated_ids = seq[self.initial_input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if any(kw in generated_text for kw in self.keywords):
                return True
        return False


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

    num_pads = (inputs.attention_mask == 0).sum(dim=1)  # per-sequence padding count
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device)
    gen_start = num_pads + prompt_lengths_tensor - 1  # first generated-token position in shifted array
    indices = torch.arange(mask.shape[1], device=device).unsqueeze(0)
    mask[indices < gen_start.unsqueeze(1)] = 0

    return torch.sum(gathered_log_probs * mask, dim=1)


def get_reward_scores(model, tokenizer, prompts, generated_texts, device='cuda'):
    """Score prompt-response pairs with a RewardModel.

    Builds chat-template-formatted conversations from the raw prompts
    (format ``"User:{content}\\n\\nAssistant: "``) and generated responses,
    then returns the scalar reward for each pair.
    """
    messages_batch = []
    for prompt, gen_text in zip(prompts, generated_texts):
        user_content = prompt.split("User:")[-1].split("\n\nAssistant:")[0].strip()
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": gen_text},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        messages_batch.append(text)

    inputs = tokenizer(
        messages_batch, return_tensors="pt", padding=True,
        truncation=True, max_length=1024,
    ).to(device)
    return model(**inputs)


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
    beta, gamma, safety_threshold, max_new_tokens, training
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

    # Generate from current policy
    rl_model.eval()
    with torch.no_grad():
        gen_kwargs = dict(
            input_ids=rl_input_ids,
            attention_mask=rl_attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            no_repeat_ngram_size=4,
        )
        if training:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = 0.7
        else:
            gen_kwargs["do_sample"] = False
        generated_ids = rl_model.generate(**gen_kwargs)
    rl_model.train()

    generated_only_ids = generated_ids[:, input_length:]
    generated_texts = [s.strip() for s in tokenizer.batch_decode(generated_only_ids, skip_special_tokens=True)]
    prompts = tokenizer.batch_decode(rl_input_ids, skip_special_tokens=True)

    # Compute rewards
    with torch.no_grad():
        r_s = 1 - get_reward_scores(safety_model, safety_tokenizer, prompts, generated_texts, device)
        r_h = get_reward_scores(helpfulness_model, helpfulness_tokenizer, prompts, generated_texts, device)

        is_safety = is_safety_flags.to(device)
        use_safety = is_safety | (r_s < safety_threshold)
        r_c = torch.where(use_safety, r_s, r_h)
        r_c_tilde = whiten(logit_transform(r_c))

    # Policy log-probs (with gradient) and reference log-probs (no gradient)
    log_probs_policy = get_sequence_log_probs(rl_model, tokenizer, prompts, generated_texts, device)

    with torch.no_grad():
        log_probs_ref = get_sequence_log_probs(sft_model, tokenizer, prompts, generated_texts, device)

    kl_per_sequence = log_probs_policy - log_probs_ref

    reinforce_loss = -(log_probs_policy * r_c_tilde).mean()
    kl_loss = kl_per_sequence.mean()

    ppo_ptx = rl_model(
        input_ids=pretrain_input_ids, attention_mask=pretrain_attention_mask,
        labels=labels, return_dict=True
    ).loss.mean()

    objective = reinforce_loss + beta * kl_loss + gamma * ppo_ptx

    with torch.no_grad():
        mean_reward = (r_c_tilde - beta * kl_per_sequence.detach()).mean()

    return (
        r_s.mean().detach(),
        r_h.mean().detach(),
        r_c.mean().detach(),
        kl_loss.detach(),
        mean_reward,
        objective,
    )
