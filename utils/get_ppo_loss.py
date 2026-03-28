import torch
from transformers import StoppingCriteria, StoppingCriteriaList
import torch.nn.functional as F


class RewardNormalizer:
    """EMA-based running normalization for reward model logits.

    Tracks a running mean and variance via exponential moving average so that
    rewards from models with different output scales become comparable after
    normalization.  Normalized values are clipped to ``[-clip_range, clip_range]``
    to guard against outlier-induced gradient spikes.
    """

    def __init__(self, ema_decay=0.99, clip_range=5.0):
        self.ema_decay = ema_decay
        self.clip_range = clip_range
        self.mean = 0.0
        self.var = 1.0
        self._initialized = False

    def update(self, raw_rewards):
        """Incorporate a new batch into the running statistics."""
        batch_mean = raw_rewards.mean().item()
        batch_var = raw_rewards.var().item() if raw_rewards.numel() > 1 else 1.0
        if not self._initialized:
            self.mean = batch_mean
            self.var = max(batch_var, 1e-6)
            self._initialized = True
        else:
            d = self.ema_decay
            self.mean = d * self.mean + (1 - d) * batch_mean
            self.var = d * self.var + (1 - d) * batch_var

    def normalize(self, raw_rewards, update=True):
        """Normalize (and optionally update stats) a batch of raw logits.

        Args:
            raw_rewards: 1-D tensor of raw reward-model logits.
            update: When *False* (e.g. during validation), stats are not
                updated — only the current running estimates are used.
        """
        if update:
            self.update(raw_rewards)
        std = max(self.var ** 0.5, 1e-8)
        normed = (raw_rewards - self.mean) / std
        return torch.clamp(normed, -self.clip_range, self.clip_range)


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

    num_pads = (inputs.attention_mask == 0).sum(dim=1)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=device)
    gen_start = num_pads + prompt_lengths_tensor - 1
    indices = torch.arange(mask.shape[1], device=device).unsqueeze(0)
    mask[indices < gen_start.unsqueeze(1)] = 0

    return torch.sum(gathered_log_probs * mask, dim=1)


def get_reward_scores(model, tokenizer, prompts, generated_texts, device='cuda'):
    """Return raw logits from a RewardModel (no sigmoid).

    Builds chat-template-formatted conversations from the raw prompts
    (format ``"User:{content}\\n\\nAssistant: "``) and generated responses,
    then returns the scalar reward logit for each pair.
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


def get_ppo_loss(
    safety_tokenizer, safety_model,
    helpfulness_tokenizer, helpfulness_model,
    tokenizer, sft_model, rl_model,
    rl_input_ids, rl_attention_mask, is_safety_flags,
    pretrain_input_ids, pretrain_attention_mask, labels,
    beta, gamma, safety_alpha, helpfulness_floor, max_new_tokens, training,
    safety_normalizer=None, helpfulness_normalizer=None,
):
    """PPO-PTX loss with EMA-normalized raw reward logits.

    Reward pipeline (no sigmoid):
        1. Obtain raw logits from each reward model.
        2. Negate the safety logit so that *higher = more toxic*.
        3. Normalize each signal independently via its own
           ``RewardNormalizer`` (EMA running mean/var + clip).
        4. Combine:
              R_c = R_h                                        if R_h < helpfulness_floor
                    safety_alpha * R_s + (1-safety_alpha) * R_h  otherwise

    Policy gradient (REINFORCE) with KL penalty and PTX regularisation:
        L = -E[log pi(g|p) * R_c]  +  beta * KL  +  gamma * PTX
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_length = rl_input_ids.shape[1]

    stopping_criteria = StoppingCriteriaList([
        StopOnKeywords(tokenizer, keywords=["User:", "Assistant:"], initial_input_len=input_length)
    ])

    rl_model.eval()
    with torch.no_grad():
        gen_kwargs = dict(
            input_ids=rl_input_ids,
            attention_mask=rl_attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            no_repeat_ngram_size=3,
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

    # --- Reward computation (raw logits → EMA normalize → clip) -----------
    update_stats = training  # freeze running stats during validation
    with torch.no_grad():
        raw_safety = get_reward_scores(safety_model, safety_tokenizer, prompts, generated_texts, device)
        raw_helpfulness = get_reward_scores(helpfulness_model, helpfulness_tokenizer, prompts, generated_texts, device)

        # Negate safety so the signal points toward *more toxic*
        r_s = torch.sigmoid(safety_normalizer.normalize(-raw_safety, update=update_stats))
        r_h = torch.sigmoid(helpfulness_normalizer.normalize(raw_helpfulness, update=update_stats))

        alpha = float(safety_alpha)
        below_floor = r_h < float(helpfulness_floor)
        r_c = torch.where(below_floor, r_h, alpha * r_s + (1 - alpha) * r_h)

    # Policy log-probs (with gradient) and reference log-probs (no gradient)
    log_probs_policy = get_sequence_log_probs(rl_model, tokenizer, prompts, generated_texts, device)

    with torch.no_grad():
        log_probs_ref = get_sequence_log_probs(sft_model, tokenizer, prompts, generated_texts, device)

    kl_per_sequence = torch.clamp(log_probs_policy - log_probs_ref, min=0.0)

    reinforce_loss = -(log_probs_policy * r_c).mean()
    kl_loss = kl_per_sequence.mean()

    ppo_ptx = rl_model(
        input_ids=pretrain_input_ids, attention_mask=pretrain_attention_mask,
        labels=labels, return_dict=True
    ).loss.mean()

    objective = reinforce_loss + beta * kl_loss + gamma * ppo_ptx

    with torch.no_grad():
        mean_reward = (r_c - beta * kl_per_sequence.detach()).mean()

    return (
        raw_safety.mean().detach(),
        raw_helpfulness.mean().detach(),
        r_s.mean().detach(),
        r_h.mean().detach(),
        r_c.mean().detach(),
        kl_loss.detach(),
        mean_reward,
        objective,
    )


def benchmark_mean_raw_rewards(
    safety_tokenizer,
    safety_model,
    helpfulness_tokenizer,
    helpfulness_model,
    tokenizer,
    rl_model,
    raw_prompt_texts,
    max_prompt_length,
    max_new_tokens,
    no_repeat_ngram_size,
    device,
    batch_size=8,
):
    """Run greedy generation on each prompt, then average raw safety/helpfulness logits.

    ``raw_prompt_texts`` are user strings only; they are wrapped like the RL dataset
    (``User:{text}\\n\\nAssistant: ``) before tokenization.
    """
    if not raw_prompt_texts:
        return float("nan"), float("nan"), [], []

    formatted = [f"User:{p.strip()}\n\nAssistant: " for p in raw_prompt_texts]
    all_raw_s = []
    all_raw_h = []
    all_prompts = []
    all_responses = []

    rl_model.eval()
    with torch.no_grad():
        for i in range(0, len(formatted), batch_size):
            batch = formatted[i : i + batch_size]
            orig_batch = raw_prompt_texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            stopping_criteria = StoppingCriteriaList(
                [
                    StopOnKeywords(
                        tokenizer,
                        keywords=["User:", "Assistant:"],
                        initial_input_len=input_length,
                    )
                ]
            )
            generated_ids = rl_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria,
                no_repeat_ngram_size=no_repeat_ngram_size,
                do_sample=False,
                use_cache=False,
            )
            generated_only_ids = generated_ids[:, input_length:]
            generated_texts = [
                s.strip() for s in tokenizer.batch_decode(generated_only_ids, skip_special_tokens=True)
            ]
            raw_s = get_reward_scores(safety_model, safety_tokenizer, batch, generated_texts, device)
            raw_h = get_reward_scores(
                helpfulness_model, helpfulness_tokenizer, batch, generated_texts, device
            )
            all_raw_s.append(raw_s.detach().flatten())
            all_raw_h.append(raw_h.detach().flatten())
            all_prompts.extend(orig_batch)
            all_responses.extend(generated_texts)

    rl_model.train()
    raw_s_cat = torch.cat(all_raw_s)
    raw_h_cat = torch.cat(all_raw_h)
    return raw_s_cat.mean().item(), raw_h_cat.mean().item(), all_prompts, all_responses
