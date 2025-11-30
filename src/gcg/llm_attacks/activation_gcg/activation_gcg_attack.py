import gc
import torch
import numpy as np

from llm_attacks import AttackPrompt, PromptManager, MultiPromptAttack
from llm_attacks import get_embedding_matrix, get_embeddings, get_nonascii_toks


def token_gradients_activation(model, input_ids, input_slice, pos, layer, direction):
    """
    Compute gradients of the activation projection onto `direction` w.r.t. control tokens.

    The activation is taken at `layer` (resid_pre) and position `pos` (can be negative;
    negative values index from the end of the sequence length).
    """
    # Clear stale grads
    model.zero_grad(set_to_none=True)

    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    # stitch together embeddings (others are detached)
    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [embeds[:, : input_slice.start, :], input_embeds, embeds[:, input_slice.stop :, :]],
        dim=1,
    )

    captured = {}

    def pre_hook(module, hook_input):
        # Keep gradient flow for the control tokens; do not detach.
        captured["act"] = hook_input[0]

    handle = model.model.layers[layer].register_forward_pre_hook(pre_hook)
    _ = model(inputs_embeds=full_embeds)
    handle.remove()

    hidden = captured["act"][0]  # [seq, d_model]
    seq_len = hidden.shape[0]
    idx = pos if pos >= 0 else seq_len + pos
    idx = max(min(idx, seq_len - 1), 0)

    direction = direction.to(hidden)
    proj = torch.dot(hidden[idx], direction)
    proj.backward()

    grad = one_hot.grad.clone()
    del embeds, full_embeds, input_embeds, one_hot, hidden, captured
    gc.collect()
    return grad


class ActivationAttackPrompt(AttackPrompt):
    def __init__(self, *args, direction=None, layer=None, pos=-1, **kwargs):
        super().__init__(*args, **kwargs)
        assert direction is not None and layer is not None
        self.direction = direction
        self.layer = layer
        self.pos = pos

    def grad_activation(self, model):
        return token_gradients_activation(
            model,
            self.input_ids.to(model.device),
            self._control_slice,
            self.pos,
            self.layer,
            self.direction,
        )

    def activation_score(self, model, direction=None, layer=None, pos=None, test_controls=None):
        """
        Compute projection score(s) for provided control strings/tensors.
        Returns a 1D tensor of scores (lower is better).
        """
        direction = direction if direction is not None else self.direction
        layer = layer if layer is not None else self.layer
        pos = self.pos if pos is None else pos

        pad_tok = -1
        if test_controls is None:
            test_controls = self.control_toks

        if isinstance(test_controls, torch.Tensor):
            if len(test_controls.shape) == 1:
                test_controls = test_controls.unsqueeze(0)
            test_ids = test_controls.to(model.device)
        elif isinstance(test_controls, list) and isinstance(test_controls[0], str):
            max_len = self._control_slice.stop - self._control_slice.start
            test_ids = [
                torch.tensor(
                    self.tokenizer(control, add_special_tokens=False).input_ids[:max_len],
                    device=model.device,
                )
                for control in test_controls
            ]
            pad_tok = 0
            while pad_tok in self.input_ids or any([pad_tok in ids for ids in test_ids]):
                pad_tok += 1
            nested_ids = torch.nested.nested_tensor(test_ids)
            test_ids = torch.nested.to_padded_tensor(nested_ids, pad_tok, (len(test_ids), max_len))
        else:
            raise ValueError("test_controls must be a tensor or list of strings")

        locs = torch.arange(self._control_slice.start, self._control_slice.stop).repeat(
            test_ids.shape[0], 1
        ).to(model.device)
        ids = torch.scatter(
            self.input_ids.unsqueeze(0).repeat(test_ids.shape[0], 1).to(model.device),
            1,
            locs,
            test_ids,
        )
        attn_mask = (ids != pad_tok).type(ids.dtype) if pad_tok >= 0 else None

        captured = {}

        def pre_hook(module, hook_input):
            captured["act"] = hook_input[0]

        handle = model.model.layers[layer].register_forward_pre_hook(pre_hook)
        _ = model(input_ids=ids, attention_mask=attn_mask)
        handle.remove()

        activations = captured["act"]  # [batch, seq, d_model]
        lengths = attn_mask.sum(-1) if attn_mask is not None else torch.tensor(
            [ids.shape[1]] * ids.shape[0], device=ids.device
        )
        idxs = []
        for b in range(activations.shape[0]):
            seq_len = int(lengths[b].item())
            idx = pos if pos >= 0 else seq_len + pos
            idx = max(min(idx, seq_len - 1), 0)
            idxs.append(idx)
        idxs = torch.tensor(idxs, device=activations.device)

        direction = direction.to(activations)
        scores = torch.stack(
            [torch.dot(activations[b, idxs[b], :], direction) for b in range(activations.shape[0])]
        )

        del ids, locs, test_ids, activations, captured
        gc.collect()
        return scores


class ActivationPromptManager(PromptManager):
    def __init__(self, *args, direction=None, layer=None, pos=-1, managers=None, **kwargs):
        assert direction is not None and layer is not None
        self.direction = direction
        self.layer = layer
        self.pos = pos

        goals, targets, tokenizer, conv_template, control_init, test_prefixes = args[:6]

        self.tokenizer = tokenizer
        self._prompts = [
            managers["AP"](
                goal,
                target,
                tokenizer,
                conv_template,
                control_init,
                test_prefixes,
                direction=direction,
                layer=layer,
                pos=pos,
            )
            for goal, target in zip(goals, targets)
        ]
        self._nonascii_toks = get_nonascii_toks(tokenizer, device="cpu")

    def grad_activation(self, model):
        # Aggregate grad_activation across prompts (mirrors PromptManager.grad)
        return sum([prompt.grad_activation(model) for prompt in self._prompts])

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):
        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.inf
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, len(control_toks), len(control_toks) / batch_size, device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos],
            1,
            torch.randint(0, topk, (batch_size, 1), device=grad.device),
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks


class ActivationMultiPromptAttack(MultiPromptAttack):
    def __init__(
        self,
        goals,
        targets,
        workers,
        direction,
        layer,
        pos=-1,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        test_prefixes=[
            "I'm sorry",
            "I apologize",
            "As an",
            "I'm just",
            "I'm an",
            "I cannot",
            "Hello!",
        ],
        logfile=None,
        managers=None,
        test_goals=[],
        test_targets=[],
        test_workers=[],
        *args,
        **kwargs,
    ):
        self.goals = goals
        self.targets = targets
        self.workers = workers
        self.test_goals = test_goals
        self.test_targets = test_targets
        self.test_workers = test_workers
        self.test_prefixes = test_prefixes
        self.models = [worker.model for worker in workers]
        self.logfile = logfile
        self.direction = direction
        self.layer = layer
        self.pos = pos
        self.managers = managers

        self.prompts = [
            managers["PM"](
                goals,
                targets,
                worker.tokenizer,
                worker.conv_template,
                control_init,
                test_prefixes,
                direction=direction,
                layer=layer,
                pos=pos,
                managers=managers,
            )
            for worker in workers
        ]

    def step(
        self,
        batch_size=1024,
        topk=256,
        temp=1,
        allow_non_ascii=True,
        target_weight=1,
        control_weight=0.0,
        verbose=False,
        opt_only=False,
        filter_cand=True,
    ):
        # opt_only unused; kept for signature compatibility
        opt_only = False
        main_device = self.models[0].device
        control_cands = []

        # 1) gradients for each prompt/model
        for j, worker in enumerate(self.workers):
            worker(self.prompts[j], "grad_activation", worker.model)

        grad = None
        for j, worker in enumerate(self.workers):
            new_grad = worker.results.get().to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j - 1].sample_control(
                        grad, batch_size, topk, temp, allow_non_ascii
                    )
                    control_cands.append(
                        self.get_filtered_cands(
                            j - 1, control_cand, filter_cand=filter_cand, curr_control=self.control_str
                        )
                    )
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(
                self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str)
            )
        del grad, control_cand
        gc.collect()

        # 2) Evaluate candidates
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                progress = (
                    tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else range(len(self.prompts[0]))
                )
                for i in progress:
                    for k, worker in enumerate(self.workers):
                        worker(
                            self.prompts[k][i],
                            "activation_score",
                            worker.model,
                            self.direction,
                            self.layer,
                            self.pos,
                            cand,
                        )
                    scores = [worker.results.get() for worker in self.workers]
                    loss[j * batch_size : (j + 1) * batch_size] += sum(
                        [s.to(main_device) for s in scores]
                    )
                    del scores
                    gc.collect()
                    if verbose:
                        progress.set_description(
                            f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}"
                        )

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]

        del control_cands, loss
        gc.collect()

        print("Current length:", len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        # average across prompts/models for reporting
        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
