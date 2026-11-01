import torch
import torch.nn as nn
from jaxtyping import jaxtyped, Float, Int
from beartype import beartype as typecheker
from math import sqrt
from typing import Tuple

class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index:str):

        self.ignore_index = ignore_index # Evita que <pad> seja considerado para o cálculo de perda

    def forward(
        self,
        logits: Float[torch.Tensor, "N Vocab_size"],
        targets: Float[torch.Tensor, "N"]
    )->Float[torch.Tensor, ""]:
        max_logit = torch.max(logits)
        stable_logits = logits - max_logit

        sum_exp = torch.sum(torch.exp(stable_logits), keepdim=True, dim=-1)
        log_sum_exp = torch.log(sum_exp)
        log_probs = stable_logits - log_sum_exp 

        target_unsqueeze = targets.unsqueeze(1)
        target_log_probs = torch.gather(log_probs, dim=-1, index=target_unsqueeze).squeeze(-1)

        valid_mask = (targets != self.ignore_index)
        valid_target_log_probs = target_log_probs[valid_mask]

        if valid_target_log_probs.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        return -torch.mean(valid_target_log_probs)       

class Adam(torch.optim.Optimizer):

    def __init__(self,
                params: Float[torch.Tensor, "batch seq_len d_model"],
                lr: float = 1e-3,
                betas: Tuple[float, float]=(0.9, 0.99),
                eps:float = 1e-9
        ):

        super().__init__()

        if lr < 0.0:
            raise ValueError("Invalid Beta Value")
        if not 0.0 <= betas[0] < 1 and not 0.0 <= betas[0] < 1:
            raise ValueError("Invalid Beta Value")
        if eps < 0.0:
            raise ValueError("Invalid epsilon value")

        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params=params, defaults=defaults)

    @torch.no_grad()
    def step(self, closure=None)-> None:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group("lr")           
            beta1,beta2 = group("betas")
            eps = group("eps")

            for p in group("params"):
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state["exp_avg"] = torch.zeros(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                t = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, alpha=1.0-beta1) 

                bias_correction1 = 1.0 - (beta1**t)
                bias_correction2 = 1.0 - (beta2**t)

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / sqrt(bias_correction2)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)
        
        return loss
    