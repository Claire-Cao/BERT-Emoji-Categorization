from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:  # as Theta in Algorithm
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # add implementation of AdamW here
                if len(state) == 0:
                    state["first_moment_m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["second_moment_v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["step"] = 0

                beta_1, beta_2 = group["betas"]
                eps = group["eps"]
                state["first_moment_m"] = beta_1 * state["first_moment_m"] + (1 - beta_1) * grad
                state["second_moment_v"] = beta_2 * state["second_moment_v"] + (1 - beta_2) * grad * grad
                state["step"] += 1
                alpha_t = alpha * math.sqrt(1 - beta_2 ** state["step"]) / (1 - beta_1 ** state["step"])
                delta_params = -alpha_t * state["first_moment_m"] / (torch.sqrt(state["second_moment_v"]) + eps)

                p.data = p.data + delta_params
                p.data = p.data - alpha * group["weight_decay"] * p.data

        return loss
