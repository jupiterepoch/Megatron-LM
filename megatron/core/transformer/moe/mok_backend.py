# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Mixture-of-Kittens (MoK) megakernel backend for Megatron-Core MoE layers.

This backend replaces the dispatcher + grouped-expert forward for the *routed* experts
only.  Megatron keeps computing its own shared expert -- including Qwen's learned sigmoid
shared-expert gate -- in the ordinary PyTorch graph, and its output is added to the routed
output.  That keeps the model and checkpoint semantics identical to the existing backend
while still collapsing all routed-expert compute and EP communication into one kernel.

The backend is opt-in and never falls back silently: if any guard fails it raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

try:  # pragma: no cover - requires the vendored extension to be built
    from megatron.core.extensions.mixture_of_kittens.mok import functional as mok_functional
    from megatron.core.extensions.mixture_of_kittens.mok.ops import (
        mxfp8_quantize as _mok_mxfp8_quantize,
    )

    HAVE_MOK = True
    _MOK_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover
    mok_functional = None
    _mok_mxfp8_quantize = None
    HAVE_MOK = False
    _MOK_IMPORT_ERROR = exc


_SUPPORTED_EP_SIZES = (1, 4, 8, 16, 32, 64)


def require_mok() -> None:
    """Raises a clear error when the optional MoK dependency is missing."""
    if not HAVE_MOK:
        raise ImportError(
            "moe_megakernel_backend='mok' requires the Mixture-of-Kittens submodule at "
            "megatron/core/extensions/mixture_of_kittens, checked out and built:\n"
            "    git submodule update --init --recursive "
            "megatron/core/extensions/mixture_of_kittens\n"
            "    cd megatron/core/extensions/mixture_of_kittens && make ARCH=SM100\n"
            "--recursive matters: MoK vendors ThunderKittens as its own submodule and the "
            "build needs those headers. Use ARCH=SM100 for GB200 or ARCH=SM103 for GB300; "
            "the default target is SM103 and emits no PTX fallback, so a default build "
            "cannot launch on an SM100 device."
        ) from _MOK_IMPORT_ERROR


@dataclass(frozen=True)
class MoKBackendConfig:
    """Tunables forwarded to MoK, plus the precision selector."""

    fwd_num_comm_sms: int = 40
    bwd_num_comm_sms: int = 28
    minibatch_size: int = 4096
    macrobatch_size: int = 131072
    schedule_capacity_multiplier: float = 0.5
    use_mxfp8: bool = True
    recompute_forward_context: bool = False

    def to_mok(self) -> Any:
        require_mok()
        return mok_functional.MoKConfig(
            fwd_num_comm_sms=self.fwd_num_comm_sms,
            bwd_num_comm_sms=self.bwd_num_comm_sms,
            minibatch_size=self.minibatch_size,
            macrobatch_size=self.macrobatch_size,
            schedule_capacity_multiplier=self.schedule_capacity_multiplier,
        )


def check_compatibility(config, ep_size: int, num_local_tokens: int) -> None:
    """Rejects every configuration whose semantics MoK does not reproduce exactly.

    Raises rather than falling back, so a mis-set recipe fails loudly at startup instead of
    silently training a different model.
    """
    require_mok()
    problems: list[str] = []

    if getattr(config, "tensor_model_parallel_size", 1) != 1:
        problems.append("tensor_model_parallel_size must be 1")
    if getattr(config, "expert_tensor_parallel_size", 1) not in (None, 1):
        problems.append("expert_tensor_parallel_size must be 1")
    if ep_size not in _SUPPORTED_EP_SIZES:
        problems.append(f"expert_model_parallel_size={ep_size} is not one of {_SUPPORTED_EP_SIZES}")
    if getattr(config, "moe_expert_capacity_factor", None) is not None:
        problems.append("token dropping (moe_expert_capacity_factor) is unsupported")
    if getattr(config, "moe_pad_expert_input_to_capacity", False):
        problems.append("moe_pad_expert_input_to_capacity is unsupported")
    if getattr(config, "moe_router_enable_expert_bias", False):
        problems.append("expert bias is unsupported")
    if not getattr(config, "gated_linear_unit", False):
        problems.append("MoK implements SwiGLU only; gated_linear_unit must be True")
    if getattr(config, "moe_mlp_glu_interleave_size", None):
        problems.append(
            "moe_mlp_glu_interleave_size changes the [gate;up] FC1 storage order and is unsupported"
        )
    shared_size = getattr(config, "moe_shared_expert_intermediate_size", None)
    routed_size = getattr(config, "moe_ffn_hidden_size", None)
    if shared_size is not None and routed_size is not None and shared_size != routed_size:
        problems.append(
            "MoK requires the shared and routed intermediate sizes to match "
            f"(shared={shared_size}, routed={routed_size})"
        )
    if routed_size is None or routed_size % 256:
        problems.append(f"moe_ffn_hidden_size={routed_size} must be divisible by 256")
    hidden = getattr(config, "hidden_size", None)
    if hidden is None or hidden % 256:
        problems.append(f"hidden_size={hidden} must be divisible by 256")
    if num_local_tokens < 512 or num_local_tokens % 256:
        problems.append(
            f"local token count {num_local_tokens} must be at least 512 and divisible by 256"
        )
    num_experts = getattr(config, "num_moe_experts", None)
    if num_experts is None or num_experts % ep_size:
        problems.append(f"num_moe_experts={num_experts} must be divisible by ep_size={ep_size}")

    if problems:
        raise ValueError(
            "moe_megakernel_backend='mok' is not compatible with this configuration:\n  - "
            + "\n  - ".join(problems)
        )


def topk_from_routing_map(
    probs: torch.Tensor, routing_map: torch.Tensor, topk: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Builds MoK's compact ``[M, topk]`` routing metadata from Megatron's dense outputs.

    Deliberately avoids ``routing_map.nonzero()``: its output size is data dependent, which
    forces a device-to-host synchronization on every MoE layer of every microbatch.  A topk
    over the 0/1 map is exact (there are exactly ``topk`` ones per row) and stays on device.

    The gradient flows through ``gather`` back into the dense ``probs`` tensor, so Megatron's
    global auxiliary-loss autograd attachment is preserved unchanged.
    """
    selected = routing_map.to(probs.dtype)
    topk_ids = selected.topk(topk, dim=1).indices.to(torch.int64).contiguous()
    topk_weights = probs.gather(1, topk_ids).to(torch.float32).contiguous()
    return topk_ids, topk_weights


def stack_expert_weights(
    fc1_weights: list[torch.Tensor], fc2_weights: list[torch.Tensor], intermediate_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts Megatron's per-expert parameters into MoK's grouped layout.

    Megatron/TE store FC1 as ``[2*I, H]`` per expert with gate first then up, and FC2 as
    ``[H, I]``.  MoK wants ``gate/up = [E, I, H]`` and ``down = [E, H, I]``.
    """
    gate = torch.stack([w[:intermediate_size] for w in fc1_weights], dim=0).contiguous()
    up = torch.stack([w[intermediate_size:] for w in fc1_weights], dim=0).contiguous()
    down = torch.stack(list(fc2_weights), dim=0).contiguous()
    return gate, up, down


def unstack_expert_grads(
    d_gate: torch.Tensor, d_up: torch.Tensor, d_down: torch.Tensor
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Inverse of :func:`stack_expert_weights` for gradients."""
    num_local_experts = d_gate.shape[0]
    d_fc1 = [torch.cat((d_gate[i], d_up[i]), dim=0) for i in range(num_local_experts)]
    d_fc2 = [d_down[i] for i in range(num_local_experts)]
    return d_fc1, d_fc2


class MoKRoutedExpertsFunction(torch.autograd.Function):
    """Autograd bridge over MoK's explicit routed-only forward/backward.

    MoK's Python package registers no autograd, so Megatron owns it here.  The original
    per-expert parameters are passed as explicit ``apply()`` inputs so that DDP and the
    distributed optimizer see ordinary per-parameter gradients in their original layout,
    leaving the checkpoint schema untouched.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        backend_config: MoKBackendConfig,
        ep_group,
        num_local_experts: int,
        intermediate_size: int,
        *expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        require_mok()
        num_local_tokens, hidden_size = x.shape
        topk = topk_ids.shape[1]
        fc1_weights = list(expert_weights[:num_local_experts])
        fc2_weights = list(expert_weights[num_local_experts:])

        mok_config = backend_config.to_mok()
        workspace = mok_functional.get_workspace(
            mok_config,
            ep_group,
            device=x.device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_size,
            topk=topk,
        )
        schedule = mok_functional.build_schedule(
            workspace, mok_config, topk_ids, num_local_experts=num_local_experts
        )

        gate, up, down = stack_expert_weights(fc1_weights, fc2_weights, intermediate_size)
        if backend_config.use_mxfp8:
            g_fp8, g_sc, g_t_fp8, g_t_sc = _mok_mxfp8_quantize(gate, True, True)
            u_fp8, u_sc, u_t_fp8, u_t_sc = _mok_mxfp8_quantize(up, True, True)
            d_fp8, d_sc, d_t_fp8, d_t_sc = _mok_mxfp8_quantize(down, True, True)
            fwd_weights = ((g_fp8, g_sc), (u_fp8, u_sc), (d_fp8, d_sc))
            ctx.quantized = (
                (g_fp8, g_sc, g_t_fp8, g_t_sc),
                (u_fp8, u_sc, u_t_fp8, u_t_sc),
                (d_t_fp8, d_t_sc),
            )
            ctx.recompute_weights = ((g_fp8, g_sc), (u_fp8, u_sc))
        else:
            fwd_weights = (gate, up, down)
            ctx.quantized = None
            ctx.recompute_weights = (gate, up)

        output, forward_context = mok_functional.forward_routed(
            mok_config, workspace, schedule, x, topk_weights, *fwd_weights
        )

        if backend_config.recompute_forward_context:
            forward_context = None

        ctx.save_for_backward(x, topk_weights, *expert_weights)
        ctx.mok_config = mok_config
        ctx.backend_config = backend_config
        ctx.workspace = workspace
        ctx.schedule = schedule
        ctx.forward_context = forward_context
        ctx.num_local_experts = num_local_experts
        ctx.intermediate_size = intermediate_size
        # BF16 weights are kept only when they are what the backward consumes.
        ctx.bf16_weights = None if backend_config.use_mxfp8 else (gate, up, down)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x, topk_weights, *expert_weights = ctx.saved_tensors
        num_local_experts = ctx.num_local_experts
        backend_config = ctx.backend_config
        grad_output = grad_output.contiguous()

        forward_context = ctx.forward_context
        if forward_context is None:
            forward_context = mok_functional.recompute_forward_context_routed(
                ctx.mok_config, ctx.workspace, ctx.schedule, x, *ctx.recompute_weights
            )

        if backend_config.use_mxfp8:
            bwd_weights = ctx.quantized
        else:
            bwd_weights = ctx.bf16_weights

        d_x, d_topk_weights, d_gate, d_up, d_down = mok_functional.backward_routed(
            ctx.mok_config,
            ctx.workspace,
            ctx.schedule,
            forward_context,
            grad_output,
            x,
            topk_weights,
            *bwd_weights,
        )

        d_fc1, d_fc2 = unstack_expert_grads(d_gate, d_up, d_down)
        weight_grads = [
            grad.to(weight.dtype) for grad, weight in zip(d_fc1 + d_fc2, expert_weights)
        ]
        # x, topk_ids, topk_weights, backend_config, ep_group, num_local_experts,
        # intermediate_size, *expert_weights
        return (d_x, None, d_topk_weights, None, None, None, None, *weight_grads)


def mok_routed_experts_forward(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    *,
    fc1_weights: list[torch.Tensor],
    fc2_weights: list[torch.Tensor],
    intermediate_size: int,
    topk: int,
    ep_group,
    backend_config: MoKBackendConfig,
) -> torch.Tensor:
    """Runs the routed experts through MoK and returns the layer-shaped routed output.

    Inputs are Megatron's layer-native ``[S, B, H]`` hidden states plus the router's dense
    probabilities and boolean routing map.  The shared expert is *not* included; the caller
    adds its own (optionally gated) shared branch.
    """
    require_mok()
    seq_len, batch, hidden_size = hidden_states.shape
    x = hidden_states.reshape(seq_len * batch, hidden_size).contiguous().to(torch.bfloat16)
    topk_ids, topk_weights = topk_from_routing_map(probs, routing_map, topk)

    output = MoKRoutedExpertsFunction.apply(
        x,
        topk_ids,
        topk_weights,
        backend_config,
        ep_group,
        len(fc1_weights),
        intermediate_size,
        *fc1_weights,
        *fc2_weights,
    )
    return output.reshape(seq_len, batch, hidden_size).to(hidden_states.dtype)
