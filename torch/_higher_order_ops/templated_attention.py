from typing import Callable, Tuple, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._dynamo.variables import UserFunctionVariable
from torch._higher_order_ops.utils import (
    _has_potential_branch_input_mutation,
    UnsupportedAliasMutationException,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class TemplatedAttentionHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("templated_attention")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        score_mod: Callable,
        *other_buffers: torch.Tensor,
    ):
        if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
            raise RuntimeError("Other buffers must be tensors.")
        return super().__call__(query, key, value, score_mod, *other_buffers)


templated_attention = TemplatedAttentionHOP()
templated_attention.__module__ = "torch.ops.higher_order"


class TemplatedAttentionBackwardHOP(HigherOrderOperator):
    def __init__(self):
        super().__init__("templated_attention")

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fw_graph: Callable,  # GraphModule type hint?
        joint_graph: Callable,
        grad_out: torch.Tensor,
        logsumexp: torch.Tensor,
        *other_buffers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not all(isinstance(buf, torch.Tensor) for buf in other_buffers):
            raise RuntimeError("Other buffers must be tensors.")
        return super().__call__(
            query,
            key,
            value,
            fw_graph,
            joint_graph,
            grad_out,
            logsumexp,
            *other_buffers,
        )


templated_attention_backward = TemplatedAttentionBackwardHOP()
templated_attention_backward.__module__ = "torch.ops.higher_order"


def math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager implementation

    This implementation uses vmap to vectorize the score_mod function over the batch, head, m, and n dimensions.
    We then apply the vectorized score_mod function to the scores matrix. Each wrap of vmap applies one of the
    batch, head, m, or n dimensions. We need to apply vmap 4 times to vectorized over all 4 dimensions.

    Args:
        query: The query tensor
        key: The key tensor
        value: The value tensor
        score_mod: The score_mod function
        other_buffers: Other buffers that are passed to the score_mod function
    """
    assert len(other_buffers) == 0, "Other buffers are not yet supported."

    scores = query @ key.transpose(-2, -1)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    in_dim_buffers = (None,) * len(other_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)

    scores = score_mod(scores, b, h, m, n, *other_buffers).to(torch.float32)

    # TODO Unconditionally return logsumexp for backwards
    # if any(t.requires_grad for t in (query, key, value)):
    logsumexp = scores.logsumexp(dim=-1)

    scores = scores.softmax(dim=-1)

    return scores.to(query.dtype) @ value, logsumexp


@templated_attention.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out, lse = math_attention(query, key, value, score_mod, *other_buffers)
    out = out.contiguous()
    return out, lse


def trace_templated_attention(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Traces the templated_attention operator with the given score_mod function and other_buffers.

    Trace SDPA will call make_fx with "fake" example vals and then trace the score_mod function
    This will produce a GraphModule that will be stored on the root tracer as "sdpa_score". We
    access this graph module in inductor to inline the score_mod function to the triton template.
    """
    example_out = templated_attention(query, key, value, score_mod, *other_buffers)

    example_vals = [
        torch.zeros((), dtype=query.dtype, requires_grad=query.requires_grad)
    ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    score_graph = make_fx(score_mod)(*example_vals, *other_buffers)
    proxy_mode.tracer.root.register_module("sdpa_score", score_graph)
    node_args = (query, key, value, score_graph, *other_buffers)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", templated_attention, proxy_args, {}, name="templated_attention"
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@templated_attention.py_impl(ProxyTorchDispatchMode)
def templated_attention_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    if mode.enable_tracing:
        return trace_templated_attention(
            mode, query, key, value, score_mod, *other_buffers
        )
    else:
        return templated_attention(query, key, value, score_mod, *other_buffers)


@templated_attention.py_functionalize_impl
def templated_attention_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the templated_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    other_buffers_unwrapped = ctx.unwrap_tensors(other_buffers)

    # Appease the mypy overlords
    assert isinstance(query_unwrapped, torch.Tensor)
    assert isinstance(key_unwrapped, torch.Tensor)
    assert isinstance(value_unwrapped, torch.Tensor)
    assert isinstance(other_buffers_unwrapped, tuple)
    assert all(isinstance(item, torch.Tensor) for item in other_buffers_unwrapped)

    example_vals = [torch.zeros((), dtype=query.dtype)] + [
        torch.zeros((), dtype=torch.int) for _ in range(4)
    ]
    with ctx.redispatch_to_next() as m:
        functional_score_mod = ctx.functionalize(score_mod)
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        mutates = _has_potential_branch_input_mutation(
            functional_score_mod, example_vals, pre_dispatch
        )
        # The only care about mutations of existing buffers since we can't replay these.
        # However, we can just error if anything is detected
        if mutates:
            raise UnsupportedAliasMutationException("Mutations detected in score_mod")

        out = templated_attention(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            functional_score_mod,
            *other_buffers_unwrapped,
        )
    return ctx.wrap_tensors(out)  # type: ignore[return-value]


@templated_attention.py_impl(FakeTensorMode)
def templated_attention_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    with mode:
        batch_size, num_heads, seq_len_q, _ = query.shape
        logsumexp = query.new_empty(
            batch_size, num_heads, seq_len_q, dtype=torch.float32
        )
        return torch.empty_like(query, memory_format=torch.contiguous_format), logsumexp


# ---------------------------- Autograd Implementation ----------------------------
# # TODO We need to implement an autograd function for this, there is some complexity to do this generically
# templated_attention.py_impl(DispatchKey.Autograd)(
#     autograd_not_implemented(templated_attention, deferred_error=True)
# )

from torch._dispatch.python import suspend_functionalization
from torch._functorch.aot_autograd import (
    AOTConfig,
    create_joint,
    default_partition,
    from_fun,
)

from torch._subclasses.functional_tensor import (
    disable_functional_mode,
    FunctionalTensor,
)
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

# from torch._higher_order_ops.utils import _unstack_pytree, _stack_pytree


dummy_aot_config = AOTConfig(
    fw_compiler=None,  # type: ignore[arg-type]
    bw_compiler=None,  # type: ignore[arg-type]
    partition_fn=None,  # type: ignore[arg-type]
    decompositions={},
    num_params_buffers=0,
    aot_id=0,
    keep_inference_input_mutations=False,
)


def create_fw_bw_graph(
    score_mod: Union[Callable, UserFunctionVariable], index_values, other_buffers
):
    # Note: We create "clean" environments for make_fx by suspending all dispatch keys
    # between Autograd and Python key. Currently, we only suspend functionalization but more can be
    # added when required. Will encounter two problems if we don't suspend functionalization:
    #
    # 1. make_fx fails to capture operations on input: the inputs are wrapped as _to_functional_tensor_wrapper,
    # but they will be unwrapped before entering ProxyTorchDispatchMode as part of the dispatching.
    # However, it's the outside wrapper that tracer creates proxies for. This casuses tracer fail to
    # fetch the proxy for the inputs and fail to capture any operations on them.
    #
    # 2. make_fx fails to capture output: the outputs after ProxyTorchDispatchMode are further
    # wrapped as FunctionalTensorWrapper in Functionalize key after return. However, the tracer
    # only associates the inner tensor with proxy in ProxyTorchDispatchMode. Therefore,
    # when creating the output node, it fails to associate the wrapped tensor with its proxy.
    # Instead, it will create _tensor_constant as output.

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            assert (
                len(other_buffers) == 0
            ), "Other buffers are not yet supported. We will properly generate the graph for them later."

            def _from_fun(t):
                if isinstance(t, torch.Tensor):
                    if t.dtype != torch.bool:
                        return torch.empty_strided(
                            t.size(),
                            t.stride(),
                            dtype=t.dtype,
                            requires_grad=t.requires_grad,
                        )
                    else:
                        # clone of a functional tensor produces a functional tensor
                        # but we want to avoid it so we clone a non-functional version
                        maybe_unfunc_t = t
                        if isinstance(t, FunctionalTensor):
                            torch._sync(t)
                            maybe_unfunc_t = from_fun(t)
                        elif torch._is_functional_tensor(t):
                            # need to handle both types of functionalization here:
                            # these are the tensors that came from the user,
                            # which could be either FunctionalTensorWrapper or FunctionalTensor
                            torch._sync(t)
                            maybe_unfunc_t = torch._from_functional_tensor(t)
                        return maybe_unfunc_t.clone()
                return t

            # See Note:[TemplatedAttention out example value]
            score_mod = (
                score_mod.fn
                if isinstance(score_mod, UserFunctionVariable)
                else score_mod
            )

            unwrapped_score_mod_indexes = pytree.tree_map(_from_fun, index_values)
            unwrapped_other_buffers = pytree.tree_map(_from_fun, other_buffers)
            example_flat_out = pytree.tree_map(
                _from_fun,
                score_mod(*unwrapped_score_mod_indexes, *unwrapped_other_buffers),
            )
            if not isinstance(example_flat_out, torch.Tensor):
                raise RuntimeError(
                    "Expected output of score_mod to be a tensor."
                    f"Got type {type(example_flat_out)}."
                )
            example_grad = _from_fun(example_flat_out)

            fw_graph = make_fx(score_mod)(
                *unwrapped_score_mod_indexes, *unwrapped_other_buffers
            )

        def joint_f(index_values, other_buffers, example_grad):
            def fw_with_masks(*args):
                fw_out = score_mod(*args)
                out_requires_grad = fw_out.requires_grad
                return ((fw_out,), (out_requires_grad,))

            joint = create_joint(fw_with_masks, aot_config=dummy_aot_config)
            args = index_values + list(other_buffers)
            optional_grad = [example_grad] if example_grad.requires_grad else []
            _, grads = joint(args, optional_grad)

            # In order to keep map functional for backward graph,
            # we clone outputs that are aliasing inputs
            # input_storage = {
            #     StorageWeakRef(arg._typed_storage())
            #     for arg in example_args
            #     if isinstance(arg, torch.Tensor)
            # }

            return grads
            # return pytree.tree_map(maybe_clone, grads)

        joint_graph = make_fx(joint_f)(
            unwrapped_score_mod_indexes, unwrapped_other_buffers, example_grad
        )
        fwd_graph, bwd_graph = default_partition(
            joint_graph,
            (unwrapped_score_mod_indexes, unwrapped_other_buffers, example_grad),
            num_fwd_outputs=1,
        )
        return fw_graph, joint_graph


class TemplatedAttentionAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, fw_graph, joint_graph, *other_buffers
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        any_buffer_requires_grad = any(buffer.requires_grad for buffer in other_buffers)
        assert (
            not any_buffer_requires_grad
        ), "Captured buffers that require grad are not yet supported."
        ctx.save_for_backward(query, key, value, *other_buffers)
        ctx._fw_graph = fw_graph
        ctx._joint_graph = joint_graph
        # ctx._num_mapped_args = num_mapped_args
        with torch._C._AutoDispatchBelowAutograd():
            # Need to have out and logsumexp returned
            return templated_attention(query, key, value, fw_graph, *other_buffers)

    @staticmethod
    def backward(ctx, grad_out, logsumexp_grad):
        fw_args = ctx.saved_tensors
        query, key, value, *other_buffers = fw_args
        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph

        # We have asserted that other_buffers do not require grad in the forward
        none_grads = [None] * (2 + len(other_buffers))
        # TODO no double backward, error loud!
        with torch._C._AutoDispatchBelowAutograd():
            grad_query, grad_key, grad_value = templated_attention_backward(
                query,
                key,
                value,
                fw_graph,
                joint_graph,
                grad_out,
                logsumexp_grad,
                *other_buffers,
            )
        return grad_query, grad_key, grad_value, *none_grads

@templated_attention.py_impl(DispatchKey.Autograd)
def templated_attention_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    *other_buffers: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_requires_grad = query.requires_grad or key.requires_grad
    example_vals = [
        torch.zeros((), dtype=query.dtype, requires_grad=input_requires_grad)
    ] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    fw_graph, bw_graph = create_fw_bw_graph(score_mod, example_vals, other_buffers)
    out, logsumexp = TemplatedAttentionAutogradOp.apply(
        query, key, value, fw_graph, bw_graph, *other_buffers
    )
    return out, logsumexp


# ---------------------------- Autograd Implementation ----------------------------


def math_attention_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fw_graph: Callable,
    joint_graph: Callable,
    grad_out: torch.Tensor,
    logsumexp: torch.Tensor,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Lets not do this efficiently for now
    # The kernel always recomputes the forward pass, lets do that
    scores = query @ key.transpose(-2, -1)

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    in_dim_buffers = (None,) * len(other_buffers)
    score_mod = torch.vmap(fw_graph, in_dims=(0, None, None, None, 0) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None) + in_dim_buffers)
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None) + in_dim_buffers)

    scores = score_mod(scores, b, h, m, n, *other_buffers).to(torch.float32)

    softmax_scores = scores.softmax(dim=-1)

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out @ value.transpose(-2, -1)

    sum_scores = torch.sum(grad_softmax_scores * softmax_scores, -1, keepdim=True)
    grad_score_mod = softmax_scores * (grad_softmax_scores - sum_scores)

    # Gradient of the inline score_mod function, with respect to the scores
    grad_scores = joint_graph(
        grad_score_mod, *other_buffers
    )  # figure how we actually call the joint graph

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query
    return grad_query, grad_key, grad_value


@templated_attention_backward.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    grad_out: torch.Tensor,
    logsumexp: torch.Tensor,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return math_attention_backward(
        query, key, value, fw_graph, joint_graph, grad_out, logsumexp, *other_buffers
    )


def trace_templated_attention_backward(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    grad_out: torch.Tensor,
    logsumexp: torch.Tensor,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ """
    pass


@templated_attention_backward.py_impl(ProxyTorchDispatchMode)
def templated_attention_backward_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    grad_out: torch.Tensor,
    logsumexp: torch.Tensor,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mode is not None, "Mode should always be enabled for python fallback key"
    pass


@templated_attention_backward.py_functionalize_impl
def templated_attention_backward_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    grad_out: torch.Tensor,
    logsumexp: torch.Tensor,
    *other_buffers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Defines the functionalization rules for the templated_attention operator.

    Write now we are unwrapping each tensor and then redispatching to the next, however we want to
    guard against any mutations in the score_mod function, to the other_buffers since those
    are free variables.
    """
    pass


@templated_attention_backward.py_impl(FakeTensorMode)
def templated_attention_backward_fake_tensor_mode(
    mode: FakeTensorMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    grad_out: torch.Tensor,
    logsumexp: torch.Tensor,
    *other_buffers: Tuple[torch.Tensor, ...],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pass
