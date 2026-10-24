"""Bidirectional prompt attention for checkpoint format 4 (Qwen3.5 backbones).

In the full-attention layers (one in four), every prompt token (the chat-templated context, all questions with
their options, and any images) attends to every other prompt token; the readout slots after the prompt stay
causal, and the linear-attention layers stay causal everywhere. Keep aligned with the training mask
(uniclass.prefix_attention). One unpadded row per forward, as the scorers run it.

The attention is computed in two parts, never as a dense (sequence x sequence) mask: the prompt queries read the
prompt keys with no mask at all, and the few readout-slot queries read every key through a small
(slots x sequence) mask. Grouped-query heads are folded into the query length, so the key and value heads are
never repeated. When the plain causal mask is not the skippable one (padding, or an attention implementation
that always materialises masks), the dense-mask path is used instead; both give the same attention.
"""
import contextlib
import threading

MODES = ('causal', 'bidirectional_prefix')
# Per thread: the mask is built inside the forward, on the thread that runs it, so concurrent forwards of
# different checkpoints in one process do not see each other's prompt length.
_state = threading.local()
_installed = []
_lock = threading.Lock()


def _install():
    with _lock:
        if not _installed:
            _patch()


def prefix_attention(query, key, value, prompt, scaling, dropout=0.0):
    """Attention output (batch, length, heads, head_dim) where queries before `prompt` read every key before
    `prompt` and later queries read causally. query: (batch, heads, length, head_dim); key and value:
    (batch, kv_heads, length, head_dim), query head h using key head h // (heads // kv_heads)."""
    import torch
    sdpa = torch.nn.functional.scaled_dot_product_attention
    batch, heads, length, dim = query.shape
    kv_heads = key.shape[1]
    groups = heads // kv_heads
    if key.shape[2] != length or groups * kv_heads != heads:
        raise ValueError('Bidirectional prompt attention needs one full pass without a cache')
    prompt = max(0, min(int(prompt), length))

    def grouped(q):  # (batch, heads, n, dim) -> (batch, kv_heads, groups * n, dim), groups sharing a key head
        n = q.shape[2]
        return q.reshape(batch, kv_heads, groups * n, dim), n

    def ungrouped(out, n):
        return out.reshape(batch, heads, n, dim)

    parts = []
    if prompt:
        q, n = grouped(query[:, :, :prompt])
        parts.append(ungrouped(sdpa(q, key[:, :, :prompt], value[:, :, :prompt], dropout_p=dropout, scale=scaling),
                               n))
    if prompt < length:
        q, n = grouped(query[:, :, prompt:])
        rows = torch.arange(prompt, length, device=query.device)[:, None]
        cols = torch.arange(length, device=query.device)[None, :]
        mask = ((cols < prompt) | (cols <= rows)).repeat(groups, 1)
        parts.append(ungrouped(sdpa(q, key, value, attn_mask=mask, dropout_p=dropout, scale=scaling), n))
    out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=2)
    return out.transpose(1, 2).contiguous()


class _Interfaces:
    """The attention-function registry seen by the Qwen3.5 modeling file: its decoder attention runs
    prefix_attention while a prompt is set and the causal mask was skipped; everything else is unchanged."""

    def __init__(self, registry, attention_class):
        self._registry = registry
        self._attention_class = attention_class

    def get_interface(self, attn_implementation, default):
        interface = self._registry.get_interface(attn_implementation, default)

        def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
            prompt = getattr(_state, 'prompt', None)
            if prompt is None or attention_mask is not None or not isinstance(module, self._attention_class):
                return interface(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout,
                                 **kwargs)
            return prefix_attention(query, key, value, prompt, scaling, dropout), None

        return attention

    def __getattr__(self, name):
        return getattr(self._registry, name)


def _patch():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original = qwen.create_causal_mask

    def create_causal_mask(*args, **kwargs):
        prompt = getattr(_state, 'prompt', None)
        if prompt is None:
            return original(*args, **kwargs)
        # No mask means a plain causal pass without padding: the attention then runs prefix_attention.
        mask = original(*args, **kwargs)
        if mask is None:
            return None

        def both_ways(batch_idx, head_idx, q_idx, kv_idx):
            return (q_idx < prompt) & (kv_idx < prompt)

        return original(*args, **kwargs, or_mask_function=both_ways)

    qwen.create_causal_mask = create_causal_mask
    qwen.ALL_ATTENTION_FUNCTIONS = _Interfaces(qwen.ALL_ATTENTION_FUNCTIONS, qwen.Qwen3_5Attention)
    _installed.append(True)


@contextlib.contextmanager
def prompt_attention(mode, prompt_tokens):
    """Within this block the first `prompt_tokens` tokens read both ways (mode 'bidirectional_prefix')."""
    if mode == 'causal':
        yield
        return
    if mode not in MODES:
        raise ValueError(f'Unsupported prompt attention {mode!r}; upgrade the saina package')
    _install()
    previous = getattr(_state, 'prompt', None)
    _state.prompt = prompt_tokens
    try:
        yield
    finally:
        _state.prompt = previous
