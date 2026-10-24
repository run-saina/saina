"""Bidirectional prompt attention for checkpoint format 4 (Qwen3.5 backbones).

In the full-attention layers (one in four), every prompt token (the chat-templated context, all questions with
their options, and any images) attends to every other prompt token; the readout slots after the prompt stay
causal, and the linear-attention layers stay causal everywhere. Keep aligned with the training mask
(uniclass.prefix_attention). One unpadded row per forward, as the scorers run it.
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


def _patch():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original = qwen.create_causal_mask

    def create_causal_mask(*args, **kwargs):
        prompt = getattr(_state, 'prompt', None)
        if prompt is None:
            return original(*args, **kwargs)

        def both_ways(batch_idx, head_idx, q_idx, kv_idx):
            return (q_idx < prompt) & (kv_idx < prompt)

        return original(*args, **kwargs, or_mask_function=both_ways)

    qwen.create_causal_mask = create_causal_mask
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
