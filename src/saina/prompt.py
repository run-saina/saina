"""Prompt formatting shared by the inference server and the token counter. No torch import."""
import json
from itertools import product
from .errors import InputTooLong

PROMPT_FORMATS = ('BaseScorer.encode_prompt', 'AgenticScorer.encode_prompt')


def answer_codes(tokenizer):
    """Single-token answer codes in training order, with their token IDs."""
    codes, ids = [], []
    for width in (1, 2):
        for letters in product('ABCDEFGHIJKLMNOPQRSTUVWXYZ', repeat=width):
            code = ''.join(letters)
            encoded = tokenizer.encode(code, add_special_tokens=False)
            if len(encoded) == 1:
                codes.append(code)
                ids.append(encoded[0])
    return codes, ids


def _render(tokenizer, content):
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer.encode(text, add_special_tokens=False)


def encode_head_prompt(tokenizer, codes, context, question, choices, max_length):
    if not 2 <= len(choices) <= len(codes):
        raise ValueError('Unsupported option count')
    # Keep byte-for-byte aligned with BaseScorer.encode_prompt used during training.
    content = ('Answer the multiple-choice question using the supplied context. '
               'Choose exactly one of the provided options. Reply with only its code, '
               'with no leading whitespace, explanation, or punctuation.\n' +
               json.dumps({'context': context, 'question': question, 'options': [
                   {'code': codes[i], 'text': text} for i, text in enumerate(choices)
               ]}, ensure_ascii=False))
    ids = _render(tokenizer, content)
    if len(ids) > max_length:
        raise InputTooLong('Input exceeds model context window; shorten it explicitly')
    return ids


def encode_agentic_prompt(tokenizer, codes, context, question, choices, max_length, mode='single_label'):
    if mode == 'single_label':
        return encode_head_prompt(tokenizer, codes, context, question, choices, max_length)
    if mode != 'multi_label' or not 2 <= len(choices) <= len(codes):
        raise ValueError('Invalid mode or option count')
    content = ('Evaluate each option independently using the supplied context. '
               'Select all applicable options; zero, one, or multiple may apply.\n' +
               json.dumps({'mode': mode, 'context': context, 'question': question,
                   'options': [{'code': codes[i], 'text': v} for i, v in enumerate(choices)]}, ensure_ascii=False))
    ids = _render(tokenizer, content)
    if len(ids) > max_length:
        raise InputTooLong('Input exceeds model context window')
    return ids


def encode(prompt_format, tokenizer, codes, context, question, choices, max_length, mode='single_label'):
    """Token IDs of the exact model input for one question."""
    if prompt_format == 'AgenticScorer.encode_prompt':
        return encode_agentic_prompt(tokenizer, codes, context, question, choices, max_length, mode)
    if mode != 'single_label':
        raise ValueError('This checkpoint format was not trained for multi-label prediction')
    return encode_head_prompt(tokenizer, codes, context, question, choices, max_length)
