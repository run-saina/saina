"""Prompt formatting shared by the inference server and the token counter. No torch import."""
import json
from itertools import product
from .errors import InputTooLong

SHARED_PROMPT = 'shared_context_query_slots_v1'
# The shared-pass prompt with images (and audio spectrograms) placed before the text of the user turn.
MULTIMODAL_PROMPT = 'shared_multimodal_native_processor_v2'
SHARED_PROMPTS = (SHARED_PROMPT, MULTIMODAL_PROMPT)
PROMPT_FORMATS = ('BaseScorer.encode_prompt', 'AgenticScorer.encode_prompt', SHARED_PROMPT, MULTIMODAL_PROMPT)
VISION_TOKENS = ('<|vision_start|>', '<|image_pad|>', '<|vision_end|>')


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


def _slot(tokenizer, i):
    return tokenizer.encode(f'\nDecision q{i}:', add_special_tokens=False)


def encode_shared_prompt(tokenizer, codes, context, questions, max_length):
    """One prompt for every question, followed by one fixed readout slot per question.

    questions: [(question, choices, mode)] in request order."""
    if not questions:
        raise ValueError('Expected at least one question')
    items = []
    for i, (question, choices, mode) in enumerate(questions):
        if not 2 <= len(choices) <= len(codes):
            raise ValueError('Unsupported option count')
        if mode not in ('single_label', 'multi_label'):
            raise ValueError('Unsupported mode')
        items.append({'id': f'q{i}', 'question': question, 'mode': mode,
                      'options': [{'code': codes[j], 'text': text} for j, text in enumerate(choices)]})
    # Keep byte-for-byte aligned with the training encoder (shared_pass_model.encode_shared).
    content = ('Use the shared context to answer every question. For single_label choose one option; '
               'for multi_label assess each option independently.\n' +
               json.dumps({'context': context, 'questions': items}, ensure_ascii=False))
    ids = _render(tokenizer, content)
    for i in range(len(items)):
        ids.extend(_slot(tokenizer, i))
    if len(ids) > max_length:
        raise InputTooLong('Input exceeds model context window; shorten it explicitly')
    return ids


def insert_images(tokenizer, ids, image_tokens, max_length):
    """Shared-prompt IDs with one vision block per image (`image_tokens` placeholders each, in order)
    inserted at the start of the user turn, before the text. Question slots stay at the end, so
    `shared_positions` still finds them. Keep aligned with the training encoder (encode_multimodal)."""
    if not image_tokens:
        return ids
    marker = tokenizer.encode('<|im_start|>user\n', add_special_tokens=False)
    start = next((i + len(marker) for i in range(len(ids) - len(marker) + 1) if ids[i:i + len(marker)] == marker), None)
    if start is None:
        raise ValueError('The chat template has no user turn')
    begin, pad, end = (tokenizer.convert_tokens_to_ids(t) for t in VISION_TOKENS)
    visual = []
    for count in image_tokens:
        visual += [begin] + [pad] * count + [end]
    ids = ids[:start] + visual + ids[start:]
    if len(ids) > max_length:
        raise InputTooLong('Input exceeds model context window; send fewer or smaller images')
    return ids


def shared_positions(tokenizer, length, count):
    """Readout positions of `count` question slots at the end of a shared prompt of `length` tokens:
    the last token of each slot, so they can be recovered from the encoded IDs alone."""
    widths = [len(_slot(tokenizer, i)) for i in range(count)]
    positions, end = [], length - sum(widths)
    for width in widths:
        end += width
        positions.append(end - 1)
    return positions


def encode(prompt_format, tokenizer, codes, context, question, choices, max_length, mode='single_label'):
    """Token IDs of the exact model input for one question."""
    if prompt_format in SHARED_PROMPTS:
        return encode_shared_prompt(tokenizer, codes, context, [(question, choices, mode)], max_length)
    if prompt_format == 'AgenticScorer.encode_prompt':
        return encode_agentic_prompt(tokenizer, codes, context, question, choices, max_length, mode)
    if mode != 'single_label':
        raise ValueError('This checkpoint format was not trained for multi-label prediction')
    return encode_head_prompt(tokenizer, codes, context, question, choices, max_length)
