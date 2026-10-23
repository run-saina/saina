"""`POST /v1/decisions`: the OpenAI Decisions request shape and the System One body on one route.

The OpenAI dialect is a wire adapter over the native `/v1/ask` contract. A request is translated into
a native request body (validated, prepared, billed and scored exactly as `/v1/ask`), and the native
answers are translated back. Nothing about scoring changes, so the same question asked natively and
through this dialect gets the same probabilities and the same bill.

Shapes follow the official OpenAI SDKs (openai 3.37.0 on PyPI, 7.43.0 on npm) and the Decisions guide.
What Saina cannot read faithfully is refused with a message naming the field, before anything is
charged: image URLs, images inside question instructions, prompt cache controls are ignored only
where ignoring them cannot change an answer (see IGNORED_FIELDS).
"""
import json
import re

DIALECT_HEADER = 'x-saina-dialect'
DIALECTS = ('openai', 'systemone')
QUESTION_TYPES = ('predicate', 'choice', 'score')

# Request fields of the OpenAI dialect. Saina extensions (decision policy and specialists) keep their
# native names and meaning; OpenAI SDK callers send them with `extra_body`.
OPENAI_FIELDS = {'model', 'input', 'questions'}
SAINA_EXTENSIONS = {'mode', 'threshold', 'min_margin', 'specialist'}
# Accepted and ignored: they control prompt caching, abuse attribution or client bookkeeping, never an answer.
# Saina has no prompt cache, so usage reports zero cached and cache-write tokens.
IGNORED_FIELDS = {'prompt_cache_key', 'prompt_cache_options', 'safety_identifier', 'metadata'}
# Generation controls have no meaning for a model that scores options instead of generating text.
SAMPLING_FIELDS = {'temperature', 'top_p', 'top_k', 'seed', 'n', 'stream', 'stream_options', 'max_tokens',
                   'max_output_tokens', 'max_completion_tokens', 'logprobs', 'top_logprobs', 'frequency_penalty',
                   'presence_penalty', 'logit_bias', 'stop', 'reasoning', 'reasoning_effort'}
# Native or System One fields that make a body ambiguous when they appear next to OpenAI fields.
FOREIGN_TOP = ('state', 'images', 'audios', 'videos', 'assets')
FOREIGN_QUESTION = ('criteria', 'question', 'options', 'descriptions')

_IMAGE_DATA_URL = re.compile(r'data:image/[a-z0-9.+-]+(;[a-z0-9_.+-]+=[^;,]*)*;base64,', re.I)


class DialectError(ValueError):
    """A request this route cannot read. The message names the field, never its value."""


def detect_dialect(body, header=None):
    """`openai` or `systemone` from the body's shape. An `X-Saina-Dialect` header may confirm it but never
    overrides the body: a header that disagrees is an error, as is a body that matches both shapes or neither."""
    if not isinstance(body, dict):
        raise DialectError('Request body must be a JSON object')
    questions = body.get('questions')
    if 'input' in body and 'state' in body:
        raise DialectError('Send either `input` (OpenAI Decisions) or `state` (System One), not both')
    if 'input' in body or isinstance(questions, list):
        found = [f for f in FOREIGN_TOP if f in body]
        if found:
            raise DialectError(f'`{found[0]}` is not an OpenAI Decisions field; this body also has '
                               + ('`input`' if 'input' in body else 'a `questions` array'))
        if isinstance(questions, list):
            for i, q in enumerate(questions):
                bad = [f for f in FOREIGN_QUESTION if isinstance(q, dict) and f in q]
                if bad:
                    raise DialectError(f'questions[{i}].{bad[0]} is not an OpenAI Decisions field '
                                       '(choices, levels and instructions are)')
        dialect = 'openai'
    elif 'state' in body and isinstance(questions, dict):
        dialect = 'systemone'
    elif 'state' in body:
        raise DialectError('System One requests need `questions` as an object keyed by question name')
    else:
        raise DialectError('Send `input` with a `questions` array (OpenAI Decisions) '
                           'or `state` with a `questions` object (System One)')
    if header is not None:
        named = header.strip().lower()
        if named not in DIALECTS:
            raise DialectError('X-Saina-Dialect must be openai or systemone')
        if named != dialect:
            raise DialectError(f'X-Saina-Dialect says {named} but the body is a {dialect} request')
    return dialect


# -- OpenAI request to native -----------------------------------------------------------------------------

class Translation:
    """A native request body plus what is needed to answer in the caller's vocabulary."""

    def __init__(self, native, names, kinds, values, image_paths):
        self.native = native            # dict for saina.contract.AskRequest
        self.names = names              # per question in order: the caller's name or None
        self.kinds = kinds              # per question: predicate | choice | score
        self.values = values            # per question: choice values as sent (typed) or score level labels
        self.image_paths = image_paths  # where each native image came from, e.g. input[0].content[1]

    @property
    def keys(self):
        return list(self.native['questions'])


def _object(value, where):
    if not isinstance(value, dict):
        raise DialectError(f'{where} must be an object')
    return value


def _string(value, where, *, empty=False):
    if not isinstance(value, str):
        raise DialectError(f'{where} must be a string')
    if not empty and not value.strip():
        raise DialectError(f'{where} must not be empty')
    return value


def _only(obj, allowed, where):
    extra = sorted(set(obj) - set(allowed))
    if extra:
        raise DialectError(f'{where}.{extra[0]} is not supported')


def _image(part, where, images, paths):
    _only(part, {'type', 'image_url', 'detail', 'prompt_cache_breakpoint'}, where)
    url = _string(part.get('image_url'), f'{where}.image_url')
    if part.get('detail') not in (None, 'auto'):
        # Saina reads every image at one resolution; another detail level would be silently ignored.
        raise DialectError(f'{where}.detail must be auto or omitted')
    if not _IMAGE_DATA_URL.match(url):
        # Hosted URLs are never fetched (and file IDs do not exist here): an image Saina cannot read is refused,
        # never answered from its text.
        raise DialectError(f'{where}.image_url must be a base64 data URL (data:image/png|jpeg|webp;base64,...); '
                           'image URLs are not fetched')
    images.append(url)
    paths.append(where)


def _json_part(part, where):
    _only(part, {'type', 'data'}, where)
    data = part.get('data')
    if not isinstance(data, dict) and not (isinstance(data, list) and all(isinstance(x, dict) for x in data)):
        raise DialectError(f'{where}.data must be a JSON object or an array of objects')
    return data


def _parts(content, where, allowed, images, paths):
    """Text and JSON parts in order (images go to `images`); a single text part collapses to a string."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or not content:
        raise DialectError(f'{where} must be a string or a non-empty array of content parts')
    out = []
    for j, part in enumerate(content):
        at = f'{where}[{j}]'
        kind = _object(part, at).get('type')
        if kind not in allowed:
            raise DialectError(f'{at}.type must be one of: ' + ', '.join(allowed))
        if kind in ('input_text', 'output_text'):
            _only(part, {'type', 'text', 'prompt_cache_breakpoint'}, at)
            out.append(_string(part.get('text'), f'{at}.text', empty=True))
        elif kind == 'refusal':
            _only(part, {'type', 'refusal'}, at)
            out.append({'refusal': _string(part.get('refusal'), f'{at}.refusal', empty=True)})
        elif kind == 'input_json':
            out.append(_json_part(part, at))
        else:
            _image(part, at, images, paths)
    if len(out) == 1 and isinstance(out[0], str):
        return out[0]
    return out


MESSAGE_PARTS = {'user': ('input_text', 'input_image', 'input_json'), 'developer': ('input_text',),
                 'system': ('input_text',), 'assistant': ('output_text', 'refusal')}
TOOL_OUTPUT_PARTS = ('input_text', 'input_image')


def _input(value, images, paths):
    """Native state: a string as sent, or one entry per input item in order."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise DialectError('input must be a string or a non-empty array of messages and tool items')
    state, calls = [], set()
    for i, item in enumerate(value):
        at = f'input[{i}]'
        _object(item, at)
        kind = item.get('type', 'message')
        if kind == 'message':
            _only(item, {'type', 'role', 'content', 'phase'}, at)
            role = item.get('role')
            if role not in MESSAGE_PARTS:
                raise DialectError(f'{at}.role must be one of: user, developer, system, assistant')
            entry = {'role': role, 'content': _parts(item.get('content'), f'{at}.content', MESSAGE_PARTS[role],
                                                     images, paths)}
        elif kind in ('function_call', 'custom_tool_call'):
            field = 'arguments' if kind == 'function_call' else 'input'
            _only(item, {'type', 'call_id', 'name', field, 'id', 'namespace', 'status'}, at)
            call = _string(item.get('call_id'), f'{at}.call_id')
            calls.add((kind, call))
            entry = {'type': kind, 'call_id': call, 'name': _string(item.get('name'), f'{at}.name'),
                     field: _string(item.get(field), f'{at}.{field}', empty=True)}
            if item.get('namespace') is not None:
                entry['namespace'] = _string(item['namespace'], f'{at}.namespace')
        elif kind in ('function_call_output', 'custom_tool_call_output'):
            _only(item, {'type', 'call_id', 'output', 'id', 'status'}, at)
            call = _string(item.get('call_id'), f'{at}.call_id')
            if (kind.removesuffix('_output'), call) not in calls:
                raise DialectError(f'{at}.call_id does not match an earlier {kind.removesuffix("_output")}')
            entry = {'type': kind, 'call_id': call,
                     'output': _parts(item.get('output'), f'{at}.output', TOOL_OUTPUT_PARTS, images, paths)}
        else:
            raise DialectError(f'{at}.type must be one of: message, function_call, function_call_output, '
                               'custom_tool_call, custom_tool_call_output')
        state.append(entry)
    return state


def _instructions(value, where):
    if isinstance(value, str):
        return _string(value, where)
    images, paths = [], []
    if isinstance(value, list):
        for j, part in enumerate(value):
            if isinstance(part, dict) and part.get('type') == 'input_image':
                # Native questions have no images of their own; reading this one as shared input would change
                # what the question compares, so it is refused.
                raise DialectError(f'{where}[{j}] is an image; images in question instructions are not supported, '
                                   'put shared images in input')
    rendered = _parts(value, where, ('input_text', 'input_json'), images, paths)
    if isinstance(rendered, str) and not rendered.strip():
        raise DialectError(f'{where} must not be empty')
    return rendered


def _choice_key(value, where):
    # Choice values are typed: a string and a boolean with the same text are different values, but the
    # native options are keyed by text, so such a pair cannot be told apart and is refused below.
    if isinstance(value, bool):
        return 'true' if value else 'false'
    return _string(value, where)


def _question(q, i, images, paths):
    at = f'questions[{i}]'
    _object(q, at)
    kind = q.get('type')
    if kind not in QUESTION_TYPES:
        raise DialectError(f'{at}.type must be one of: predicate, choice, score')
    common = {'type', 'name', 'instructions', 'threshold', 'min_margin'}
    native = {'question': _instructions(q.get('instructions'), f'{at}.instructions')}
    for policy in ('threshold', 'min_margin'):
        if policy in q:
            native[policy] = q[policy]
    values = None
    if kind == 'predicate':
        _only(q, common, at)
        native['type'] = 'yes_no'
    elif kind == 'choice':
        _only(q, common | {'choices'}, at)
        choices = q.get('choices')
        if not isinstance(choices, list):
            raise DialectError(f'{at}.choices must be an array')
        options, values = {}, []
        for j, c in enumerate(choices):
            cat = f'{at}.choices[{j}]'
            _only(_object(c, cat), {'value', 'description'}, cat)
            key = _choice_key(c.get('value'), f'{cat}.value')
            if key in options:
                raise DialectError(f'{cat}.value repeats an earlier choice'
                                   + (' (a string and a boolean with the same text cannot both be used)'
                                      if values[list(options).index(key)] != c.get('value') else ''))
            description = c.get('description')
            options[key] = None if description is None else _string(description, f'{cat}.description', empty=True)
            values.append(c['value'])
        native.update(type='single_choice', options=options)
    else:
        _only(q, common - {'min_margin'} | {'levels'}, at)
        levels = q.get('levels')
        if not isinstance(levels, list):
            raise DialectError(f'{at}.levels must be an array')
        rendered, values = [], []
        for j, level in enumerate(levels):
            lat = f'{at}.levels[{j}]'
            _only(_object(level, lat), {'label', 'description'}, lat)
            label = _string(level.get('label'), f'{lat}.label')
            description = level.get('description')
            rendered.append(label if description is None else
                            f'{label}: {_string(description, f"{lat}.description", empty=True)}')
            values.append(label)
        native.update(type='rating', levels=rendered)
    return native, kind, values


def translate(body):
    """Native `/v1/ask` body for an OpenAI Decisions request, or DialectError naming the first problem.
    Native limits (questions, options, levels, policy values) are checked by the native contract afterwards."""
    _object(body, 'Request body')
    sampling = sorted(set(body) & SAMPLING_FIELDS)
    if sampling:
        raise DialectError(f'{sampling[0]} is not supported: Saina scores options instead of generating text')
    _only(body, OPENAI_FIELDS | SAINA_EXTENSIONS | IGNORED_FIELDS, 'Request body')
    for field in ('model', 'input', 'questions'):
        if field not in body:
            raise DialectError(f'{field} is required')
    model = _string(body['model'], 'model')
    images, paths = [], []
    state = _input(body['input'], images, paths)
    questions = body['questions']
    if not isinstance(questions, list) or not questions:
        raise DialectError('questions must be a non-empty array')
    native_questions, names, kinds, values, seen = {}, [], [], [], {}
    for i, q in enumerate(questions):
        native, kind, vals = _question(q, i, images, paths)
        name = q.get('name')
        if name is not None:
            name = _string(name, f'questions[{i}].name')
            if name in seen:
                raise DialectError(f'questions[{i}].name repeats questions[{seen[name]}].name; names must be unique')
            seen[name] = i
        # Native keys are positions: names are optional here, and a key never reaches the model or the bill.
        native_questions[str(i)] = native
        names.append(name)
        kinds.append(kind)
        values.append(vals)
    native = {'model': model, 'state': state, 'questions': native_questions}
    for field in SAINA_EXTENSIONS - {'model'}:
        if field in body:
            native[field] = body[field]
    if images:
        native['images'] = images
    return Translation(native, names, kinds, values, paths)


def native_error_path(loc):
    """A native validation location in the caller's vocabulary: ('questions', '3', 'options') → questions[3].choices."""
    parts, out = [p for p in loc if p != 'body'], ''
    rename = {'question': 'instructions', 'options': 'choices', 'levels': 'levels', 'state': 'input'}
    for k, p in enumerate(parts):
        if isinstance(p, int) or (k == 1 and parts[0] == 'questions' and str(p).isdigit()):
            out += f'[{p}]'
        elif k >= 2 and parts[0] == 'questions' and p in ('yes_no', 'single_choice', 'rating', 'multi_choice'):
            continue  # pydantic's discriminator tag
        elif p == 'images' and not out:
            out = 'input'  # images come from input parts
            break
        else:
            out += ('.' if out else '') + rename.get(p, str(p))
    return out or 'request'


# -- native response to OpenAI ------------------------------------------------------------------------------

def _usage(tokens):
    # Saina bills input tokens only and generates none; there is no prompt cache.
    return {'input_tokens': tokens, 'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
            'output_tokens': 0, 'output_tokens_details': {'reasoning_tokens': 0}, 'total_tokens': tokens}


def _distribution(kind, answer, values):
    if kind == 'choice':
        return [{'value': v, 'probability': answer['probabilities'][_choice_key(v, '')]} for v in values]
    return [{'value': i, 'label': label, 'probability': answer['probabilities'][str(i)]}
            for i, label in enumerate(values)]


def respond(translation, native):
    """OpenAI Decisions response from a native `/v1/ask` response to `translation.native`."""
    answers = []
    for key, name, kind, values in zip(translation.keys, translation.names, translation.kinds, translation.values):
        a = native['answers'][key]
        reason = a.get('reason')
        if reason not in (None, 'accepted'):
            # A decision policy did not accept this answer. The refusal keeps the native reason and the
            # distribution (Saina extensions) so the caller can act without a second call. It is billed.
            refusal = {'type': 'refusal', 'name': name, 'reason': reason}
            if kind == 'predicate':
                refusal['probability'] = a['yes']
            else:
                refusal['probabilities'] = _distribution(kind, a, values)
            answers.append(refusal)
        elif kind == 'predicate':
            answers.append({'type': 'predicate', 'name': name, 'probability': a['yes']})
        elif kind == 'choice':
            winner = values[list(a['probabilities']).index(a['selection'])]
            answers.append({'type': 'choice', 'name': name, 'choice': winner, 'confidence': a['confidence'],
                            'probabilities': _distribution(kind, a, values)})
        else:
            answers.append({'type': 'score', 'name': name, 'score': a['expected_level'], 'confidence': a['confidence'],
                            'probabilities': _distribution(kind, a, values)})
    body = {'model': native['model'], 'answers': answers, 'usage': _usage(native['usage']['input_tokens'])}
    if native.get('specialist'):
        body['specialist'] = native['specialist']
    return body


def dumps(body):
    return json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
