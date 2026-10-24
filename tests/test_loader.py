import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import torch
except ImportError:  # local extras not installed
    torch = None

REVISION = 'a' * 40


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(text[0])] if len(text) == 1 else [1, 2]


def write_checkpoint(root, weights=True, frozen=False, shared=False, package=None, **extra):
    codes = [chr(c) for c in range(ord('A'), ord('Z')+1)]
    config = {
        'prompt': 'shared_context_query_slots_v1' if shared else 'AgenticScorer.encode_prompt',
        'model': 'Qwen/Qwen3.5-0.8B', 'revision': REVISION,
        'frozen_backbone': frozen, 'supported_modes': ['single_label', 'multi_label'],
        'codes': codes, 'code_ids': [ord(c) for c in codes]}
    if shared:
        config.update(head_type='routed_residual_v1', experts=3, bottleneck=2, route_progress=.26, max_length=262144)
    (root / 'head-config.json').write_text(json.dumps({**config, **extra}))
    if package is None and shared:
        package = {'model_type': 'qwen3_5', 'saina_checkpoint_format': 2,
                   'saina_prompt_format': config['prompt'], 'saina_head_type': 'routed_residual_v1',
                   'shared_trunk_pass': True}
    if package is not None:
        (root / 'config.json').write_text(json.dumps(package))
    if weights:
        (root / 'model.safetensors').write_bytes(b'')


def write_multimodal(root, version=4, attention='bidirectional_prefix', package_attention=None, head_attention=None,
                     inputs=('text', 'image', 'audio')):
    """A format 3 or 4 package: the shared multimodal prompt, a routed head and, for format 4, prompt attention."""
    package = {'model_type': 'qwen3_5', 'saina_checkpoint_format': version,
               'saina_prompt_format': 'shared_multimodal_native_processor_v2', 'saina_head_type': 'routed_residual_v1',
               'shared_trunk_pass': True, 'saina_inputs': list(inputs)}
    extra = {}
    if package_attention or attention != 'causal':
        package['saina_prompt_attention'] = package_attention or attention
    if (head_attention or attention) != 'causal':
        extra['prompt_attention'] = head_attention or attention
    write_checkpoint(root, shared=True, package=package, **extra)
    config = json.loads((root / 'head-config.json').read_text())
    config['prompt'] = 'shared_multimodal_native_processor_v2'
    (root / 'head-config.json').write_text(json.dumps(config))


@unittest.skipIf(torch is None, 'requires the local extras (torch, transformers)')
class LoaderTests(unittest.TestCase):
    def load(self, root, loader=None, weights=None):
        from saina.loader import FrozenHeadScorer
        loader = loader or FrozenHeadScorer
        model = MagicMock()
        model.lm_head.weight.shape = (10, 4)
        backbone = MagicMock()
        backbone.from_pretrained.return_value.to.return_value.eval.return_value.requires_grad_.return_value = model
        tokenizer = MagicMock()
        tokenizer.from_pretrained.return_value = FakeTokenizer()
        with patch('transformers.AutoTokenizer', tokenizer), \
                patch('transformers.Qwen3_5ForConditionalGeneration', backbone), \
                patch('safetensors.torch.load_file', side_effect=weights or [{'weight': torch.zeros(26, 4)}]):
            scorer = loader(str(root), device='cpu')
        return scorer, tokenizer, backbone

    def test_bundled_weights_load_from_checkpoint_not_base_model(self):
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d))
            scorer, tokenizer, backbone = self.load(d)
        tokenizer.from_pretrained.assert_called_once_with(d, revision=None, trust_remote_code=False)
        self.assertEqual(backbone.from_pretrained.call_args.args, (d,))
        self.assertIsNone(backbone.from_pretrained.call_args.kwargs['revision'])
        self.assertEqual(scorer.supported_modes, ['single_label', 'multi_label'])

    def test_frozen_head_still_loads_pinned_base_model(self):
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d), weights=False, frozen=True)
            _, tokenizer, backbone = self.load(d)
        tokenizer.from_pretrained.assert_called_once_with('Qwen/Qwen3.5-0.8B', revision=REVISION, trust_remote_code=False)
        self.assertEqual(backbone.from_pretrained.call_args.kwargs['revision'], REVISION)

    def test_unfrozen_head_without_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d), weights=False)
            with self.assertRaisesRegex(ValueError, 'Unsupported or unpinned'):
                self.load(d)

    def test_legacy_context_default_and_override(self):
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d))
            self.assertEqual(self.load(d)[0].max_length, 8192)
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d), max_length=32768)
            self.assertEqual(self.load(d)[0].max_length, 32768)

    def test_format_dispatch(self):
        from saina.loader import checkpoint_format
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d))
            self.assertEqual(checkpoint_format(Path(d)), 1)
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d), shared=True)
            self.assertEqual(checkpoint_format(Path(d)), 2)
        for package, message in (({'saina_checkpoint_format': 5}, 'Unsupported Helm checkpoint format'),
                                 ({'saina_checkpoint_format': 3}, 'does not match'),
                                 ({'saina_checkpoint_format': 4}, 'does not match'),
                                 ({'saina_checkpoint_format': 2, 'saina_prompt_attention': 'bidirectional_prefix'},
                                  'prompt attention'),
                                 ({'saina_checkpoint_format': '2'}, 'Unsupported Helm checkpoint format'),
                                 ({'saina_checkpoint_format': 2, 'saina_head_type': 'linear'}, 'Conflicting'),
                                 ({'model_type': 'qwen3_5'}, 'does not match')):
            with tempfile.TemporaryDirectory() as d:
                write_checkpoint(Path(d), shared=True, package=package)
                with self.assertRaisesRegex(ValueError, message):
                    checkpoint_format(Path(d))

    def test_format_4_is_format_3_with_prompt_attention(self):
        from saina.loader import checkpoint_format
        with tempfile.TemporaryDirectory() as d:
            write_multimodal(Path(d), version=3, attention='causal')
            self.assertEqual(checkpoint_format(Path(d)), 3)
        with tempfile.TemporaryDirectory() as d:
            write_multimodal(Path(d))
            self.assertEqual(checkpoint_format(Path(d)), 4)
        # Each half of the metadata must agree with the format, and an unknown attention is refused.
        for kwargs in ({'version': 4, 'attention': 'causal'},
                       {'version': 4, 'head_attention': 'causal'},
                       {'version': 4, 'package_attention': 'causal'},
                       {'version': 3},
                       {'version': 4, 'attention': 'sliding_window'},
                       {'version': 4, 'head_attention': 'sliding_window'}):
            with tempfile.TemporaryDirectory() as d:
                write_multimodal(Path(d), **kwargs)
                with self.assertRaisesRegex(ValueError, 'prompt attention'):
                    checkpoint_format(Path(d))

    def test_format_4_reads_images_and_audio(self):
        from saina.loader import checkpoint_inputs
        for version in (3, 4):
            with tempfile.TemporaryDirectory() as d:
                write_multimodal(Path(d), version=version, attention='causal' if version == 3 else 'bidirectional_prefix')
                self.assertEqual(checkpoint_inputs(Path(d)), ('text', 'image', 'audio'))
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d), shared=True, package={'saina_checkpoint_format': 2, 'saina_inputs': ['text', 'image']})
            with self.assertRaisesRegex(ValueError, 'Only format 3 and 4'):
                checkpoint_inputs(Path(d))

    def test_shared_scorer_refuses_unknown_prompt_attention(self):
        from saina.loader import SharedHeadScorer
        experts = {'shared.weight': torch.zeros(26, 4), 'norm.weight': torch.ones(4), 'norm.bias': torch.zeros(4),
                   'router.weight': torch.zeros(3, 4), 'down': torch.zeros(3, 2, 4), 'up': torch.zeros(3, 26, 2)}
        for attention, ok in (('bidirectional_prefix', True), ('sliding_window', False)):
            with tempfile.TemporaryDirectory() as d:
                write_checkpoint(Path(d), shared=True, prompt_attention=attention)
                if ok:
                    scorer, _, _ = self.load(d, loader=SharedHeadScorer,
                                             weights=[{'weight': torch.zeros(26, 4)}, experts])
                    self.assertEqual(scorer.prompt_attention, attention)
                else:
                    with self.assertRaisesRegex(ValueError, 'Unsupported prompt attention'):
                        self.load(d, loader=SharedHeadScorer, weights=[{'weight': torch.zeros(26, 4)}, experts])

    def test_helm_3_names_itself(self):
        from saina.loader import checkpoint_model_id
        with tempfile.TemporaryDirectory() as d:
            write_multimodal(Path(d))
            config = json.loads((Path(d) / 'config.json').read_text())
            (Path(d) / 'config.json').write_text(json.dumps({**config, 'saina_model': 'saina-helm-3-0.8b'}))
            self.assertEqual(checkpoint_model_id(Path(d)), 'saina-helm-3-0.8b')

    def test_shared_checkpoint_loads_experts_and_its_context(self):
        from saina.loader import SharedHeadScorer
        experts = {'shared.weight': torch.zeros(26, 4), 'norm.weight': torch.ones(4), 'norm.bias': torch.zeros(4),
                   'router.weight': torch.zeros(3, 4), 'down': torch.zeros(3, 2, 4), 'up': torch.zeros(3, 26, 2)}
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d), shared=True)
            scorer, _, _ = self.load(d, loader=SharedHeadScorer, weights=[{'weight': torch.zeros(26, 4)}, experts])
        self.assertTrue(scorer.shared_trunk_pass)
        self.assertEqual(scorer.max_length, 262144)
        self.assertEqual(scorer.head.progress, .26)
        self.assertFalse(hasattr(scorer.head, 'last_features'))

    def test_checkpoint_names_itself(self):
        from saina.loader import checkpoint_model_id
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d))
            self.assertEqual(checkpoint_model_id(Path(d)), 'saina-helm-0.8b')
        for name, ok in (('saina-helm-2-0.8b', True), ('saina-helm-2-4b', True), ('../x', False), ('gpt-4', False)):
            with tempfile.TemporaryDirectory() as d:
                write_checkpoint(Path(d), package={'saina_model': name})
                if ok:
                    self.assertEqual(checkpoint_model_id(Path(d)), name)
                else:
                    with self.assertRaises(ValueError):
                        checkpoint_model_id(Path(d))

    def test_merged_specialist_answers_only_as_itself(self):
        from types import SimpleNamespace
        from saina.contract import ModelNotServed, specialist_of
        from saina.loader import standalone_specialist
        with tempfile.TemporaryDirectory() as d:
            plain = SimpleNamespace(model_id='saina-helm-2-omni-0.8b')
            write_checkpoint(Path(d), package={'saina_model': 'saina-helm-2-omni-0.8b'})
            standalone_specialist(plain, Path(d), 'b' * 40)
            self.assertFalse(hasattr(plain, 'specialist'))
            write_checkpoint(Path(d), package={'saina_model': 'saina-helm-2-omni-0.8b', 'saina_specialist': 'mechanical',
                                               'saina_specialist_description': 'Machines'})
            s = SimpleNamespace(model_id='saina-helm-2-omni-0.8b')
            standalone_specialist(s, Path(d), 'b' * 40)
            self.assertEqual((s.specialist, s.specialist_revision, s.specialist_description), ('mechanical', 'b' * 40, 'Machines'))
            self.assertIs(specialist_of(s, 'mechanical'), s)
            with self.assertRaises(ModelNotServed):
                specialist_of(s, None)
            write_checkpoint(Path(d), package={'saina_model': 'saina-helm-2-omni-0.8b', 'saina_specialist': '../x'})
            with self.assertRaises(ValueError):
                standalone_specialist(SimpleNamespace(), Path(d), None)

    def test_shared_checkpoint_requires_merged_weights(self):
        from saina.loader import SharedHeadScorer
        with tempfile.TemporaryDirectory() as d:
            write_checkpoint(Path(d), shared=True, weights=False, requires_adapter='adapter')
            with self.assertRaisesRegex(ValueError, 'merged backbone'):
                SharedHeadScorer(d, device='cpu')


@unittest.skipIf(torch is None, 'requires the local extras (torch, transformers)')
class RoutedHeadTests(unittest.TestCase):
    def test_saved_schedule_routes_and_includes_residuals(self):
        from saina.routed import RoutedResidualHead
        torch.manual_seed(0)
        head = RoutedResidualHead(torch.nn.Linear(4, 5, bias=False), experts=3, bottleneck=2, progress=.26)
        for parameter in head.parameters():
            torch.nn.init.normal_(parameter)
        h = torch.randn(2, 4)
        p = head.routes(h)
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(2)))
        self.assertTrue(((p > 0).sum(-1) == 2).all())  # progress >= .2 means fully top-two
        expected = head.shared(h) + torch.einsum('ne,nec->nc', p, torch.einsum(
            'neb,ecb->nec', torch.nn.functional.gelu(torch.einsum('nh,ebh->neb', h, head.down)), head.up))
        self.assertTrue(torch.allclose(head(h), expected, atol=1e-5))


@unittest.skipIf(torch is None, 'requires the local extras (torch, transformers)')
class PromptAttentionTests(unittest.TestCase):
    """Format 4: prompt tokens read both ways in the full-attention layers; readout slots stay causal."""
    @classmethod
    def setUpClass(cls):
        from transformers import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
        config = Qwen3_5TextConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                   layer_types=['full_attention', 'full_attention'])
        torch.manual_seed(0)
        cls.model = Qwen3_5TextModel(config).eval()
        cls.ids = torch.randint(0, 64, (1, 10))

    def hidden(self, ids, mode, prompt=6):
        from saina.attention import prompt_attention
        with torch.no_grad(), prompt_attention(mode, prompt):
            return self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0]

    def changed(self, mode, position):
        ids = self.ids.clone()
        ids[0, position] = (ids[0, position] + 1) % 64
        return (self.hidden(ids, mode) - self.hidden(self.ids, mode)).abs().amax(-1) > 1e-6

    def test_prompt_tokens_see_later_prompt_tokens(self):
        self.assertFalse(self.changed('causal', 4)[:4].any())
        self.assertTrue(self.changed('bidirectional_prefix', 4)[:4].all())

    def test_readout_slots_stay_causal(self):
        # Changing a slot token reaches neither the prompt nor the slot tokens before it.
        changed = self.changed('bidirectional_prefix', 8)
        self.assertFalse(changed[:8].any())
        self.assertTrue(changed[8:].all())

    def test_mask_is_scoped_to_the_block(self):
        bidirectional = self.hidden(self.ids, 'bidirectional_prefix')
        causal = self.hidden(self.ids, 'causal')
        self.assertFalse(torch.allclose(bidirectional[:6], causal[:6]))
        with torch.no_grad():
            after = self.model(input_ids=self.ids, attention_mask=torch.ones_like(self.ids)).last_hidden_state[0]
        self.assertTrue(torch.equal(after, causal))

    def test_unknown_prompt_attention_is_refused(self):
        from saina.attention import prompt_attention
        with self.assertRaisesRegex(ValueError, 'Unsupported prompt attention'):
            with prompt_attention('sliding_window', 6):
                pass


class SharedPromptTests(unittest.TestCase):
    def test_slots_follow_all_questions_and_limit_is_enforced(self):
        from saina.errors import InputTooLong
        from saina.loader import encode_shared_prompt, shared_positions, shared_prompt_tokens

        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]['content']
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())
        questions = [('a?', ['x', 'y'], 'single_label'), ('b?', ['x', 'y', 'z'], 'multi_label')]
        ids = encode_shared_prompt(Tok(), ['A', 'B', 'C'], 'ctx', questions, 10_000)
        self.assertEqual(bytes(ids).decode().split('\nDecision ')[1:], ['q0:', 'q1:'])
        # Slot positions are recovered from the IDs alone: the last token of each slot.
        positions = shared_positions(Tok(), len(ids), 2)
        self.assertEqual(positions[-1], len(ids) - 1)
        self.assertEqual(bytes(ids[:positions[0] + 1]).decode().rsplit('\n', 1)[1], 'Decision q0:')
        # Format 4 reads both ways every token before the first slot.
        prompt = shared_prompt_tokens(Tok(), len(ids), 2)
        self.assertEqual(bytes(ids[prompt:]).decode(), '\nDecision q0:\nDecision q1:')
        with self.assertRaises(InputTooLong):
            encode_shared_prompt(Tok(), ['A', 'B', 'C'], 'ctx', questions, len(ids) - 1)
        with self.assertRaises(ValueError):
            encode_shared_prompt(Tok(), ['A', 'B'], 'ctx', questions, 10_000)

if __name__ == '__main__': unittest.main()
