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
        for package, message in (({'saina_checkpoint_format': 3}, 'Unsupported Helm checkpoint format'),
                                 ({'saina_checkpoint_format': '2'}, 'Unsupported Helm checkpoint format'),
                                 ({'saina_checkpoint_format': 2, 'saina_head_type': 'linear'}, 'Conflicting'),
                                 ({'model_type': 'qwen3_5'}, 'does not match')):
            with tempfile.TemporaryDirectory() as d:
                write_checkpoint(Path(d), shared=True, package=package)
                with self.assertRaisesRegex(ValueError, message):
                    checkpoint_format(Path(d))

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


class SharedPromptTests(unittest.TestCase):
    def test_slots_follow_all_questions_and_limit_is_enforced(self):
        from saina.errors import InputTooLong
        from saina.loader import encode_shared_prompt

        class Tok:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]['content']
            def encode(self, text, add_special_tokens=False):
                return list(text.encode())
        questions = [{'question': 'a?', 'choices': ['x', 'y'], 'mode': 'single_label'},
                     {'question': 'b?', 'choices': ['x', 'y', 'z'], 'mode': 'multi_label'}]
        ids, positions = encode_shared_prompt(Tok(), ['A', 'B', 'C'], 'ctx', questions, 10_000)
        self.assertEqual(len(positions), 2)
        self.assertEqual(bytes(ids).decode().split('\nDecision ')[1:], ['q0:', 'q1:'])
        self.assertEqual(positions[-1], len(ids) - 1)
        with self.assertRaises(InputTooLong):
            encode_shared_prompt(Tok(), ['A', 'B', 'C'], 'ctx', questions, len(ids) - 1)
        with self.assertRaises(ValueError):
            encode_shared_prompt(Tok(), ['A', 'B'], 'ctx', questions, 10_000)


if __name__ == '__main__': unittest.main()
