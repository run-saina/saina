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


def write_checkpoint(root, weights=True, frozen=False):
    codes = [chr(c) for c in range(ord('A'), ord('Z')+1)]
    (root / 'head-config.json').write_text(json.dumps({
        'prompt': 'AgenticScorer.encode_prompt', 'model': 'Qwen/Qwen3.5-0.8B', 'revision': REVISION,
        'frozen_backbone': frozen, 'supported_modes': ['single_label', 'multi_label'],
        'codes': codes, 'code_ids': [ord(c) for c in codes]}))
    if weights:
        (root / 'model.safetensors').write_bytes(b'')


@unittest.skipIf(torch is None, 'requires the local extras (torch, transformers)')
class LoaderTests(unittest.TestCase):
    def load(self, root):
        from saina.loader import FrozenHeadScorer
        model = MagicMock()
        model.lm_head.weight.shape = (10, 4)
        backbone = MagicMock()
        backbone.from_pretrained.return_value.to.return_value.eval.return_value.requires_grad_.return_value = model
        tokenizer = MagicMock()
        tokenizer.from_pretrained.return_value = FakeTokenizer()
        with patch('transformers.AutoTokenizer', tokenizer), \
                patch('transformers.Qwen3_5ForConditionalGeneration', backbone), \
                patch('safetensors.torch.load_file', return_value={'weight': torch.zeros(26, 4)}):
            scorer = FrozenHeadScorer(str(root), device='cpu')
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


if __name__ == '__main__': unittest.main()
