import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
try:
    import torch
    from transformers import pipeline
    from saina.transformers import register, HelmConfig
except ImportError:  # local extras not installed
    torch = None


class FakeScorer:
    def __init__(self):
        self.model = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)
    def predict(self, context, question, choices, mode):
        return [1/len(choices)]*len(choices) if mode == 'single_label' else [.8]*len(choices)


@unittest.skipIf(torch is None, 'requires the local extras (torch, transformers)')
class IntegrationTests(unittest.TestCase):
    def test_factory_local_and_both_modes(self):
        register(); register()
        with tempfile.TemporaryDirectory() as d:
            HelmConfig().save_pretrained(d)
            with patch('saina.loader.load_scorer', return_value=FakeScorer()):
                classifier = pipeline('saina-classification', model=d, device='cpu')
            request = dict(task='Choose', input='text', choices=['a', 'b'])
            self.assertEqual(classifier(request), [.5, .5])
            self.assertEqual(classifier([{**request, 'mode':'multi_label'}, request]), [[.8,.8],[.5,.5]])
            with self.assertRaises(ValueError): classifier({**request, 'mode':'wrong'})
            with self.assertRaises(ValueError): classifier({**request, 'choices':['a']})

if __name__ == '__main__': unittest.main()
