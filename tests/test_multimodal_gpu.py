"""Integration tests on a real multimodal Helm package (GPU).

    SAINA_TEST_MULTIMODAL_CHECKPOINT=/path/to/package SAINA_TEST_DEVICE=cuda:0 pytest -m gpu tests/test_multimodal_gpu.py
"""
import base64
import io
import os
import unittest
from pathlib import Path

try:
    import pytest
    pytestmark = pytest.mark.gpu
except ImportError:
    pass

CHECKPOINT = os.environ.get('SAINA_TEST_MULTIMODAL_CHECKPOINT')
DEVICE = os.environ.get('SAINA_TEST_DEVICE', 'cuda:0')
try:
    import torch
    HAVE_GPU = torch.cuda.is_available()
except ImportError:
    HAVE_GPU = False

QUESTIONS = {
    'kind': {'type': 'single_choice', 'question': 'What kind of image is this?',
             'options': {'photo': 'A photograph', 'chart': 'A chart or plot', 'document': 'A document'}},
    'text': {'type': 'yes_no', 'question': 'Is there readable text?'},
    'colors': {'type': 'multi_choice', 'question': 'Which colors are prominent?',
               'options': {'red': None, 'green': None, 'blue': None}},
}


def png_url(size=(320, 240), color=(220, 20, 20)):
    from PIL import Image, ImageDraw
    image = Image.new('RGB', size, color)
    ImageDraw.Draw(image).rectangle([20, 20, size[0] // 2, size[1] // 2], fill=(20, 20, 220))
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def wav_url(seconds=2.0):
    import numpy as np
    from saina.media import audio_data_url
    t = np.arange(int(16000 * seconds)) / 16000
    return audio_data_url((0.3 * np.sin(2 * np.pi * 440 * t), 16000))


@unittest.skipUnless(CHECKPOINT and Path(CHECKPOINT).is_dir() and HAVE_GPU,
                     'set SAINA_TEST_MULTIMODAL_CHECKPOINT to a multimodal package and run on a GPU')
class MultimodalPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from saina.tokenize import TokenCounter
        from saina.transformers import HelmModel
        cls.model = HelmModel.from_pretrained(CHECKPOINT, device=DEVICE)
        cls.counter = TokenCounter.from_checkpoint(CHECKPOINT)
        cls.name = cls.model.scorer.model_id

    def request(self, **fields):
        return {'model': self.name, 'state': 'A customer attached this.', 'questions': QUESTIONS, **fields}

    def test_images_answer_every_question_and_usage_counts_image_tokens(self):
        from saina.contract import AskRequest
        self.assertIn('image', self.model.inputs)
        request = self.request(images=[png_url(), png_url((1800, 1200), (20, 200, 20))])
        result = self.model.ask(**request)
        self.assertEqual(set(result['answers']), set(QUESTIONS))
        text = self.model.ask(**self.request())
        policy = self.model.scorer.image_policy
        self.assertEqual(result['usage']['input_tokens'] - text['usage']['input_tokens'],
                         policy.tokens(320, 240) + policy.tokens(1800, 1200))
        # The gateway's torch-free counter gives the exact same numbers before anything runs.
        parsed = AskRequest.model_validate(request)
        self.assertEqual(self.counter.bill_ask(parsed).total, result['usage']['input_tokens'])
        self.assertEqual(self.counter.bill_ask(AskRequest.model_validate(self.request())).total,
                         text['usage']['input_tokens'])

    def test_paths_and_pil_images_are_accepted_locally(self):
        import tempfile
        from PIL import Image
        image = Image.new('RGB', (256, 256), (30, 30, 200))
        with tempfile.TemporaryDirectory() as d:
            image.save(Path(d) / 'a.jpg')
            a = self.model.ask(**self.request(images=[str(Path(d) / 'a.jpg')]))
        b = self.model.ask(**self.request(images=[image]))
        self.assertEqual(a['usage'], b['usage'])

    def test_server_usage_and_metering_equal_the_counter(self):
        from fastapi.testclient import TestClient
        from saina.contract import AskRequest
        from saina.jev import SystemOneRequest
        from saina.server import create_app
        audio = 'audio' in self.model.inputs
        fields = {'images': [png_url(), png_url((640, 480))], **({'audios': [wav_url()]} if audio else {})}
        with TestClient(create_app(scorer=self.model.scorer, service_token='svc')) as c:
            body = self.request(**fields)
            r = c.post('/v1/ask', headers={'Authorization': 'Bearer svc'}, json=body)
            self.assertEqual(r.status_code, 200, r.text)
            parsed = AskRequest.model_validate(body)
            self.assertEqual(r.json()['usage']['input_tokens'], self.counter.bill_ask(parsed).total)
            self.assertEqual(int(r.headers['x-saina-input-tokens']), sum(self.counter.count_ask(parsed)))
            self.assertEqual(int(r.headers['x-saina-media-tokens']), self.counter.bill_ask(parsed).media_tokens)
            # The gateway's expected-token header is honored for media requests.
            r2 = c.post('/v1/ask', json=body, headers={'Authorization': 'Bearer svc',
                        'X-Saina-Expected-Input-Tokens': str(sum(self.counter.count_ask(parsed)))})
            self.assertEqual(r2.status_code, 200, r2.text)
            s1 = {'model': self.name, 'state': 's', 'images': fields['images'],
                  'questions': {'u': {'type': 'noul', 'instructions': 'Is it a chart?'}}}
            r = c.post('/v1/systemone', headers={'Authorization': 'Bearer svc'}, json=s1)
            self.assertEqual(r.status_code, 200, r.text)
            self.assertEqual(r.json()['usage']['input_tokens'],
                             self.counter.bill_systemone(SystemOneRequest.model_validate(s1)).total)
            if not audio:
                r = c.post('/v1/ask', headers={'Authorization': 'Bearer svc'}, json=self.request(audios=[wav_url()]))
                self.assertEqual(r.status_code, 422)
                self.assertEqual(r.json()['error']['message'], f'{self.name} accepts text and images only')

    def test_audio_reads_as_a_spectrogram_image(self):
        from saina.contract import AskRequest
        scorer, counter = self.model.scorer, self.counter
        saved = scorer.inputs, counter.inputs
        # Exercise the audio path even on a package trained for images only.
        scorer.inputs = counter.inputs = ('text', 'image', 'audio')
        try:
            request = self.request(images=[png_url()], audios=[wav_url(2.0)])
            result = self.model.ask(**request)
            text = self.model.ask(**self.request(images=[png_url()]))
            # A 2 s clip is a 200 x 128 spectrogram, read at 65,536 pixels: 320 x 224, 70 tokens.
            self.assertEqual(result['usage']['input_tokens'] - text['usage']['input_tokens'], 70)
            self.assertEqual(counter.bill_ask(AskRequest.model_validate(request)).total, result['usage']['input_tokens'])
            from saina.media import MediaError
            with self.assertRaisesRegex(MediaError, 'at most 10.0 s'):
                self.model.ask(**self.request(audios=[wav_url(10.5)]))
        finally:
            scorer.inputs, counter.inputs = saved
