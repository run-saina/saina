from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from saina.local import create_local_app
from saina.backends import select_backend
from test_server import FakeScorer


def test_backend_selection():
    assert select_backend('auto', 'Darwin', 'arm64') == 'mlx'
    assert select_backend('auto', 'Linux', 'x86_64') == 'native'
    assert select_backend('gguf', 'Darwin', 'arm64') == 'gguf'
    with pytest.raises(ValueError):
        select_backend('mlx', 'Linux', 'x86_64')


def test_local_boundary_and_real_api_contract():
    scorer = FakeScorer()
    scorer.model_id = 'saina-helm-0.8b'
    ui = Path(__file__).parents[1] / 'src/saina/ui'
    with TestClient(create_local_app(scorer, 'secret', ui), base_url='http://127.0.0.1:8765') as c:
        assert c.get('/console/').status_code == 200
        config = c.get('/console/runtime-config.js')
        assert '"mode": "local"' in config.text and 'secret' not in config.text
        assert "connect-src 'self'" in config.headers['content-security-policy']
        body = {'model': 'saina-helm', 'state': 'Charged twice', 'questions': {'route': {'type': 'single_choice',
            'question': 'Which team?', 'options': {'billing': 'Billing', 'shipping': 'Shipping'}}}}
        assert c.post('/v1/ask', json=body).status_code == 401
        auth = {'Authorization': 'Bearer secret'}
        assert c.post('/v1/ask', json=body, headers={**auth, 'Origin': 'https://evil.example'}).status_code == 403
        assert c.get('/healthz', headers={'Host': 'attacker.example:8765'}).status_code == 403
        r = c.post('/v1/ask', json=body, headers=auth)
        assert r.status_code == 200, r.text
        assert 'route' in r.json()['answers']
        assert scorer.forwards == 1


def test_mlx_adapter_head_readout_and_validation(tmp_path, monkeypatch):
    """Exercise the shipped MLX algorithm without requiring Apple GPU hardware."""
    import json
    import sys
    import types
    import numpy as np
    from saina.backends import MLXScorer
    from saina.prompt import answer_codes
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [ord(text)] if len(text) == 1 else [1, 2]
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]['content']
    tok = Tokenizer()
    codes, ids = answer_codes(tok)
    (tmp_path/'head-config.json').write_text(json.dumps({'prompt':'AgenticScorer.encode_prompt',
        'codes':codes,'code_ids':ids,'supported_modes':['single_label','multi_label'],'max_length':8}))
    (tmp_path/'config.json').write_text('{}')
    weights = np.zeros((100, 2),dtype=np.float32)
    weights[ids[0]] = [1,0];weights[ids[1]] = [0,1]
    model = types.SimpleNamespace(model=lambda x:np.array([[[2.,1.]]]),
        language_model=types.SimpleNamespace(lm_head=types.SimpleNamespace(weight=weights)))
    mx = types.ModuleType('mlx.core')
    mx.array, mx.float32 = np.array, np.float32
    mx.softmax = lambda z: np.exp(z-z.max())/np.exp(z-z.max()).sum()
    mx.sigmoid = lambda z: 1/(1+np.exp(-z))
    mlx = types.ModuleType('mlx');mlx.core = mx
    monkeypatch.setitem(sys.modules,'mlx',mlx);monkeypatch.setitem(sys.modules,'mlx.core',mx)
    monkeypatch.setitem(sys.modules,'mlx_lm',types.SimpleNamespace(load=lambda root:(model,tok)))
    scorer=MLXScorer(tmp_path)
    assert scorer.forward([1,2],2) == pytest.approx([0.7310586,0.2689414])
    assert scorer.forward([1,2],2,'multi_label') == pytest.approx([0.880797,0.7310586])
    with pytest.raises(ValueError):scorer.forward([1],2,temperature=0)
    (tmp_path/'config.json').write_text('{"quantization":{"bits":4}}')
    with pytest.raises(ValueError,match='unquantized'):MLXScorer(tmp_path)


def test_gguf_adapter_protocol_integrity_and_shutdown(tmp_path, monkeypatch):
    import hashlib
    import json
    import os
    import sys
    import types
    import numpy as np
    from safetensors.numpy import save_file
    from saina.backends import GGUFScorer
    from saina.prompt import answer_codes
    class Tokenizer:
        def encode(self, text, **kwargs):return [ord(text)] if len(text)==1 else [1,2]
        def apply_chat_template(self, messages, **kwargs):return messages[0]['content']
    tok=Tokenizer();codes,ids=answer_codes(tok)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:tok)))
    (tmp_path/'head-config.json').write_text(json.dumps({'prompt':'AgenticScorer.encode_prompt','codes':codes,
        'code_ids':ids,'supported_modes':['single_label','multi_label']}))
    weight=np.zeros((len(codes),2),dtype=np.float32);weight[0]=[1,0];weight[1]=[0,1]
    save_file({'weight':weight},str(tmp_path/'head.safetensors'))
    (tmp_path/'helm-trunk-f32.gguf').write_bytes(b'test-only')
    (tmp_path/'merged-hf').mkdir()
    for name in ['config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        (tmp_path/'merged-hf'/name).write_text('{}')
    files={str(p.relative_to(tmp_path)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path/'gguf-manifest.json').write_text(json.dumps({'files':files}))
    runner=tmp_path/'runner'
    runner.write_text('#!'+sys.executable+'\nimport sys\nfor line in sys.stdin: print("2 1",flush=True)\n')
    runner.chmod(0o755)
    scorer=GGUFScorer(tmp_path,runner)
    try:
        assert scorer.predict('hello','which?',['a','b']) == pytest.approx([.7310586,.2689414])
    finally:scorer.close()
    assert scorer.helper.poll()==0
    (tmp_path/'helm-trunk-f32.gguf').write_bytes(b'corrupted')
    with pytest.raises(ValueError):GGUFScorer(tmp_path,runner)
