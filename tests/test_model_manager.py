import time
import threading
from fastapi.testclient import TestClient
from saina.local import create_local_app
from saina.model_manager import ModelManager, EmptyScorer
from test_server import FakeScorer


def test_catalog_auth_load_and_concurrent_admission(monkeypatch):
    import saina.releases as r
    import saina.backends as b
    import saina.model_manager as mm
    gate=threading.Event()
    monkeypatch.setattr(r,'dependencies',lambda _:None)
    monkeypatch.setattr(r,'plan_release',lambda *a,**k:{'key':'helm','revision':'a'*40,'missing_files':[]})
    monkeypatch.setattr(r,'stage_release',lambda *a,**k: gate.wait(3) or 'cache')
    scorer=FakeScorer();scorer.model_id='saina-helm'
    monkeypatch.setattr(b,'load_backend',lambda *a:scorer)
    monkeypatch.setattr(mm,'smoke',lambda s:None)
    manager=ModelManager('native')
    monkeypatch.setattr(manager,'listing',lambda:{'phase':manager.phase,'selected':manager.selected})
    app=create_local_app(EmptyScorer(),'secret',manager=manager)
    auth={'Authorization':'Bearer secret'}
    with TestClient(app,base_url='http://127.0.0.1:8765') as c:
        assert manager.active is None
        assert c.get('/local/models').status_code==401
        assert c.post('/local/models/select',json={'key':'helm'}).status_code==401
        assert c.get('/local/models',headers=auth).json()['phase']=='idle'
        assert c.post('/v1/ask',json={},headers=auth).status_code==503
        assert c.post('/local/models/select',json={'key':'invalid'},headers=auth).status_code==400
        assert c.post('/local/models/select',json={'key':'helm'},headers=auth).status_code==202
        assert c.post('/local/models/select',json={'key':'helm'},headers=auth).status_code==409
        assert c.post('/v1/ask',json={},headers=auth).status_code==503
        gate.set()
        for _ in range(100):
            if not manager.busy:break
            time.sleep(.01)
        assert manager.phase=='ready' and app.state.scorer is scorer
        manager.inflight=1
        assert c.post('/local/models/select',json={'key':'helm'},headers=auth).status_code==409
        manager.inflight=0
        def fail(*a,**k):raise ValueError('broken checkpoint')
        monkeypatch.setattr(r,'stage_release',fail)
        assert c.post('/local/models/select',json={'key':'helm'},headers=auth).status_code==202
        for _ in range(100):
            if not manager.busy:break
            time.sleep(.01)
        assert manager.phase=='error' and manager.error=='broken checkpoint'
        assert manager.active is None
