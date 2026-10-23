import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from saina.releases import stage_release, plan_release, default_release


def plan(tmp_path):
    data=b'verified checkpoint'
    return {'repo':'test/model','revision':'a'*40,'snapshot':str(tmp_path/'snapshot'),
        'cache':str(tmp_path),'files':{'config.json':{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}},
        'missing_files':['config.json'],'download_bytes':len(data),'free_bytes':1024**3},data


def test_explicit_download_and_space_check_before_network(tmp_path,monkeypatch):
    p,data=plan(tmp_path)
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=lambda *a,**k:pytest.fail('network called')))
    with pytest.raises(ValueError,match='saina serve'):stage_release(p)
    with pytest.raises(ValueError,match='cached'):stage_release(p,download=True,offline=True)
    p['free_bytes']=0
    with pytest.raises(ValueError,match='disk'):stage_release(p,download=True)


def test_interrupted_download_cannot_load_partial_snapshot(tmp_path,monkeypatch):
    p,data=plan(tmp_path)
    def interrupted(*a,**k):
        Path(p['snapshot']).mkdir()
        (Path(p['snapshot'])/'config.json.incomplete').write_bytes(data[:3])
        raise OSError('download interrupted')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=interrupted))
    with pytest.raises(OSError):stage_release(p,download=True)
    def resumed(*a,**kwargs):
        assert kwargs['revision']=='a'*40
        assert kwargs['allow_patterns']==['config.json']
        (Path(p['snapshot'])/'config.json').write_bytes(data)
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=resumed))
    assert stage_release(p,download=True)==Path(p['snapshot'])
    p['missing_files']=[]
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=lambda *a,**k:pytest.fail('offline network')))
    assert stage_release(p,offline=True)==Path(p['snapshot'])
    (Path(p['snapshot'])/'config.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):stage_release(p,offline=True)


def test_packaged_release_pins_and_no_download_plan(tmp_path):
    for backend in ['native','mlx']:
        release=default_release(backend)
        assert len(release['revision'])==40
        assert all(len(f['sha256'])==64 and f['size']>0 for f in release['files'].values())
        p=plan_release(backend,cache=tmp_path)
        assert p['download_bytes']==p['total_bytes']>10**9
        assert not Path(p['snapshot']).exists()

@pytest.mark.parametrize('cached,interactive,answer,expected',[
    (['one'],False,None,'one'),([],True,'2','two'),
    (['one','two'],True,'2','two'),([],False,None,None),(['one','two'],False,None,None)])
def test_model_selection(cached,interactive,answer,expected,monkeypatch):
    import saina.releases as r
    monkeypatch.setattr(r,'catalog',lambda b:{'one':{},'two':{}})
    monkeypatch.setattr(r,'plan_release',lambda b,model_key:{'key':model_key,
        'missing_files':[] if model_key in cached else ['weights'],'download_bytes':100})
    if expected:
        assert r.choose_release('native',interactive=interactive,reader=lambda _:answer)['key']==expected
    else:
        with pytest.raises(ValueError,match='non-interactive'): r.choose_release('native',interactive=False)
    assert r.choose_release('native','two',interactive=False)['key']=='two'
