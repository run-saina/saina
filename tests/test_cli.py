import json
import pytest
from saina.cli import main


def test_launch_defaults(monkeypatch):
    import saina.local
    calls=[]
    monkeypatch.setattr(saina.local,'run_local',lambda args:calls.append(args) or 0)
    assert main(['serve','helm','--backend','native'])==0
    args=calls[0]
    assert args.no_browser and args.api_only and args.port==8765 and args.model_key=='helm'
    assert args.gguf_file=='helm-trunk-f32.gguf' and args.threads==8


@pytest.mark.parametrize('flags',[
 ['--revision','a'*40], ['--port','0'], ['--plan','--port','65536'],
 ['--max-length','0'], ['--offline','--download'], ['--plan','--download'],
 ['--checkpoint','missing-repo'], ['--runner','helper'], ['--threads','4'],
 ['--gguf-file','model.gguf'], ['--api-only','--ui-dir','ui'],
 ['--backend','gguf'], ['--plan','--checkpoint','.'],
 ['--download','--checkpoint','.'], ['--checkpoint','.','--revision','main'],
])
def test_bad_options_fail_before_runtime(flags,monkeypatch,capsys):
    import saina.local
    monkeypatch.setattr(saina.local,'run_local',lambda args:pytest.fail('runtime called'))
    with pytest.raises(SystemExit) as exc:main(['serve','--backend','native',*flags])
    assert exc.value.code==2
    assert 'error:' in capsys.readouterr().err


def test_doctor_default_uses_release_lock_without_loading(monkeypatch,capsys):
    import saina.releases
    import saina.cli
    monkeypatch.setattr(saina.releases,'choose_release',lambda *a,**k:{'snapshot':'cache','files':{'model':{'sha256':'a'*64}}})
    calls=[]
    monkeypatch.setattr(saina.cli,'verify_files',lambda *a,**k:calls.append((a,k)) or {'model':'a'*64})
    assert main(['doctor','--backend','mlx','--json'])==0
    assert json.loads(capsys.readouterr().out)['status']=='integrity_verified'
    assert calls[0][0]==('cache',{'model':'a'*64})


def test_runtime_error_is_stderr_and_dependency_hint_is_installable(monkeypatch,capsys):
    import saina.local
    def fail(args):raise ImportError('mlx_lm missing')
    monkeypatch.setattr(saina.local,'run_local',fail)
    assert main(['serve','--backend','native'])==1
    output=capsys.readouterr()
    assert output.out=='' and 'wheel with [server,local]' in output.err


def test_version(capsys):
    from saina import __version__
    with pytest.raises(SystemExit) as exc:main(['--version'])
    assert exc.value.code==0
    assert __version__ in capsys.readouterr().out

@pytest.mark.parametrize('argv',[['local'],['serve','--plan'],['serve','--download'],['serve','--offline']])
def test_unreleased_setup_flags_and_alias_removed(argv):
    with pytest.raises(SystemExit) as exc:main(argv)
    assert exc.value.code==2


def test_corrupt_ui_fails_before_downloading_or_loading(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import saina.local
    import saina.releases
    import saina.backends
    (tmp_path/'ui-manifest.json').write_text('{}')
    monkeypatch.setattr(saina.releases,'dependencies',lambda b:None)
    monkeypatch.setattr(saina.releases,'plan_release',lambda b:{'repo':'pinned/model','revision':'a'*40,
        'total_bytes':100,'download_bytes':100,'free_bytes':1000})
    monkeypatch.setattr(saina.releases,'stage_release',lambda *a,**k:pytest.fail('downloaded before UI validation'))
    monkeypatch.setattr(saina.backends,'load_backend',lambda *a,**k:pytest.fail('loaded before UI validation'))
    monkeypatch.setattr(saina.local.socket,'socket',lambda *a,**k:SimpleNamespace(bind=lambda *a:None,close=lambda:None))
    args=SimpleNamespace(command='playground',model_key=None,backend='native',checkpoint=None,port=8765,ui_dir=str(tmp_path),api_only=False)
    with pytest.raises(ValueError,match='nonempty'):
        saina.local.run_local(args)


def test_playground_has_no_model_argument(monkeypatch):
    import saina.local
    calls=[]
    monkeypatch.setattr(saina.local,'run_local',lambda args:calls.append(args) or 0)
    assert main(['playground','--no-browser'])==0
    assert calls[0].model_key is None and not calls[0].api_only
    with pytest.raises(SystemExit): main(['playground','helm'])
