"""Visible, logged training and matched evaluation with bounded subprocesses."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/ubuntu22/miniforge3/envs/env_isaaclab/bin/python'
FFMPEG = '/home/ubuntu22/miniforge3/envs/env_isaaclab/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
SOURCE = Path('/home/ubuntu22/saved_models/V22_APPROVED/model_299.pt')
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
BATCH = ROOT / 'results' / STAMP
ENV = dict(os.environ, PYTHONPATH='/home/ubuntu22:/data/sac:/data/sac/scripts', WANDB_MODE='offline')
TASK = ['--task', 'Isaac-Go2-Stairs-V23-v0', '--agent', 'rsl_rl_sac_cfg_entry_point']


def status(stage, **details):
    temp = BATCH / 'status.tmp'
    temp.write_text(json.dumps(dict(stage=stage, updated_at=time.time(), supervisor_pid=os.getpid(), **details), ensure_ascii=False, indent=2))
    temp.replace(BATCH / 'status.json')


def run(command, stage, timeout=1800, env=ENV):
    print(f'\n[监管] {stage}', flush=True)
    with (BATCH / f'{stage}.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT.parent, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            with (BATCH / f'{stage}.log').open() as reader:
                while True:
                    chunk = reader.read()
                    if chunk:
                        print(chunk, end='', flush=True)
                    elapsed = round(time.monotonic()-started)
                    status(stage, child_pid=process.pid, elapsed_seconds=elapsed)
                    if process.poll() is not None:
                        print(reader.read(), end='', flush=True)
                        if process.returncode:
                            raise RuntimeError(f'{stage}: exit {process.returncode}')
                        break
                    if elapsed > timeout:
                        raise TimeoutError(stage)
                    time.sleep(2)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    BATCH.mkdir(parents=True)
    (ROOT / 'latest.txt').write_text(str(BATCH)+'\n')
    snapshot = BATCH / 'source_snapshot'
    snapshot.mkdir()
    for p in ROOT.iterdir():
        if p.suffix in ('.py', '.md', '.sh'):
            shutil.copy2(p, snapshot / p.name)
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    # Evaluation needs a writable checkpoint directory for simulator metadata.
    parent = BATCH / 'v22_reference'
    parent.mkdir()
    reference = parent / SOURCE.name
    shutil.copy2(SOURCE, reference)
    name = f'v23_{STAMP}'
    try:
        run([PYTHON, '-u', str(ROOT/'train.py'), *TASK, '--num_envs', '64', '--max_iterations', '29',
             '--seed', '1', '--resume', '--run_name', f'smoke_{STAMP}', '--headless',
             'agent.algorithm.replay_buffer_size=8192', 'agent.algorithm.num_mini_batches=2',
             'agent.algorithm.mini_batch_size=128'], '接口与更新检查', 600)
        # Keep training visible through terminal metrics without a simulator GUI.
        run([PYTHON, '-u', str(ROOT/'train.py'), *TASK, '--num_envs', '4096', '--max_iterations', '300',
             '--seed', '1', '--resume', '--run_name', name, '--headless'], '正式训练300轮', 7200)
        checkpoints = list(Path('/home/ubuntu22/logs/rsl_rl/stairs_v23_from_v22').glob(f'*_{name}/model_299.pt'))
        assert len(checkpoints) == 1, checkpoints
        trained = checkpoints[0]
        audit = json.loads((trained.parent/'audit.json').read_text())
        assert audit['iteration'] == 299 and audit['finite'] and audit['actor_change'] > 0, audit
        videos = BATCH/'videos'
        videos.mkdir()
        results = {}
        for label, checkpoint in [('V22', reference), ('V23', trained)]:
            results[label] = {}
            for height in (0., .05, .06, .07, .08, .10):
                stem = f'{label}_up_{height:.2f}m'
                output = BATCH/stem
                env = dict(ENV, STAIR_DIRECTION='up', STAIR_HEIGHT=str(height), STAIR_OUTPUT=str(output))
                previous = Path('/home/ubuntu22/stair_v14_transfer/results/20260916T003201Z') / (stem+'.json')
                if label == 'V22' and previous.exists():
                    shutil.copy2(previous, str(output)+'.json')
                    print('[复用相同测试协议的V22基线]', previous, flush=True)
                else:
                    run([PYTHON, '-u', str(ROOT/'evaluate.py'), *TASK, '--num_envs', '32', '--seed', '7',
                         '--checkpoint', str(checkpoint), '--headless'], f'批量评估_{stem}', env=env)
                data = json.loads(Path(str(output)+'.json').read_text())
                assert data['num_envs'] == 32 and sum(data[k] for k in ('success','terminated','other_no_success')) == 32
                results[label][str(height)] = data
        for label, checkpoint in [('V22', reference), ('V23', trained)]:
            for height in (0., .05, .08):
                stem = f'{label}_up_{height:.2f}m'
                env = dict(ENV, STAIR_DIRECTION='up', STAIR_HEIGHT=str(height))
                video_output = videos/stem
                env['STAIR_OUTPUT'] = str(video_output)
                run([PYTHON, '-u', str(ROOT/'evaluate.py'), *TASK, '--num_envs', '1', '--seed', '7',
                     '--checkpoint', str(checkpoint), '--headless', '--video', '--video_length', '2400'], f'录制_{stem}', env=env)
                run([FFMPEG, '-y', '-v', 'error', '-i', str(video_output)+'_video/rl-video-step-0.mp4',
                     '-vf', 'fps=50', '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0', '-an', str(video_output)+'.webm'], f'编码_{stem}')
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
        (BATCH/'results.json').write_text(json.dumps(results, indent=2))
        rows = ['| 条件 | 原V22 | V23楼梯微调 |', '|---|---:|---:|']
        cards = []
        for height in (0., .05, .06, .07, .08, .10):
            label = '平地' if height == 0 else f'{height*100:g} cm上楼'
            rows.append(f'| {label} | {results["V22"][str(height)]["success"]}/32 | {results["V23"][str(height)]["success"]}/32 |')
            if height not in (0., .05, .08):
                continue
            clips = ''.join(f'<div><h3>{arm}</h3><video controls loop muted src="videos/{arm}_up_{height:.2f}m.webm"></video></div>' for arm in ('V22','V23'))
            cards.append(f'<h2>{label}</h2><section>{clips}</section>')
        (BATCH/'结果.md').write_text('# V22 → V23 楼梯训练结果\n\n'+'\n'.join(rows)+'\n\n平地通过表示走过指定距离，不代表完整回合步态质量。单训练种子；视频连续12秒，保留重置。\n\n模型：'+str(trained)+'\n')
        page = BATCH/'视频对比.html'
        page.write_text('<!doctype html><meta charset="utf-8"><title>V22与V23：平地及楼梯对比</title><style>body{font-family:sans-serif;margin:24px;background:#eee}section{display:flex;gap:16px}section div{width:49%}video{width:100%}</style><h1>原V22与V23楼梯微调</h1><p>相同测试条件，seed=7；连续12秒，保留失败和重置。左V22，右V23。</p><button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">从头同步播放</button><pre>'+'\n'.join(rows)+'</pre>'+''.join(cards))
        subprocess.Popen(['firefox', '--new-window', page.as_uri()], env=ENV)
        status('complete', checkpoint=str(trained), page=str(page))
        print('\n[完成] '+str(page), flush=True)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
