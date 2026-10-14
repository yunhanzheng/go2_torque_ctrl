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
SOURCE = Path('/data/sac/saved_models/ETH_SAC_V24_10CM/model_299.pt')
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
BATCH = ROOT / 'results' / STAMP
ENV = dict(os.environ, PYTHONPATH='/home/ubuntu22:/data/sac:/data/sac/scripts', WANDB_MODE='offline')
TASK = ['--task', 'Isaac-Go2-Terrain-V25-v0', '--agent', 'rsl_rl_sac_cfg_entry_point']


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



from catalog import NAMES, DIFFICULTIES


def publish(results, checkpoint=None):
    (BATCH/'results.json').write_text(json.dumps(results,indent=2))
    rows=['| 地形 | 难度 | V24 | V25 |','|---|---|---:|---:|']
    for case,label in NAMES.items():
        for level in DIFFICULTIES:
            a=results.get('V24',{}).get(case,{}).get(level)
            b=results.get('V25',{}).get(case,{}).get(level)
            rows.append(f'| {label} | {level} | {str(a["success"])+"/32" if a else "待评估"} | {str(b["success"])+"/32" if b else "待评估"} |')
    (BATCH/'结果.md').write_text('# V25 非楼梯地形结果\n\n'+'\n'.join(rows)+'\n\n固定seed7、首回合、规定3.5m路线穿越率；每类8个地图实例32环境。单训练种子。\n\n模型：'+str(checkpoint)+'\n')
    cards=[]
    for case,label in NAMES.items():
        path=BATCH/'videos'/f'V25_{case}_hard.webm'
        if path.exists():
            data=results.get('V25',{}).get(case,{}).get('hard',{})
            cards.append(f'<article><h2>{label}（hard：{data.get("success","?")}/32）</h2><video controls loop muted preload="metadata" src="videos/{path.name}"></video></article>')
    page=BATCH/'视频演示.html'
    page.write_text('<!doctype html><meta charset="utf-8"><title>V25 非楼梯地形</title><style>body{font-family:sans-serif;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}video{width:100%}article{background:white;padding:16px}pre{white-space:pre-wrap}</style><h1>V25 非楼梯地形训练结果</h1><p>固定seed7，每段连续约12秒，保留失败及重置。所有演示均为hard条件；单段视频不替代32环境通过率。</p><details><summary>展开全部批量评估</summary><pre>'+'\n'.join(rows)+'</pre></details><main>'+''.join(cards)+'</main>')
    return page


def main():
    BATCH.mkdir(parents=True)
    (ROOT/'latest.txt').write_text(str(BATCH)+'\n')
    snapshot=BATCH/'source_snapshot';snapshot.mkdir()
    for p in ROOT.iterdir():
        if p.suffix in ('.py','.sh','.md'):shutil.copy2(p,snapshot/p.name)
    original_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    reference_dir=BATCH/'v24_reference';reference_dir.mkdir()
    reference=reference_dir/'model_299.pt';shutil.copy2(SOURCE,reference)
    name=f'v25_{STAMP}'
    results={'V24':{},'V25':{}}
    try:
        run([PYTHON,'-u',str(ROOT/'preflight.py'),str(BATCH/'catalog.json')],'14类地形生成检查',600)
        run([PYTHON,'-u',str(ROOT/'train.py'),*TASK,'--num_envs','64','--max_iterations','29',
             '--seed','1','--resume','--run_name',f'smoke_{STAMP}','--headless',
             'agent.algorithm.replay_buffer_size=8192','agent.algorithm.num_mini_batches=2',
             'agent.algorithm.mini_batch_size=128'],'奖励与更新检查',600)
        run([PYTHON,'-u',str(ROOT/'train.py'),*TASK,'--num_envs','4096','--max_iterations','600',
             '--seed','1','--resume','--run_name',name,'--headless'],'正式训练600轮',10800)
        paths=list(Path('/home/ubuntu22/logs/rsl_rl/terrain_v25_from_v24').glob(f'*_{name}/model_599.pt'))
        assert len(paths)==1,paths
        trained=paths[0]
        audit=json.loads((trained.parent/'audit.json').read_text())
        assert audit['finite'] and audit['iteration']==599 and audit['actor_change']>0,audit
        for label,checkpoint in [('V24',reference),('V25',trained)]:
            for case in NAMES:
                results[label][case]={}
                for level,difficulty in DIFFICULTIES.items():
                    stem=f'{label}_{case}_{level}'
                    output=BATCH/stem
                    env=dict(ENV,TERRAIN_CASE=case,TERRAIN_DIFFICULTY=str(difficulty),TERRAIN_OUTPUT=str(output))
                    run([PYTHON,'-u',str(ROOT/'evaluate.py'),*TASK,'--num_envs','32','--seed','7',
                         '--checkpoint',str(checkpoint),'--headless'],f'评估_{stem}',1200,env)
                    data=json.loads(Path(str(output)+'.json').read_text())
                    assert data['num_envs']==32 and sum(data[k] for k in ('success','terminated','other_no_success'))==32
                    results[label][case][level]=data
                    publish(results,trained)
        videos=BATCH/'videos';videos.mkdir()
        for case in NAMES:
            stem=f'V25_{case}_hard';output=videos/stem
            env=dict(ENV,TERRAIN_CASE=case,TERRAIN_DIFFICULTY=str(DIFFICULTIES['hard']),TERRAIN_OUTPUT=str(output))
            run([PYTHON,'-u',str(ROOT/'evaluate.py'),*TASK,'--num_envs','1','--seed','7',
                 '--checkpoint',str(trained),'--headless','--video','--video_length','2400'],f'录制_{stem}',1800,env)
            webm=Path(str(output)+'.webm')
            run([FFMPEG,'-y','-v','error','-i',str(output)+'_video/rl-video-step-0.mp4','-vf','fps=50',
                 '-c:v','libvpx-vp9','-crf','30','-b:v','0','-an',str(webm)],f'编码_{stem}',900)
            run([FFMPEG,'-v','error','-i',str(webm),'-f','null','-'],f'校验_{stem}',120)
            page=publish(results,trained)
        assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==original_hash
        subprocess.Popen(['firefox','--new-window',page.as_uri()],env=ENV)
        status('complete',checkpoint=str(trained),page=str(page))
        print('[全部完成]',page,flush=True)
    except BaseException as exc:
        status('failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
