"""Measured content RD metadata and full-window decoder latency distributions."""
import csv
import json
from pathlib import Path
import time
import numpy as np
from .codec import BUDGETS
from .asset import VideoAsset
from .media import read_prepared,write_video,to_uint8
from .metrics import QualityMetrics


def latency_summary(values):
    a=np.asarray(values,dtype=float)
    return {'mean_ms':float(a.mean()),'p50_ms':float(np.percentile(a,50)),
            'p95_ms':float(np.percentile(a,95)),'samples_ms':list(map(float,a))}


def profile_codec(asset_path,output_dir,backend,config):
    asset=VideoAsset.load(asset_path)
    if asset.metadata['model_id']!=backend.model_id.hex():
        raise ValueError('Profiling model differs from encoder checkpoint')
    video=read_prepared(asset.metadata['prepared_dir'])
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    metrics=QualityMetrics(backend.device)
    rows=[];uniform_rows=[];timing={};peak=0
    for _ in range(config.evaluation.warmup):
        backend.decode_tokens(asset.tokens[0],np.ones((4,256),bool))
    for k in BUDGETS:
        measurements=[];recon=[]
        for w,tokens in enumerate(asset.tokens):
            mask=np.arange(256)[None,:]<np.full((4,1),k)
            trials=[]
            for _ in range(config.evaluation.repeats):
                result=backend.decode_tokens(tokens,mask)
                trials.append(result.elapsed_ms)
                peak=max(peak,result.peak_memory_bytes)
            measurements.extend(trials)
            pixels=to_uint8(result.frames)
            valid=min(16,len(video.frames)-w*16)
            recon.append(pixels[:valid])
            for g in range(4):
                index=w*4+g
                if index>=asset.block_count:break
                count=min(4,len(video.frames)-index*4)
                q=metrics.evaluate(video.frames[index*4:index*4+count],result.frames[:,:,g*4:g*4+count])
                uniform_rows.append({'window_id':w,'block_id':g,'token_budget':k,
                                     'actual_bytes':asset.packet(index,0,k).wire_bytes,
                                     'token_payload_bytes':len(asset.packet(index,0,k).payload),
                                     'decode_time_ms':float(np.mean(trials)),**q})
        timing[str(k)]=latency_summary(measurements)
        if config.evaluation.save_reconstructions:
            pixels=np.concatenate(recon)
            write_video(out/f'budget_{k}.nut',pixels,video.pts_us,video.fps)
            from PIL import Image
            Image.fromarray(pixels[len(pixels)//2]).save(out/f'budget_{k}.png')
    # Independent per-group curves use one explicit fixed historical budget.
    history=config.evaluation.history_budget
    for index in range(asset.block_count):
        w,g=divmod(index,4)
        for k in BUDGETS:
            mask=np.zeros((4,256),bool);mask[:g,:history]=True;mask[g,:k]=True
            result=backend.decode_tokens(asset.tokens[w],mask)
            count=min(4,len(video.frames)-index*4)
            q=metrics.evaluate(video.frames[index*4:index*4+count],result.frames[:,:,g*4:g*4+count])
            rows.append({'window_id':w,'block_id':g,'token_budget':k,
                         'actual_bytes':asset.packet(index,0,k).wire_bytes,
                         'token_payload_bytes':len(asset.packet(index,0,k).payload),
                         'history_budgets':[history]*g,'decode_time_ms':result.elapsed_ms,
                         'decode_p95_ms':timing[str(k)]['p95_ms'],**q})
    result={'format':1,'model_id':backend.model_id.hex(),'asset_path':str(Path(asset_path).resolve()),
            'history_budget':history,'estimator':'measured VOD curves, history-dependent approximation',
            'decode_scope':'full native 16-frame window, then extract 4 frames',
            'timing':timing,'peak_memory_bytes':peak,'records':rows,
            'precompute_seconds':time.perf_counter()-started}
    (out/'rd.json').write_text(json.dumps(result,indent=2))
    result['metadata_storage_bytes']=(out/'rd.json').stat().st_size
    (out/'profile_summary.json').write_text(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    with open(out/'experiment_a.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(uniform_rows[0]))
        writer.writeheader();writer.writerows(uniform_rows)
    return result
