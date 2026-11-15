"""Disk VOD asset; sender reads indices, receiver receives only stream metadata."""
from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from .protocol import TokenPacket, TokenPacker, NO_EXPIRY, BASE


@dataclass
class VideoAsset:
    path: Path
    metadata: dict
    tokens: np.ndarray

    @classmethod
    def load(cls,path):
        path=Path(path)
        meta=json.loads((path/'asset.json').read_text())
        tokens=np.load(path/'tokens.npy',allow_pickle=False)
        expected=((meta['frame_count']+15)//16,4,256)
        if tokens.shape!=expected or tokens.dtype!=np.uint16 or np.any(tokens>=8192):
            raise ValueError('Invalid pre-encoded token asset')
        return cls(path,meta,tokens)

    @property
    def block_count(self):return (self.metadata['frame_count']+3)//4

    @property
    def fps(self):return Fraction(self.metadata['fps'])

    @property
    def block_duration(self):return 4/float(self.fps)

    def pts(self,index):return self.metadata['pts_us'][index*4]

    def duration(self,index):
        count=self.metadata['frame_count']
        end=min((index+1)*4,count)
        pts=self.metadata['pts_us']
        endpoint=pts[end] if end<count else pts[-1]+round(1e6/float(self.fps))
        return (endpoint-pts[index*4])/1e6

    def packet(self,index,start,end,deadline_us=NO_EXPIRY):
        window,g=divmod(index,4)
        return TokenPacket(1,bytes.fromhex(self.metadata['model_id']),window,g,self.pts(index),
                           start,end-start,deadline_us,TokenPacker.pack_tokens(self.tokens[window,g,start:end]),
                           BASE if start==0 else 0)


def encode_video(prepared_dir,output_dir,backend):
    from .media import read_prepared
    from .codec import ProgressiveVideoEncoder
    video=read_prepared(prepared_dir)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    encoder=ProgressiveVideoEncoder(backend)
    windows=[encoder.encode_window(video.window(w),w) for w in range((len(video.frames)+15)//16)]
    np.save(output_dir/'tokens.npy',np.stack([w.tokens for w in windows]),allow_pickle=False)
    meta={**video.metadata,'model_id':backend.model_id.hex(),'checkpoint_sha256':backend.sha256,
          'native_tokens_per_block':backend.native_tokens_per_block,'stream_tokens_per_block':256,
          'prepared_dir':str(Path(prepared_dir).resolve()),'encode_calls':encoder.encoding_calls,
          'encode_time_ms':[w.encode_time_ms for w in windows]}
    (output_dir/'asset.json').write_text(json.dumps(meta,indent=2))
    return VideoAsset.load(output_dir)
