import pytest
from adaptivetokstream.buffer import TokenBuffer
from adaptivetokstream.protocol import TokenPacket, TokenPacker

def pkt(start,end,block=0,window=0,deadline=1000,values=None):
    ids=list(range(start,end)) if values is None else values
    return TokenPacket(1,b'a'*16,window,block,block*133333,start,len(ids),deadline,TokenPacker.pack_tokens(ids))

def test_out_of_order_duplicates_and_prefix():
    b=TokenBuffer(1,b'a'*16)
    assert b.append(pkt(8,16),now_us=0).status == 'accepted'
    assert b.prefix(0,0)==0
    b.append(pkt(0,8),now_us=0)
    assert b.prefix(0,0)==16
    assert b.append(pkt(0,8),now_us=0).status=='duplicate'
    b.append(pkt(12,32),now_us=0)
    assert b.prefix(0,0)==32
    assert b.snapshot(0,0)[0][0][:32]==list(range(32))
    assert b.stats.duplicate_payload_bytes>0

def test_conflicting_overlap_is_atomic():
    b=TokenBuffer(1,b'a'*16); b.append(pkt(0,8),0)
    with pytest.raises(ValueError):
        b.append(pkt(4,16,values=[99]*12),0)
    assert b.prefix(0,0)==8

def test_deadline_freeze_and_retirement():
    b=TokenBuffer(1,b'a'*16)
    assert b.append(pkt(0,8),1001).status=='expired'
    b.append(pkt(0,8),1000)
    b.finalize(0,0)
    assert b.append(pkt(8,16),1000).status=='played'
    assert b.prefix(0,0)==8
    for g in range(1,4):
        b.append(pkt(0,8,block=g),0); b.finalize(0,g)
    b.clear_window(0)
    assert not b.windows
    assert b.append(pkt(0,8),0).status=='retired'
    b.append(pkt(0,8,window=1),0)
    assert b.prefix(1,0)==8

def test_future_tokens_are_hidden_and_history_frozen():
    b=TokenBuffer(1,b'a'*16)
    b.append(pkt(0,32,block=0),0); b.append(pkt(0,64,block=2),0)
    ids,mask=b.snapshot(0,0)
    assert sum(mask[0])==32 and not any(mask[2])
    b.finalize(0,0)
    b.append(pkt(0,16,block=1),0)
    _,mask=b.snapshot(0,1)
    assert [sum(m) for m in mask]==[32,16,0,0]

def test_foreign_model_and_stream_rejected():
    from dataclasses import replace
    b=TokenBuffer(1,b'a'*16)
    for p in [replace(pkt(0,8),stream_id=2),replace(pkt(0,8),model_id=b'b'*16)]:
        with pytest.raises(ValueError): b.append(p,0)
