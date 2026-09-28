#!/usr/bin/env python3
"""Profile the canonical Eval9 text-only sample and materialize token windows."""
from __future__ import annotations
import argparse, hashlib, json, os, tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

FORMAT='eval9-code-page-profile-v1'
ILLEGAL_CONTROL_RE = __import__("re").compile(
    "[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent);tmp=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally: tmp.unlink(missing_ok=True)

def quantiles(vals):
    vals=sorted(vals)
    def q(p): return vals[min(len(vals)-1,round((len(vals)-1)*p))]
    return {'min':vals[0],'p10':q(.1),'p25':q(.25),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'p99':q(.99),'max':vals[-1],'mean':sum(vals)/len(vals)}

def windows(n,max_len=512,max_windows=8):
    if n<=max_len:return [(0,n)]
    if n<=max_len*max_windows:return [(s,min(n,s+max_len)) for s in range(0,n,max_len)]
    starts=[round(i*(n-max_len)/(max_windows-1)) for i in range(max_windows)]
    return [(s,s+max_len) for s in starts]

def normalize_text(text):
    return ILLEGAL_CONTROL_RE.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n"))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-parquet',required=True);ap.add_argument('--model',required=True);ap.add_argument('--output-dir',required=True);ap.add_argument('--overwrite',action='store_true');a=ap.parse_args()
    inp=Path(a.input_parquet).resolve();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    prof=out/'sample_profile.json';win=out/'document_windows.jsonl'
    if (prof.exists() or win.exists()) and not a.overwrite: raise FileExistsError('profile outputs exist; use --overwrite')
    table=pq.read_table(inp); assert table.column_names==['text']; texts=table['text'].to_pylist(); assert len(texts)==10000
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=True)
    chars=[];utf8=[];tokens=[];wins=[];empty=0;dups=0;seen=set(); previews=[]
    fd,tmpname=tempfile.mkstemp(prefix='.document_windows.',suffix='.tmp',dir=out);os.close(fd);tmp=Path(tmpname)
    with tmp.open('w',encoding='utf-8') as f:
        for i,text in enumerate(texts):
            if not isinstance(text,str) or not text.strip():empty+=1
            if text in seen:dups+=1
            else:seen.add(text)
            ids=tok(normalize_text(text),add_special_tokens=False).input_ids
            spans=windows(len(ids));chars.append(len(text));utf8.append(len(text.encode()));tokens.append(len(ids));wins.append(len(spans))
            for j,(s,e) in enumerate(spans):
                f.write(json.dumps({'document_index':i,'qwen_token_count':len(ids),'window_index':j,'token_start':s,'token_end':e,'valid_tokens':e-s},ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
            if len(previews)<8:previews.append({'document_index':i,'preview':text[:500]})
            if (i+1)%1000==0:print(f'[profile] {i+1}/10000',flush=True)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,win)
    payload={'format':FORMAT,'complete':True,'generated_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'input':{'path':str(inp),'bytes':inp.stat().st_size,'sha256':sha256(inp),'rows':len(texts),'schema':'text: string'},'tokenizer':str(Path(a.model).resolve()),'rows':len(texts),'empty_texts':empty,'distinct_texts':len(seen),'duplicate_rows':dups,'character_length':quantiles(chars),'utf8_bytes':quantiles(utf8),'qwen_token_count':quantiles(tokens),'windows_per_document':quantiles(wins),'total_windows':sum(wins),'total_processed_tokens':sum(min(x,4096) for x in tokens),'previews':previews,'files':{'document_windows':{'path':win.name,'bytes':win.stat().st_size,'sha256':sha256(win)}}}
    atomic_json(prof,payload);print(json.dumps({'profile':str(prof),'windows':str(win),'rows':len(texts),'total_windows':sum(wins),'total_processed_tokens':payload['total_processed_tokens'],'duplicates':dups},indent=2))
if __name__=='__main__':main()
