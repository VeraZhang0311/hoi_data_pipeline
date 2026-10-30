"""Unpack immutable code/dependencies on the compute node, then run CPU labeling."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def unpack(path,expected,directory):
    if digest(path)!=expected: raise RuntimeError("Archive SHA mismatch: "+path)
    directory.mkdir()
    with tarfile.open(path,"r:gz") as archive:
        for member in archive.getmembers():
            target=(directory/member.name).resolve()
            if not target.is_relative_to(directory.resolve()) or not (member.isfile() or member.isdir()):
                raise RuntimeError("Unsafe archive member "+member.name)
        archive.extractall(directory,filter="data")

def wait_inputs(frozen,source,timeout):
    root=Path(frozen)
    manifest=json.loads((root/"manifest.json").read_text())
    if digest(root/"inventory.json")!=manifest["inventory_sha256"]:raise RuntimeError("Inventory changed")
    inventory=json.loads((root/"inventory.json").read_text())
    items={item["path"]:item for record in inventory[source] for item in record["_input_files"]}
    if not items:raise RuntimeError("Empty readiness input set")
    deadline=time.monotonic()+timeout;last=None
    while True:
        missing=[]
        for path,item in items.items():
            p=Path(path)
            if not p.is_file() or ("size" in item and p.stat().st_size!=item["size"]):missing.append(path)
        if len(missing)!=last:print(json.dumps({"source":source,"missing_or_incomplete_files":len(missing)}),flush=True);last=len(missing)
        if not missing:break
        if time.monotonic()>deadline:raise TimeoutError("Input readiness deadline exceeded")
        time.sleep(30)
    for path,item in items.items():
        if digest(path)!=item["sha256"]:raise RuntimeError("Input SHA mismatch: "+path)
    print(json.dumps({"source":source,"ready":True,"verified_files":len(items)}),flush=True)

def main():
    if os.environ.get("WORKER_MODE")=="gate":
        wait_inputs(os.environ["FROZEN_INPUTS"],os.environ["SOURCE"],float(os.environ.get("READY_TIMEOUT_SECONDS","43200")))
        return
    with tempfile.TemporaryDirectory(prefix="hoi-label-",dir=os.environ.get("SLURM_TMPDIR") or "/tmp") as tmp:
        root=Path(tmp)
        unpack(os.environ["PIPELINE_ARCHIVE"],os.environ["PIPELINE_ARCHIVE_SHA256"],root/"code")
        unpack(os.environ["RUNTIME_ARCHIVE"],os.environ["RUNTIME_ARCHIVE_SHA256"],root/"runtime")
        env=dict(os.environ)
        env["PYTHONPATH"]=str(root/"runtime/site")
        env["LD_LIBRARY_PATH"]=str(root/"runtime/native")+":"+env.get("LD_LIBRARY_PATH","")
        env.update(OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONDONTWRITEBYTECODE="1",PYTHONUNBUFFERED="1")
        if env.get("WORKER_MODE")=="smoke":
            unpack(env["SMOKE_SOURCE_ARCHIVE"],env["SMOKE_SOURCE_ARCHIVE_SHA256"],root/"omni")
            args=[sys.executable,"-B","-m","hoi_pipeline.smoke",env["FROZEN_INPUTS"],env["LABEL_OUTPUT"],str(root/"omni")]
        else:
            args=[sys.executable,"-B","-m","hoi_pipeline.cli","run","--frozen",env["FROZEN_INPUTS"],"--output",env["LABEL_OUTPUT"],"--source",env["SOURCE"],"--shard",env["SLURM_ARRAY_TASK_ID"],"--shards",env["SHARDS"]]
            if env.get("SMOKE_LIMIT"):args.extend(["--limit",env["SMOKE_LIMIT"]])
        subprocess.run(args,cwd=root/"code",env=env,check=True)

if __name__=="__main__":main()
