from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from .common import sha256, identity, canonical_json
from .pipeline import Labeler, atomic_json, save_clip
from .adapters import ultra, hoi_retarget, grail, omnicontact

def read(path): return json.loads(Path(path).read_text(encoding="utf8"))

def split_index(config):
    source=config["sources"]["ultra"]
    return hoi_retarget.frozen_ultra_split_index(source["selection_path"],source["eval_selection_path"])

def records(config,source):
    s=config["sources"][source]
    if s.get("inventory_override"):
        return read(s["inventory_override"])
    if source=="ultra":
        a=ultra.list_clips(s["selection_path"])
        for r in a: r["selection_path"]=s["selection_path"]
        b=ultra.list_clips(s["eval_selection_path"])
        for r in b: r["selection_path"]=s["eval_selection_path"]
        return a+b
    if source=="hoi_retarget": return hoi_retarget.list_clips(s["parquet_path"],revision=s["revision"],split_index=split_index(config))
    if source=="grail_sitting": return grail.list_clips(**s)
    if source=="omnicontact": return omnicontact.list_clips(**s)
    raise ValueError(source)

def load(config,source,record):
    s=config["sources"][source]
    if source=="ultra":
        return ultra.load_clip(record["selection_path"],record["clip_id"],bindings_path=s["bindings_path"],assets_path=s["assets_path"],timebase=s["timebase"])
    if source=="hoi_retarget":
        return hoi_retarget.load_clip(**s,row_index=record["row_index"],split_index=split_index(config))
    if source=="grail_sitting": return grail.load_clip(record,source_revision=s["source_revision"])
    if source=="omnicontact": return omnicontact.load_clip(record,source_root=s["source_root"])
    raise ValueError(source)

def freeze(config_path,output):
    from .usd_geometry import robot_geometry
    from .kinematics import RobotFK
    config=read(config_path);out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    if sha256(config["robot_bundle"])!=config["robot_bundle_sha256"]: raise ValueError("Frozen v1.2 bundle SHA mismatch")
    if sha256(config["urdf_path"])!=config["urdf_sha256"]: raise ValueError("Frozen URDF SHA mismatch")
    bundle=read(config["robot_bundle"]);native=bundle["native_robot_asset"]
    if sha256(native["path"])!=native["sha256"]: raise ValueError("Frozen robot asset SHA mismatch")
    fk=RobotFK(config["urdf_path"],hoi_retarget.G1_JOINT_NAMES)
    recipe=robot_geometry(native["path"],fk,coordinate_contract=config["robot_coordinate_contract"])
    recipe["binding"]={"bundle":config["robot_bundle"],"bundle_sha256":config["robot_bundle_sha256"],"urdf_sha256":config["urdf_sha256"],"native_robot_asset":native}
    atomic_json(out/"robot_geometry.json",recipe)
    atomic_json(out/"config.json",config)
    inventory={s:records(config,s) for s in config["sources"]}
    common_inputs={}
    digest_cache={}
    def evidence(path):
        path=str(path)
        if path not in digest_cache:
            p=Path(path);digest_cache[path]={"path":path,"sha256":sha256(path),"size":p.stat().st_size}
        return digest_cache[path]
    for source,items in inventory.items():
        s=config["sources"][source]
        if source=="hoi_retarget":
            common_inputs[source]=[evidence(p) for p in sorted(Path(s["source_root"]).rglob("*")) if p.is_file() and ".cache" not in p.parts]
        elif source=="ultra":
            common_inputs[source]=[evidence(s[key]) for key in ("selection_path","eval_selection_path","bindings_path","assets_path")]
            for r in items: r["_input_files"]=[{"path":r["source_path"],"sha256":r["motion_sha256"]}]
        elif source=="grail_sitting" and not s.get("inventory_override"):
            for r in items:
                paths=[r.get(k) for k in ("robot_path","object_path","meta_path","object_asset_path")]
                if any(p is None or not Path(p).is_file() for p in paths):
                    r["_unavailable_at_freeze"]=True;r["_input_files"]=[]
                else: r["_input_files"]=[evidence(p) for p in paths]
        elif source=="omnicontact" and not s.get("inventory_override"):
            raise ValueError("OmniContact requires SHA-bound inventory override")
        elif source in ("omnicontact","grail_sitting"):
            for r in items:
                if not r.get("_input_files"): raise ValueError(source+" override record lacks SHA-bound inputs")
                for item in r["_input_files"]:
                    if not item.get("path") or len(item.get("sha256",""))!=64: raise ValueError("Invalid frozen source identity")
    atomic_json(out/"inventory.json",inventory)
    atomic_json(out/"common_inputs.json",common_inputs)
    code_root=Path(__file__).resolve().parent.parent
    code=[{"path":str(p.relative_to(code_root)),"sha256":sha256(p)} for p in sorted(code_root.rglob("*.py")) if ".venv" not in p.parts]
    packages={}
    for name in ("numpy","scipy","trimesh","python-fcl","usd-core","pyarrow","torch"):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]="not_installed"
    manifest={"config_sha256":sha256(out/"config.json"),"robot_geometry_sha256":sha256(out/"robot_geometry.json"),
              "inventory_sha256":sha256(out/"inventory.json"),"common_inputs_sha256":sha256(out/"common_inputs.json"),"code":code,"runtime_packages":packages,
              "counts":{s:len(r) for s,r in inventory.items()},"time_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    manifest["recipe_id"]=identity(manifest)
    atomic_json(out/"manifest.json",manifest)
    print(canonical_json({"frozen":str(out),"counts":manifest["counts"],"recipe_id":manifest["recipe_id"]}),flush=True)

def verify(frozen):
    frozen=Path(frozen);manifest=read(frozen/"manifest.json")
    for file,key in (("config.json","config_sha256"),("robot_geometry.json","robot_geometry_sha256"),("inventory.json","inventory_sha256"),("common_inputs.json","common_inputs_sha256")):
        if sha256(frozen/file)!=manifest[key]: raise ValueError("Frozen file changed: "+file)
    root=Path(__file__).resolve().parent.parent
    for item in manifest["code"]:
        if sha256(root/item["path"])!=item["sha256"]: raise ValueError("Frozen code changed: "+item["path"])
    config=read(frozen/"config.json")
    if sha256(config["urdf_path"])!=config["urdf_sha256"]: raise ValueError("URDF changed")
    robot=read(frozen/"robot_geometry.json")
    for dep in robot["dependencies"]:
        if sha256(dep["path"])!=dep["sha256"]: raise ValueError("Robot dependency changed")
    return config,robot,manifest,read(frozen/"inventory.json")

def check_files(items,cache):
    for item in items:
        path=Path(item["path"]);stat=path.stat();fingerprint=(str(path),stat.st_size,stat.st_mtime_ns)
        if fingerprint not in cache: cache[fingerprint]=sha256(path)
        if cache[fingerprint]!=item["sha256"]: raise ValueError("Frozen source SHA mismatch: "+str(path))
        if "size" in item and stat.st_size!=item["size"]: raise ValueError("Frozen source size mismatch: "+str(path))

def run(frozen,output,source,shard,shards,limit=None):
    config,robot,manifest,inventory=verify(frozen)
    if not 0<=shard<shards: raise ValueError("Invalid shard")
    selected=[r for i,r in enumerate(inventory[source]) if i%shards==shard]
    if limit is not None: selected=selected[:limit]
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    claim=root/"claims"/f"{source}-{shard:04d}-of-{shards:04d}.json"
    claim.parent.mkdir(exist_ok=True)
    with claim.open("x",encoding="utf8") as f:
        f.write(canonical_json({"pid":os.getpid(),"host":socket.gethostname(),"job_id":os.getenv("SLURM_JOB_ID"),"recipe_id":manifest["recipe_id"],"count":len(selected)}))
    labeler=Labeler(robot,config["urdf_path"],hoi_retarget.G1_JOINT_NAMES,config["on_m"],config["off_m"],config["target_hz"])
    summary={"source":source,"shard":shard,"shards":shards,"selected":len(selected),"labeled":0,"review":0,"failed":0,"entries":[],"recipe_id":manifest["recipe_id"]}
    checked={};check_files(read(Path(frozen)/"common_inputs.json").get(source,[]),checked)
    iterator=None
    if source=="hoi_retarget":
        iterator=iter(hoi_retarget.iter_clips(**config["sources"][source],split_index=split_index(config),row_indices=[r["row_index"] for r in selected],on_error="yield"))
    for record in selected:
        started=time.time();clip_id=record.get("clip_id",record.get("robot_path",str(record)))
        name=identity({"source":source,"clip_id":clip_id})[:24]
        destination=root/source/name
        entry={"clip_id":clip_id,"output":str(destination)}
        streamed=None
        if iterator is not None:
            try: streamed=next(iterator)
            except Exception as exc:
                summary.update(completed=False,fatal_stream_error=str(exc),fatal_stream_traceback=traceback.format_exc())
                atomic_json(root/"summaries"/claim.name,summary)
                raise
        try:
            if record.get("_unavailable_at_freeze"): raise ValueError("Incomplete source pair at freeze; requires a new inventory after ingest")
            check_files(record.get("_input_files",[]),checked)
            clip=streamed if iterator is not None else load(config,source,record)
            if "_adapter_error" in clip: raise ValueError("Adapter: "+str(clip["_adapter_error"]))
            if clip["clip_id"]!=record["clip_id"]: raise ValueError("Stream identity differs from frozen inventory")
            arrays,metadata=labeler.label(clip)
            geometry=metadata.pop("object_geometry");asset_hash=metadata["object_geometry_sha256"]
            asset_dir=root/"object_geometry";asset_dir.mkdir(exist_ok=True)
            asset_path=asset_dir/(asset_hash+".json")
            if asset_path.exists():
                if identity(read(asset_path))!=asset_hash: raise ValueError("Asset cache collision")
            else: atomic_json(asset_path,geometry)
            metadata["object_geometry_path"]=str(asset_path)
            metadata["recipe_id"]=manifest["recipe_id"]
            save_clip(destination,arrays,metadata)
            entry.update(status="labeled",review_flags=metadata["review_flags"],statistics=metadata["statistics"])
            summary["labeled"]+=1
            if metadata["review_flags"]: summary["review"]+=1
        except Exception as exc:
            entry.update(status="requires_review",error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            summary["failed"]+=1
        entry["seconds"]=time.time()-started;summary["entries"].append(entry)
        print(canonical_json({k:v for k,v in entry.items() if k not in ("traceback","statistics")}),flush=True)
        atomic_json(root/"summaries"/claim.name,summary)
    summary["completed"]=True;summary["completed_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    atomic_json(root/"summaries"/claim.name,summary)
    if selected and summary["labeled"]==0: raise RuntimeError("No clip labeled in this shard; inspect review summary")

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("freeze");p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("run");p.add_argument("--frozen",required=True);p.add_argument("--output",required=True)
    p.add_argument("--source",required=True,choices=("ultra","hoi_retarget","grail_sitting","omnicontact"))
    p.add_argument("--shard",type=int,default=0);p.add_argument("--shards",type=int,default=1);p.add_argument("--limit",type=int)
    args=vars(parser.parse_args());command=args.pop("command")
    if command=="freeze": args["config_path"]=args.pop("config");freeze(**args)
    else: run(**args)

if __name__=="__main__": main()
