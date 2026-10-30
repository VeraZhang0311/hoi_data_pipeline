"""One complete real reference per source, with a local OmniContact source fixture."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time
from .cli import verify,load,check_files
from .adapters import omnicontact,hoi_retarget
from .pipeline import Labeler,save_clip,atomic_json

def main():
    frozen,output,omni_root=sys.argv[1:]
    config,robot,manifest,inventory=verify(frozen)
    labeler=Labeler(robot,config["urdf_path"],hoi_retarget.G1_JOINT_NAMES,config["on_m"],config["off_m"],config["target_hz"])
    output=Path(output);output.mkdir(parents=True,exist_ok=False);results=[]
    for source in ("ultra","hoi_retarget","grail_sitting","omnicontact"):
        started=time.time()
        if source=="omnicontact":
            paths=sorted(Path(omni_root).glob("npz_clips/box/case1_carry/*.npz"))
            if len(paths)!=1:raise ValueError("Smoke fixture must contain exactly one carry NPZ")
            clip=omnicontact.load_clip(paths[0],source_root=omni_root)
        else:
            rows=inventory[source]
            record=next(r for r in rows if all(Path(i["path"]).is_file() for i in r.get("_input_files",[])))
            check_files(record.get("_input_files",[]),{})
            clip=load(config,source,record)
        print(json.dumps({"source":source,"clip":clip["clip_id"],"states":len(clip["timestamps_s"]),"status":"labeling_complete_reference"}),flush=True)
        arrays,metadata=labeler.label(clip)
        metadata["recipe_id"]=manifest["recipe_id"]
        save_clip(output/source,arrays,metadata)
        record={"source":source,"clip_id":clip["clip_id"],"seconds":time.time()-started,"statistics":metadata["statistics"],"review_flags":metadata["review_flags"]}
        results.append(record);atomic_json(output/"results.json",results)
        print(json.dumps({"source":source,"saved":True,"seconds":record["seconds"]}),flush=True)
    atomic_json(output/"ALL_SOURCES_PASSED.json",{"recipe_id":manifest["recipe_id"],"sources":len(results)})

if __name__=="__main__":main()
