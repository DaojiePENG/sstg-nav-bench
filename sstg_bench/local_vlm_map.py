"""Offline Qwen2.5-VL fallback for full-val semantic-map construction."""
from __future__ import annotations
import argparse,json,re,time
from pathlib import Path

import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from .vlm import CATEGORIES

PROMPT=("Label this 2x2 indoor panorama for ObjectNav. TOP-LEFT is the anchor view. Allowed categories: "
        "chair, bed, plant, toilet, tv_monitor, sofa. Return JSON only: "
        '{"primary_category":"category or null","primary_confidence":0.0,'
        '"detections":[{"category":"category","confidence":0.0}],"description":"short"}. '
        "primary_category is the clearest central allowed object in TOP-LEFT. Other detections may use all tiles. "
        "Map TV/monitor to tv_monitor. Confidence is 0 to 1.")

def parse(raw):
    text=raw.replace("```json","").replace("```","")
    m=re.search(r"\{.*\}",text,re.S)
    if not m:raise ValueError("no JSON: "+raw[:300])
    d=json.loads(m.group(0)); scores={}
    for x in d.get("detections",[]):
        if x.get("category") in CATEGORIES:
            try:confidence=float(x.get("confidence",0))
            except:confidence=0.
            scores[x["category"]]=max(scores.get(x["category"],0.),confidence)
    primary=d.get("primary_category") if d.get("primary_category") in CATEGORIES else None
    return d,primary,scores

def infer(model,processor,image,max_new_tokens):
    messages=[{"role":"user","content":[{"type":"image","image":str(image)},{"type":"text","text":PROMPT}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    images,videos=process_vision_info(messages)
    inputs=processor(text=[text],images=images,videos=videos,padding=True,return_tensors="pt").to("cuda")
    with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
    trimmed=[x[len(y):] for y,x in zip(inputs.input_ids,generated)]
    return processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

def run(args):
    root=Path(args.root).resolve();source=root/args.source;out=root/args.output;out.mkdir(parents=True,exist_ok=True)
    cache_path=out/"local_responses.json";cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    # Reparse successful raw responses on every resume. This applies parser
    # fixes (for example, max confidence over duplicate category entries)
    # without repeating an expensive image-model call.
    for response in cache.values():
        if response.get("status")=="ok" and response.get("raw"):
            try:
                parsed,primary,scores=parse(response["raw"])
                response["primary_category"]=primary
                response["primary_confidence"]=float(parsed.get("primary_confidence",scores.get(primary,0)) or 0)
                response["category_scores"]=scores
                response["predicted_categories"]=list(scores)
            except Exception:
                pass
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,dtype=torch.float16,device_map="cuda",attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    )
    processor=AutoProcessor.from_pretrained(
        args.model,min_pixels=256*28*28,max_pixels=768*28*28,
        local_files_only=args.local_files_only,
    )
    maps=[]
    for p in sorted(source.glob("*/vlm_topological_map.json")):
        m=json.loads(p.read_text());maps.append((p.parent.name,p.parent,m))
    jobs=[]
    for scene,scene_dir,m in maps:
        for n in m["nodes"]:
            image=scene_dir/Path(n["observation_path"]).name
            key=str(image.relative_to(root));jobs.append((scene,n,image,key))
    todo=[x for x in jobs if x[3] not in cache or cache[x[3]].get("status")!="ok"]
    if args.limit is not None:
        todo=todo[:args.limit]
    for i,(scene,n,image,key) in enumerate(todo,1):
        started=time.time()
        try:
            raw=infer(model,processor,image,args.max_new_tokens);d,primary,scores=parse(raw)
            cache[key]={"status":"ok","model":args.model,"primary_category":primary,
                        "primary_confidence":float(d.get("primary_confidence",scores.get(primary,0)) or 0),
                        "category_scores":scores,"predicted_categories":list(scores),"description":d.get("description",""),
                        "raw":raw,"latency_s":round(time.time()-started,3)}
        except Exception as e:cache[key]={"status":"error","model":args.model,"predicted_categories":[],"error":repr(e)}
        cache_path.write_text(json.dumps(cache,indent=2));print(f"[{i}/{len(todo)}] {key}: {cache[key].get('predicted_categories')}",flush=True)
    # During a limited smoke test, maps would be incomplete; persist only the
    # response cache so the full resumable run can pick it up later.
    if args.limit is not None:
        print(json.dumps({"processed":len(todo),"cache":str(cache_path)},indent=2))
        return
    for scene,scene_dir,m in maps:
        for n in m["nodes"]:
            image=scene_dir/Path(n["observation_path"]).name;r=cache[str(image.relative_to(root))]
            n["vlm"]=r;n["categories_all"]=r.get("predicted_categories",[])
            n["categories_primary"]=[r["primary_category"]] if r.get("primary_category") else []
            n["categories"]=n["categories_primary"];n["category_scores"]=r.get("category_scores",{})
        d=out/scene;d.mkdir(exist_ok=True);(d/"vlm_topological_map.json").write_text(json.dumps(m,indent=2))
    nodes=[n for _,_,m in maps for n in m["nodes"]];ok=[n for n in nodes if n["vlm"]["status"]=="ok"]
    labelled=[n for n in nodes if n.get("oracle_category")]
    report={"model":args.model,"nodes":len(nodes),"inference_success":len(ok)/len(nodes),
            "primary_node_recall":(sum(n["oracle_category"] in n["categories_primary"] for n in labelled)/len(labelled)) if labelled else None,
            "all_detection_node_recall":(sum(n["oracle_category"] in n["categories_all"] for n in labelled)/len(labelled)) if labelled else None}
    (out/"mapping_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--source",default="outputs/hm3d_val_vlm/maps")
    p.add_argument("--output",default="outputs/hm3d_val_qwen/maps");p.add_argument("--model",default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--limit",type=int,default=None,help="Process at most N pending images (smoke testing).")
    p.add_argument("--max-new-tokens",type=int,default=160)
    p.add_argument("--local-files-only",action=argparse.BooleanOptionalAction,default=True)
    run(p.parse_args())
if __name__=="__main__":main()
