"""Local VLM inference with view-localized detections for RGB-D fusion."""
from __future__ import annotations
import argparse,json,re,time
from pathlib import Path

from .vlm import CATEGORIES

PROMPT_VERSION="rgbd-absolute-bbox-v5-600k"
PROMPT="""This is a 2x2 indoor panorama made from four full-resolution 90-degree views at one robot position:
top-left=view 0 yaw 0, top-right=view 1 yaw 90, bottom-left=view 2 yaw 180, bottom-right=view 3 yaw 270.
Detect only clearly visible ObjectNav objects from: chair, bed, plant, toilet, tv_monitor, sofa.
Do not guess from room context. Empty detections are valid. Ground every detection with a tight bbox_2d using ABSOLUTE coordinates in the original 1280x720 panorama: [x1,y1,x2,y2]. A box must stay entirely inside one tile and must never cross x=640 or y=360. Use at most one clearest detection per category per tile. Map televisions and monitors to tv_monitor.
Return a JSON list only, following Qwen2.5-VL's grounding format. Each object must contain exactly the keys bbox_2d, label, and confidence. Compute coordinates from this image; do not use placeholder or example coordinates. Only include confidence >= 0.55."""

VIEW_PROMPT="""Detect only clearly visible ObjectNav objects in this indoor image.
Allowed labels: chair, bed, plant, toilet, tv_monitor, sofa. Map televisions
and computer monitors to tv_monitor. Do not infer objects from room context.
Return a JSON list only. Each detection must be
{"bbox_2d":[x1,y1,x2,y2],"label":"allowed label","confidence":0..1},
where coordinates use Qwen's 0..1000 scale within this image. Empty [] is a
valid answer. Include only visually grounded detections with confidence >=0.55."""

VIEW_RETRY_PROMPT="""Ground visible chair, bed, plant, toilet, tv_monitor, or
sofa instances in this image. Return only [] or a JSON list of objects with
label and bbox_2d. bbox_2d is [x1,y1,x2,y2] on a 0..1000 image coordinate
scale. Do not describe the room and do not output Markdown."""


def parse(raw):
    cleaned=raw.replace("```json","").replace("```","")
    # Grounding output is normally a JSON list, including the important valid
    # empty response ``[]``.  Parse it before looking for an object wrapper so
    # empty scenes are not incorrectly recorded as inference failures.
    list_match=re.search(r"\[.*\]",cleaned,re.S)
    object_match=re.search(r"\{.*\}",cleaned,re.S)
    if list_match:
        data=json.loads(list_match.group(0))
    elif object_match:
        data=json.loads(object_match.group(0))
    else:
        raise ValueError("no JSON: "+raw[:300])
    source=data if isinstance(data,list) else data.get("detections",[]);detections=[]
    for detection in source:
        category=detection.get("label",detection.get("category"));bbox=detection.get("bbox_2d")
        if category not in CATEGORIES or not isinstance(bbox,list) or len(bbox)!=4:continue
        try:confidence=float(detection.get("confidence",.7));x1,y1,x2,y2=map(float,bbox)
        except Exception:continue
        if confidence<.55 or not (0<=x1<x2<=1280 and 0<=y1<y2<=720):continue
        left_column=0 if x1<640 else 1;right_column=0 if x2<=640 else 1
        top_row=0 if y1<360 else 1;bottom_row=0 if y2<=360 else 1
        if left_column!=right_column or top_row!=bottom_row:continue
        cx=(x1+x2)/2;cy=(y1+y2)/2;column=left_column;row=top_row;view=row*2+column
        detections.append({"category":category,"confidence":confidence,"view_index":view,
                           "center":[(cx-column*640)/640,(cy-row*360)/360],"bbox_2d":[x1,y1,x2,y2]})
    return detections,""


def parse_view(raw,view_index):
    cleaned=raw.replace("```json","").replace("```","")
    match=re.search(r"\[.*\]",cleaned,re.S)
    if not match:
        raise ValueError("no JSON list: "+raw[:300])
    payload=match.group(0)
    try:
        data=json.loads(payload)
    except json.JSONDecodeError:
        # Local Qwen occasionally omits a comma between otherwise valid
        # grounding fields. Recover only explicit box/label pairs; never infer
        # a category or coordinate that is absent from the response.
        data=[]
        for object_text in re.findall(r"\{[^{}]*\}",payload,re.S):
            box_match=re.search(r'"bbox_2d"\s*:\s*\[([^]]+)\]',object_text)
            label_match=re.search(r'"(?:label|category)"\s*:\s*"([^"]+)"',object_text)
            confidence_match=re.search(r'"confidence"\s*:\s*([0-9.]+)',object_text)
            if not box_match or not label_match:continue
            try:box=[float(value.strip()) for value in box_match.group(1).split(",")]
            except ValueError:continue
            data.append({"bbox_2d":box,"label":label_match.group(1),
                         "confidence":float(confidence_match.group(1)) if confidence_match else .7})
        if not data and payload.strip()!="[]":
            raise
    detections=[]
    for detection in data:
        category=detection.get("label",detection.get("category"));bbox=detection.get("bbox_2d")
        if category not in CATEGORIES or not isinstance(bbox,list) or len(bbox)!=4:continue
        try:confidence=float(detection.get("confidence",.7));x1,y1,x2,y2=map(float,bbox)
        except Exception:continue
        if confidence<.55 or not (0<=x1<x2<=1000 and 0<=y1<y2<=1000):continue
        normalized=[x1/1000,y1/1000,x2/1000,y2/1000]
        detections.append({"category":category,"confidence":confidence,"view_index":view_index,
                           "center":[(x1+x2)/2000,(y1+y2)/2000],"bbox_norm":normalized})
    return detections


def infer(model,processor,image,max_new_tokens,prompt=PROMPT):
    import torch
    from qwen_vl_utils import process_vision_info

    messages=[{"role":"user","content":[{"type":"image","image":str(image)},{"type":"text","text":prompt}]}]
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);images,videos=process_vision_info(messages)
    inputs=processor(text=[text],images=images,videos=videos,padding=True,return_tensors="pt").to("cuda")
    with torch.inference_mode():generated=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
    trimmed=[output[len(source):] for source,output in zip(inputs.input_ids,generated)]
    return processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]


def run(args):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration

    root=Path(args.root).resolve();source=root/args.source;output=root/args.output;output.mkdir(parents=True,exist_ok=True)
    cache_path=output/"rgbd_vlm_responses.json";cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,dtype=torch.float16,device_map="cuda",attn_implementation="sdpa",local_files_only=True)
    processor=AutoProcessor.from_pretrained(args.model,min_pixels=256*28*28,max_pixels=args.max_visual_tokens*28*28,local_files_only=True)
    jobs=[];maps=[]
    for path in sorted(source.glob("*/rgbd_topological_map.json")):
        mapping=json.loads(path.read_text());maps.append((path.parent.name,mapping))
        for node in mapping["nodes"]:
            image=root/node["observation_path"];jobs.append((path.parent.name,node,image,str(image.relative_to(root))))
    todo=[job for job in jobs if cache.get(job[3],{}).get("status")!="ok" or cache.get(job[3],{}).get("prompt_version")!=PROMPT_VERSION]
    if args.limit is not None:todo=todo[:args.limit]
    for index,(scene,node,image,key) in enumerate(todo,1):
        started=time.time()
        try:
            raw=infer(model,processor,image,args.max_new_tokens);detections,description=parse(raw)
            cache[key]={"status":"ok","prompt_version":PROMPT_VERSION,"model":args.model,"detections":detections,
                        "description":description,"raw":raw,"latency_s":round(time.time()-started,3)}
        except Exception as primary_error:
            # Some panoramas trigger degenerate text generation in the local
            # model.  Repair only those nodes with four simpler view-local
            # grounding calls, preserving the same RGB-D observations.
            try:
                detections=[];raw_views=[]
                for view_index,view in enumerate(node["rgbd_views"]):
                    view_path=root/view["rgb_path"]
                    try:
                        raw_view=infer(model,processor,view_path,args.max_new_tokens,VIEW_PROMPT)
                        parsed_view=parse_view(raw_view,view_index)
                    except Exception:
                        try:
                            raw_view=infer(model,processor,view_path,args.max_new_tokens,VIEW_RETRY_PROMPT)
                            parsed_view=parse_view(raw_view,view_index)
                        except Exception:
                            # A deterministic vision-token degeneration can be
                            # image-grid specific. Re-encode the same view at a
                            # nearby patch grid before the final retry.
                            from PIL import Image
                            repair_dir=output/"_repair_inputs";repair_dir.mkdir(exist_ok=True)
                            repaired=repair_dir/f"{scene}_node_{node['id']:04d}_view{view_index}.jpg"
                            image=Image.open(view_path).convert("RGB")
                            image.resize((608,342)).save(repaired,quality=92)
                            raw_view=infer(model,processor,repaired,args.max_new_tokens,VIEW_RETRY_PROMPT)
                            parsed_view=parse_view(raw_view,view_index)
                    raw_views.append(raw_view);detections.extend(parsed_view)
                cache[key]={"status":"ok","prompt_version":PROMPT_VERSION,"model":args.model,
                            "detections":detections,"description":"","raw_views":raw_views,
                            "repair_mode":"four_view_local_grounding","primary_error":repr(primary_error),
                            "latency_s":round(time.time()-started,3)}
            except Exception as error:
                cache[key]={"status":"error","prompt_version":PROMPT_VERSION,"model":args.model,
                            "detections":[],"error":repr(error),"primary_error":repr(primary_error)}
        cache_path.write_text(json.dumps(cache,indent=2));print(f"[{index}/{len(todo)}] {key}: {len(cache[key]['detections'])} detections",flush=True)
    if args.limit is not None:return
    for scene,mapping in maps:
        for node in mapping["nodes"]:node["localized_vlm"]=cache[node["observation_path"]]
        scene_out=output/scene;scene_out.mkdir(exist_ok=True);(scene_out/"rgbd_semantic_map.json").write_text(json.dumps(mapping,indent=2))
    actual=[cache[node["observation_path"]] for _,mapping in maps for node in mapping["nodes"]]
    report={"model":args.model,"prompt_version":PROMPT_VERSION,"nodes":len(actual),"inference_success":sum(x["status"]=="ok" for x in actual)/len(actual),
            "detections":sum(len(x.get("detections",[])) for x in actual),"empty_nodes":sum(not x.get("detections") for x in actual),
            "panorama_nodes":sum(x.get("status")=="ok" and not x.get("repair_mode") for x in actual),
            "view_local_repair_nodes":sum(x.get("repair_mode")=="four_view_local_grounding" for x in actual)}
    (output/"semantic_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--source",default="outputs/hm3d_minival_uniform/rgbd_capture")
    p.add_argument("--output",default="outputs/hm3d_minival_uniform/rgbd_semantics");p.add_argument("--model",default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--max-new-tokens",type=int,default=350);p.add_argument("--max-visual-tokens",type=int,default=768)
    p.add_argument("--limit",type=int,default=None);run(p.parse_args())


if __name__=="__main__":main()
