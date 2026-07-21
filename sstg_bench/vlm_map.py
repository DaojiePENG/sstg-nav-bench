"""Build a VLM-labelled semantic topology from multi-view Habitat observations."""
from __future__ import annotations
import argparse, json, math, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .benchmark import build_topology_edges, load_scene_episodes, scene_paths, make_sim, render
from .vlm import annotate, CATEGORIES


def yaw_quaternion(base_xyzw, delta_degrees):
    # HM3D episode rotations are yaw-only [0, y, 0, w].
    y,w=float(base_xyzw[1]),float(base_xyzw[3])
    yaw=2*math.atan2(y,w)+math.radians(delta_degrees)
    return [0.0, math.sin(yaw/2), 0.0, math.cos(yaw/2)]


def panorama(sim, node, offsets=None):
    offsets=offsets or [0,-45,45,180]
    views=[render(sim,node["position"],yaw_quaternion(node["rotation"],d)) for d in offsets]
    canvas=Image.new("RGB",(640,360))
    draw=ImageDraw.Draw(canvas)
    for i,(arr,d) in enumerate(zip(views,offsets)):
        im=Image.fromarray(arr).resize((320,180))
        x=(i%2)*320; y=(i//2)*180; canvas.paste(im,(x,y))
        draw.rectangle((x,y,x+65,y+18),fill=(0,0,0)); draw.text((x+4,y+2),f"yaw {d:+d}",fill=(255,255,255))
    return canvas


def oracle_nodes(data):
    nodes=[]
    for goals in data["goals_by_category"].values():
        for goal in goals:
            if not goal.get("view_points"): continue
            vp=max(goal["view_points"],key=lambda x:x.get("iou",0)); st=vp["agent_state"]
            nodes.append({"id":len(nodes),"position":st["position"],"rotation":st["rotation"],
                          "oracle_category":goal["object_category"],"object_id":goal.get("object_id"),
                          "object_position":goal["position"],"iou":vp.get("iou",0),
                          "source":"coverage_viewpoint_multiview"})
    return nodes


PROMPT_VERSION="primary-anchor-v2"

def annotate_one(path,key,url,model):
    started=time.time()
    try:
        parsed,raw=annotate(path,key,url,model)
        detections=parsed.get("detections",[])
        scores={x.get("category"):float(x.get("confidence",0)) for x in detections if x.get("category") in CATEGORIES}
        preds=[x for x in parsed.get("visible_categories",[]) if x in CATEGORIES]
        if not preds: preds=list(scores)
        primary=parsed.get("primary_category") if parsed.get("primary_category") in CATEGORIES else None
        return {"status":"ok","prompt_version":PROMPT_VERSION,"predicted_categories":preds,
                "primary_category":primary,"primary_confidence":float(parsed.get("primary_confidence",scores.get(primary,0)) or 0),
                "category_scores":scores,"description":parsed.get("description",""),
                "response_id":parsed.get("_response_id"),"usage":parsed.get("_usage",{}),
                "raw":raw,"latency_s":round(time.time()-started,3)}
    except Exception as e:
        return {"status":"error","prompt_version":PROMPT_VERSION,"predicted_categories":[],"error":repr(e),"latency_s":round(time.time()-started,3)}


def run(args):
    root=Path(args.root).resolve(); dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split
    scenes=root/"data/hm3d"/args.scene_dir; out=root/args.output; out.mkdir(parents=True,exist_ok=True)
    key=os.getenv("PeterAI_KEY")
    if not key and not args.cache_only: raise SystemExit("PeterAI_KEY is not set")
    jobs=[]; maps={}
    for _,data in load_scene_episodes(dataset):
        sid,name,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"])
        sim=make_sim(base,nav); nodes=oracle_nodes(data); scene_dir=out/name; scene_dir.mkdir(exist_ok=True)
        for n in nodes:
            p=scene_dir/f"node_{n['id']:04d}_panorama.jpg"
            if not p.exists(): panorama(sim,n).save(p,quality=90)
            jobs.append((name,n,p))
        maps[name]={"scene":sid,"nodes":nodes,"edges":build_topology_edges(sim,nodes)}; sim.close()
    cache_path=out/"vlm_responses.json"; cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    # Resume safely and retry only failed transport/model calls. Successful
    # responses are immutable cache entries and are never billed twice.
    todo=[x for x in jobs if str(x[2].relative_to(root)) not in cache
          or cache[str(x[2].relative_to(root))].get("status")!="ok"
          or cache[str(x[2].relative_to(root))].get("prompt_version")!=PROMPT_VERSION]
    if args.cache_only:todo=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(annotate_one,p,key,args.base_url,args.model):(name,n,p) for name,n,p in todo}
        for i,f in enumerate(as_completed(futures),1):
            name,n,p=futures[f]; rel=str(p.relative_to(root)); cache[rel]=f.result()
            cache_path.write_text(json.dumps(cache,indent=2)); print(f"[{i}/{len(todo)}] {rel}: {cache[rel]['predicted_categories']}")
    for name,m in maps.items():
        for n in m["nodes"]:
            rel=str((out/name/f"node_{n['id']:04d}_panorama.jpg").relative_to(root))
            n["observation_path"]=rel; n["vlm"]=cache[rel]
            n["categories_all"]=cache[rel]["predicted_categories"]
            n["categories_primary"]=[cache[rel]["primary_category"]] if cache[rel].get("primary_category") else []
            n["categories"]=n["categories_primary"]
            n["category_scores"]=cache[rel].get("category_scores",{})
        (out/name/"vlm_topological_map.json").write_text(json.dumps(m,indent=2))
    rows=[n for m in maps.values() for n in m["nodes"]]
    ok=[n for n in rows if n["vlm"]["status"]=="ok"]
    report={"model":args.model,"nodes":len(rows),"api_success":len(ok)/len(rows),
            "primary_node_recall":sum(n["oracle_category"] in n["categories_primary"] for n in rows)/len(rows),
            "all_detection_node_recall":sum(n["oracle_category"] in n["categories_all"] for n in rows)/len(rows),
            "macro_primary_recall":{c:sum(n["oracle_category"]==c and c in n["categories_primary"] for n in rows)/max(1,sum(n["oracle_category"]==c for n in rows)) for c in CATEGORIES},
            "empty_predictions":sum(not n["categories"] for n in rows)}
    (out/"mapping_report.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="."); ap.add_argument("--output",default="outputs/hm3d_minival_vlm/maps")
    ap.add_argument("--split",default="val_mini"); ap.add_argument("--scene-dir",default="minival")
    ap.add_argument("--model",default="gpt-5.5"); ap.add_argument("--base-url",default="https://api.peterai.cc.cd/v1"); ap.add_argument("--workers",type=int,default=6)
    ap.add_argument("--cache-only",action="store_true",help="Rebuild maps/report without making API calls.")
    run(ap.parse_args())
if __name__=="__main__": main()
