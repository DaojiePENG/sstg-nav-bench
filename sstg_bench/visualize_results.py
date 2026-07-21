"""Render representative first-person/top-down videos from an episode CSV."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .benchmark import draw_map,load_scene_episodes,make_sim,render,save_video_artifacts,scene_paths,shortest


def display_nodes(nodes):
    result=[]
    for node in nodes:
        item=dict(node);scores=item.get("category_scores",{})
        item["category"]=max(scores,key=scores.get) if scores else "unlabeled";result.append(item)
    return result


def run(args):
    root=Path(args.root).resolve();dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split
    scenes=root/"data/hm3d"/args.scene_dir;maps=root/args.maps;output=root/args.output
    videos=output/"videos";failures=output/"failures";videos.mkdir(parents=True,exist_ok=True);failures.mkdir(exist_ok=True)
    rows={r["episode"]:r for r in csv.DictReader((root/args.episodes).open())}
    success_left=args.successes;failure_left=args.failures;rendered=[]
    for _,data in load_scene_episodes(dataset):
        _,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"])
        map_path=maps/scene/"vlm_topological_map.json"
        if not map_path.exists():continue
        nodes=json.loads(map_path.read_text())["nodes"];draw_nodes=display_nodes(nodes);sim=make_sim(base,nav)
        for episode in data["episodes"]:
            key=f"{scene}_{episode['object_category']}_{episode['episode_id']}";row=rows.get(key)
            if not row:continue
            if args.only and key not in args.only:continue
            success=float(row["success"])>0.5;is_failure=not success
            if success and success_left<=0:continue
            if is_failure and failure_left<=0:continue
            target=next((n for n in nodes if str(n["id"])==row.get("target_node","")),None)
            if target is None:
                rgb=render(sim,episode["start_position"],episode["start_rotation"])
                top=draw_map(sim,draw_nodes,current=np.asarray(episode["start_position"]),title=f"FAIL {key}: no semantic candidate")
                imageio.mimsave(videos/f"{key}_FAIL_first_person.mp4",[rgb]*16,fps=args.fps)
                imageio.mimsave(videos/f"{key}_FAIL_topdown.mp4",[top]*16,fps=args.fps)
            else:
                ok,_,route=shortest(sim,episode["start_position"],target["position"])
                if not ok:continue
                stem=key if success else f"{key}_FAIL"
                save_video_artifacts(sim,output,stem,draw_nodes,route,target,{"fps":args.fps})
            if success:success_left-=1
            else:
                failure_left-=1
                (failures/f"{key}.json").write_text(json.dumps({"episode":episode,"metrics":row,"selected_node":target},indent=2))
            rendered.append({"episode":key,"success":success,"category":episode["object_category"],"failure":row.get("failure","")})
        sim.close()
        if success_left<=0 and failure_left<=0:break
    (output/"visualized_episodes.json").write_text(json.dumps(rendered,indent=2));print(json.dumps(rendered,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--split",default="val")
    p.add_argument("--scene-dir",default="val");p.add_argument("--maps",default="outputs/hm3d_val_qwen/maps")
    p.add_argument("--episodes",default="outputs/hm3d_val_qwen_analysis/episodes_vlm_all_confidence.csv")
    p.add_argument("--output",default="outputs/hm3d_val_qwen/visuals");p.add_argument("--successes",type=int,default=4)
    p.add_argument("--failures",type=int,default=6);p.add_argument("--fps",type=int,default=8)
    p.add_argument("--only",nargs="*",default=[],help="Optional exact episode keys to render.")
    run(p.parse_args())


if __name__=="__main__":main()
