"""Render representative real-VLM successes and every real-VLM failure."""
import argparse, json, math
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

from .benchmark import (load_scene_episodes, scene_paths, make_sim, shortest,
                        save_video_artifacts, draw_map, render)


def run(root):
    root=Path(root).resolve(); dataset=root/"data/datasets/objectnav_hm3d_v2/val_mini"; scenes=root/"data/hm3d/minival"
    out=root/"outputs/hm3d_minival_vlm"; (out/"videos").mkdir(parents=True,exist_ok=True); (out/"failures").mkdir(exist_ok=True)
    successes_left=4
    for _,data in load_scene_episodes(dataset):
        _,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"]); sim=make_sim(base,nav)
        nodes=json.loads((out/"maps"/scene/"vlm_topological_map.json").read_text())["nodes"]
        draw_nodes=[]
        for n in nodes:
            x=dict(n); scores=x.get("category_scores",{}); x["category"]=max(scores,key=scores.get) if scores else "unlabeled"; draw_nodes.append(x)
        for e in data["episodes"]:
            cat=e["object_category"]; candidates=[]
            for n in nodes:
                if cat not in n.get("categories_all",[]): continue
                ok,d,route=shortest(sim,e["start_position"],n["position"])
                if ok: candidates.append((float(n.get("category_scores",{}).get(cat,0)),d,n,route))
            ep=f"{scene}_{e['object_category']}_{e['episode_id']}"
            if not candidates:
                start_rgb=render(sim,e["start_position"],e["start_rotation"])
                gt=[dict(n,category=n["oracle_category"]) for n in nodes if n["oracle_category"]==cat]
                td=draw_map(sim,gt,current=np.asarray(e["start_position"]),title=f"FAIL: {ep} | no VLM candidate for {cat}")
                imageio.imwrite(out/"failures"/f"{ep}_start.jpg",start_rgb); imageio.imwrite(out/"failures"/f"{ep}_topdown.png",td)
                imageio.mimsave(out/"videos"/f"{ep}_FAIL_first_person.mp4",[start_rgb]*16,fps=8)
                imageio.mimsave(out/"videos"/f"{ep}_FAIL_topdown.mp4",[td]*16,fps=8)
                (out/"failures"/f"{ep}.json").write_text(json.dumps({"episode":e,"reason":"no_semantic_candidate","available_node_labels":[n.get("categories_all",[]) for n in nodes]},indent=2))
                continue
            score,d,target,route=max(candidates,key=lambda x:(x[0],-x[1]))
            if successes_left:
                save_video_artifacts(sim,out,ep,draw_nodes,route,target,{"fps":8}); successes_left-=1
        sim.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=".");run(ap.parse_args().root)
if __name__=="__main__":main()
