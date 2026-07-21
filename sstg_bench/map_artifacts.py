"""Render semantic-map galleries and a flat pose/observation manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

from .benchmark import draw_map,load_scene_episodes,make_sim,scene_paths


def labelled_nodes(nodes):
    result=[]
    for node in nodes:
        item=dict(node);scores=item.get("category_scores",{})
        item["category"]=max(scores,key=scores.get) if scores else "unlabeled"
        result.append(item)
    return result


def contact_sheet(items,path,columns=6,size=(320,260)):
    if not items:return
    rows=(len(items)+columns-1)//columns
    canvas=Image.new("RGB",(columns*size[0],rows*size[1]),"white");draw=ImageDraw.Draw(canvas)
    for i,(label,image_path) in enumerate(items):
        image=Image.open(image_path).convert("RGB");image.thumbnail((size[0],size[1]-24))
        x=(i%columns)*size[0]+(size[0]-image.width)//2;y=(i//columns)*size[1]+24
        canvas.paste(image,(x,y));draw.text(((i%columns)*size[0]+5,(i//columns)*size[1]+5),label,fill="black")
    canvas.save(path,quality=90)


def run(args):
    root=Path(args.root).resolve();maps=root/args.maps;output=root/args.output;output.mkdir(parents=True,exist_ok=True)
    semantic_dir=output/"semantic_maps";semantic_dir.mkdir(exist_ok=True)
    dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split;scenes=root/"data/hm3d"/args.scene_dir
    manifests=[];map_gallery=[];observation_gallery=[];counts=Counter()
    for _,data in load_scene_episodes(dataset):
        _,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"])
        map_path=maps/scene/"vlm_topological_map.json"
        if not map_path.exists():continue
        mapping=json.loads(map_path.read_text());nodes=labelled_nodes(mapping["nodes"])
        sim=make_sim(base,nav);preview=draw_map(sim,nodes,title=f"{scene}: VLM semantic topology")
        preview_path=semantic_dir/f"{scene}.png";imageio.imwrite(preview_path,preview);sim.close();map_gallery.append((scene,preview_path))
        if nodes:
            observation_path=nodes[0].get("observation_path")
            if observation_path:
                obs=root/observation_path
                if obs.is_file():observation_gallery.append((scene,obs))
        for node in nodes:
            labels=node.get("categories_all",[]);counts.update(labels)
            p=node["position"];q=node.get("rotation",[0,0,0,1])
            manifests.append({"scene":scene,"node_id":node["id"],"x":p[0],"y":p[1],"z":p[2],
                              "qx":q[0],"qy":q[1],"qz":q[2],"qw":q[3],
                              "primary_category":node.get("categories_primary",[None])[0] if node.get("categories_primary") else "",
                              "all_categories":"|".join(labels),"category_scores":json.dumps(node.get("category_scores",{}),sort_keys=True),
                              "observation_path":node.get("observation_path","")})
    if manifests:
        with (output/"node_pose_manifest.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=manifests[0].keys());writer.writeheader();writer.writerows(manifests)
    contact_sheet(map_gallery,output/"semantic_map_contact_sheet.jpg")
    contact_sheet(observation_gallery,output/"observation_contact_sheet.jpg",size=(320,220))
    fig,ax=plt.subplots(figsize=(7,4));cats=sorted(counts);ax.bar(cats,[counts[c] for c in cats],color="#26a69a")
    ax.set(ylabel="VLM-labelled topology nodes",title=f"Semantic-map label distribution ({len(manifests)} nodes)")
    ax.tick_params(axis="x",rotation=25);fig.tight_layout();fig.savefig(output/"label_distribution.png",dpi=200);plt.close(fig)
    summary={"scenes":len(map_gallery),"nodes":len(manifests),"label_counts":dict(counts),
             "pose_manifest":str((output/"node_pose_manifest.csv").relative_to(root))}
    (output/"artifact_report.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--split",default="val")
    p.add_argument("--scene-dir",default="val");p.add_argument("--maps",default="outputs/hm3d_val_qwen/maps")
    p.add_argument("--output",default="outputs/hm3d_val_qwen/visuals");run(p.parse_args())


if __name__=="__main__":main()
