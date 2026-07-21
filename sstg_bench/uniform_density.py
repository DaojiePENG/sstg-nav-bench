"""Oracle-semantic density ablation on annotation-independent map nodes."""
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .benchmark import load_scene_episodes,make_sim,scene_paths
from .experiments import CATEGORIES,cached_goal_distance,evaluate_scene,prepare_scene,summarize


def run(args):
    root=Path(args.root).resolve();dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split
    scenes=root/"data/hm3d"/args.scene_dir;maps=root/args.maps;output=root/args.output;output.mkdir(parents=True,exist_ok=True)
    fractions=[float(x) for x in args.fractions.split(",")];rows=defaultdict(list);scene_stats={}
    for _,data in load_scene_episodes(dataset):
        _,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"]);sim=make_sim(base,nav)
        mapping=json.loads((maps/scene/"vlm_topological_map.json").read_text());nodes=mapping["nodes"]
        path_cache,goal_cache=prepare_scene(sim,data,nodes)
        for node in nodes:
            labels=[category for category in CATEGORIES if cached_goal_distance(goal_cache,node,category)<=args.success_distance]
            node["categories"]=labels;node["category_scores"]={category:1. for category in labels}
        for fraction in fractions:
            count=max(1,int(math.ceil(len(nodes)*fraction)))
            rows[fraction]+=evaluate_scene(data,nodes[:count],path_cache,goal_cache,scene,success_distance=args.success_distance)
        scene_stats[scene]={"nodes":len(nodes),"labelled_nodes":sum(bool(n["categories"]) for n in nodes)};sim.close()
    summaries={}
    table=[]
    for fraction in fractions:
        summary=summarize(f"uniform_density_{fraction:g}",rows[fraction]);summaries[str(fraction)]=summary
        table.append({"node_fraction":fraction,"episodes":summary["episodes"],"sr":summary["sr"],"spl":summary["spl"],"dtg":summary["dtg"]})
        episode_path=output/f"episodes_uniform_density_{fraction:g}.csv"
        with episode_path.open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=rows[fraction][0].keys());writer.writeheader();writer.writerows(rows[fraction])
    with (output/"density_ablation.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=table[0].keys());writer.writeheader();writer.writerows(table)
    (output/"density_ablation.json").write_text(json.dumps({"protocol":"goal-independent nodes; oracle semantics assigned only for evaluation","scenes":scene_stats,"summaries":summaries},indent=2))
    fig,ax=plt.subplots(figsize=(6,4));ax.plot([x["node_fraction"] for x in table],[x["sr"] for x in table],"o-",label="SR")
    ax.plot([x["node_fraction"] for x in table],[x["spl"] for x in table],"s-",label="SPL")
    ax.set(xlabel="Fraction of nested farthest-point topology",ylabel="Metric",ylim=(0,1.05));ax.grid(alpha=.3);ax.legend();fig.tight_layout()
    fig.savefig(output/"density_ablation.png",dpi=200);fig.savefig(output/"density_ablation.pdf");plt.close(fig);print(json.dumps(table,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--split",default="val_mini")
    p.add_argument("--scene-dir",default="minival");p.add_argument("--maps",default="outputs/hm3d_minival_uniform/source")
    p.add_argument("--output",default="outputs/hm3d_minival_uniform/density_analysis")
    p.add_argument("--fractions",default="0.25,0.5,0.75,1.0");p.add_argument("--success-distance",type=float,default=1.0);run(p.parse_args())


if __name__=="__main__":main()
