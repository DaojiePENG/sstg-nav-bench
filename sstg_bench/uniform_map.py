"""Build an annotation-independent, uniformly covered pre-map.

Unlike the target-view coverage upper bound in :mod:`vlm_map`, sampling here
never reads ObjectNav goal viewpoints. Official goals are used only later by
the evaluator. This is the protocol to use when measuring a deployable
pre-mapping pass rather than isolating semantic recognition.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from .benchmark import load_scene_episodes, map_pixel, scene_paths, make_sim, shortest, topdown_base
from .vlm_map import panorama


def sample_pool(pathfinder, count: int, seed: int) -> np.ndarray:
    pathfinder.seed(seed)
    points=[]
    for _ in range(count):
        p=np.asarray(pathfinder.get_random_navigable_point(),dtype=np.float32)
        if np.isfinite(p).all(): points.append(p)
    if not points:
        raise RuntimeError("Habitat returned no navigable samples")
    return np.asarray(points)


def farthest_cover(pool: np.ndarray, radius: float, max_nodes: int) -> tuple[np.ndarray,float]:
    """Greedy farthest-point cover over an annotation-independent sample pool."""
    center=np.mean(pool,axis=0)
    first=int(np.argmin(np.sum((pool-center)**2,axis=1)))
    selected=[first]
    nearest_sq=np.sum((pool-pool[first])**2,axis=1)
    while len(selected)<max_nodes:
        index=int(np.argmax(nearest_sq))
        if nearest_sq[index] <= radius*radius:
            break
        selected.append(index)
        nearest_sq=np.minimum(nearest_sq,np.sum((pool-pool[index])**2,axis=1))
    return pool[selected],float(math.sqrt(float(np.max(nearest_sq))))


def build_edges(sim, points: np.ndarray, neighbors: int, max_edge: float):
    edges=[]
    for i,p in enumerate(points):
        distances=np.linalg.norm(points-p,axis=1)
        for j in np.argsort(distances)[1:neighbors+1]:
            j=int(j)
            if j<=i or distances[j]>max_edge:
                continue
            ok,geodesic,_=shortest(sim,p,points[j])
            if ok and math.isfinite(geodesic) and geodesic<=max_edge*1.6:
                edges.append({"source":i,"target":j,"geodesic_distance":geodesic})
    return edges


def preview(sim, nodes, edges, title):
    base=topdown_base(sim)
    fig,ax=plt.subplots(figsize=(8,8));ax.imshow(base,cmap="gray",origin="upper")
    for edge in edges:
        a=map_pixel(sim,nodes[edge["source"]]["position"],base.shape)
        b=map_pixel(sim,nodes[edge["target"]]["position"],base.shape)
        ax.plot([a[0],b[0]],[a[1],b[1]],color="#26a69a",lw=.45,alpha=.35)
    xy=np.asarray([map_pixel(sim,n["position"],base.shape) for n in nodes])
    ax.scatter(xy[:,0],xy[:,1],s=8,c="#ffca28",edgecolors="none")
    ax.set_title(title);ax.axis("off");fig.tight_layout();fig.canvas.draw()
    image=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();plt.close(fig);return image


def run(args):
    root=Path(args.root).resolve()
    dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split
    scenes=root/"data/hm3d"/args.scene_dir
    output=root/args.output;output.mkdir(parents=True,exist_ok=True)
    report={"protocol":"annotation-independent farthest-point pre-map","split":args.split,"scenes":{}}
    for scene_index,(_,data) in enumerate(load_scene_episodes(dataset)):
        sid,name,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"])
        sim=make_sim(base,nav)
        pool=sample_pool(sim.pathfinder,args.pool_size,args.seed+scene_index)
        points,empirical_radius=farthest_cover(pool,args.cover_radius,args.max_nodes)
        nodes=[];scene_out=output/name;scene_out.mkdir(exist_ok=True)
        for i,point in enumerate(points):
            node={"id":i,"position":point.tolist(),"rotation":[0.,0.,0.,1.],
                  "source":"uniform_farthest_point","sampling_seed":args.seed+scene_index}
            if not args.skip_panorama:
                path=scene_out/f"node_{i:04d}_panorama.jpg"
                if not path.exists(): panorama(sim,node,[0,90,180,270]).save(path,quality=90)
                node["observation_path"]=str(path.relative_to(root))
            nodes.append(node)
        edges=build_edges(sim,points,args.neighbors,args.max_edge)
        mapping={"scene":sid,"nodes":nodes,"edges":edges,"sampling":{
            "goal_annotations_used":False,"pool_size":len(pool),"cover_radius_requested_m":args.cover_radius,
            "empirical_pool_cover_radius_m":empirical_radius,"seed":args.seed+scene_index}}
        (scene_out/"vlm_topological_map.json").write_text(json.dumps(mapping,indent=2))
        imageio.imwrite(scene_out/"topology.png",preview(sim,nodes,edges,f"{name}: {len(nodes)} nodes, r={empirical_radius:.2f} m"))
        report["scenes"][name]={"nodes":len(nodes),"edges":len(edges),"empirical_pool_cover_radius_m":empirical_radius}
        sim.close()
    report["total_nodes"]=sum(x["nodes"] for x in report["scenes"].values())
    (output/"sampling_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--split",default="val_mini")
    p.add_argument("--scene-dir",default="minival");p.add_argument("--output",default="outputs/hm3d_minival_uniform/source")
    p.add_argument("--pool-size",type=int,default=12000);p.add_argument("--cover-radius",type=float,default=.8)
    p.add_argument("--max-nodes",type=int,default=1200);p.add_argument("--neighbors",type=int,default=8)
    p.add_argument("--max-edge",type=float,default=2.4);p.add_argument("--seed",type=int,default=20260719)
    p.add_argument("--skip-panorama",action="store_true",help="Skip redundant RGB panoramas when a later RGB-D capture pass will be used.")
    run(p.parse_args())


if __name__=="__main__":main()
