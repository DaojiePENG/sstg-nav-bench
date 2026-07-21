"""Evaluate oracle, real-VLM, and controlled degradation variants."""
from __future__ import annotations
import argparse, csv, json, math, random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import habitat_sim

from .benchmark import load_scene_episodes, scene_paths, make_sim, shortest
from .vlm_map import oracle_nodes

CATEGORIES=["chair","bed","plant","toilet","tv_monitor","sofa"]


def goal_viewpoints(data, category):
    pts=[]
    for goals in data["goals_by_category"].values():
        for g in goals:
            if g["object_category"]==category:
                pts.extend(v["agent_state"]["position"] for v in g.get("view_points",[]))
    return pts


def load_nodes(root, scene, backend, data=None, vlm_maps="outputs/hm3d_minival_vlm/maps"):
    if backend=="oracle":
        nodes=oracle_nodes(data)
        for n in nodes: n["categories"]=[n["oracle_category"]]
        return nodes
    p=root/vlm_maps/scene/"vlm_topological_map.json"
    nodes=json.loads(p.read_text())["nodes"]
    field="categories_all" if backend.startswith("vlm_all") else "categories_primary"
    for n in nodes: n["categories"]=n.get(field,[])
    return nodes


def wilson(success,n,z=1.96):
    if not n:return [0.,0.]
    p=success/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]


def bootstrap_ci(values,seed=0,n_boot=10000):
    a=np.asarray(values,float)
    if not len(a): return [0.,0.]
    rng=np.random.default_rng(seed); means=np.mean(rng.choice(a,(n_boot,len(a)),replace=True),axis=1)
    return np.quantile(means,[.025,.975]).tolist()


def prepare_scene(sim,data,nodes):
    # Lazy caches avoid O(episodes*all_nodes) and O(nodes*all_viewpoints)
    # precomputation on the 1000-episode full validation split.
    points={category:goal_viewpoints(data,category) for category in CATEGORIES}
    return {"sim":sim,"paths":{}},{"sim":sim,"data":data,"goal_points":points,"distances":{}}

def cached_path(cache,e,n):
    # Habitat ObjectNav reuses numeric episode_id across goal categories in a
    # scene. Include category and start pose or paths from different episodes
    # can be silently reused, corrupting candidate ranking and SPL.
    # Node ids are only unique *within one map*.  Oracle, target-view VLM and
    # independent maps all restart ids at zero, so using n["id"] here silently
    # reused oracle paths for unrelated independent-topology positions.
    key=(e["object_category"],e["episode_id"],tuple(e["start_position"]),tuple(float(x) for x in n["position"]))
    if key not in cache["paths"]:
        ok,d,pts=shortest(cache["sim"],e["start_position"],n["position"])
        cache["paths"][key]=(d,pts) if ok else (float("inf"),[])
    return cache["paths"][key]

def cached_goal_distance(cache,n,cat):
    # As above, map-local ids cannot identify a spatial candidate across
    # evaluation variants. Goal distance is a function of its actual pose.
    key=(tuple(float(x) for x in n["position"]),cat)
    if key not in cache["distances"]:
        # Only evaluator-created oracle nodes may short-circuit to zero.  A
        # VLM node's ``category`` is a prediction, not ground truth; treating
        # it as oracle truth makes every depth-projected detection succeed.
        if n.get("oracle_category")==cat:
            cache["distances"][key]=0.0
        else:
            # Habitat's multi-goal query returns the exact geodesic distance
            # to the closest official success viewpoint in one graph search.
            # The former loop launched one shortest-path search per viewpoint,
            # which is equivalent but prohibitively slow at full validation
            # scale (6,642 topology nodes and thousands of goal viewpoints).
            points=cache.get("goal_points",{}).get(cat)
            if points is None:
                points=goal_viewpoints(cache["data"],cat)
            if points and hasattr(cache["sim"],"pathfinder"):
                query=habitat_sim.MultiGoalShortestPath()
                query.requested_start=np.asarray(n["position"],dtype=np.float32)
                query.requested_ends=np.asarray(points,dtype=np.float32)
                ok=cache["sim"].pathfinder.find_path(query)
                cache["distances"][key]=float(query.geodesic_distance) if ok else float("inf")
            elif points:
                ds=[]
                for point in points:
                    ok,d,_=shortest(cache["sim"],n["position"],point)
                    if ok and math.isfinite(d):
                        ds.append(d)
                cache["distances"][key]=min(ds) if ds else float("inf")
            else:
                cache["distances"][key]=float("inf")
    return cache["distances"][key]


def mutate_labels(nodes,dropout=0.,false_positive=0.,keep_probability=1.,seed=0):
    rng=random.Random(seed); result=[]
    for n in nodes:
        if rng.random()>keep_probability: continue
        x=dict(n); labels=list(n.get("categories",[]))
        labels=[c for c in labels if rng.random()>=dropout]
        if rng.random()<false_positive:
            wrong=rng.choice([c for c in CATEGORIES if c not in labels]); labels.append(wrong)
        x["categories"]=labels; result.append(x)
    return result


def evaluate_scene(data,nodes,cache,goal_cache,scene,success_distance=1.0,rank_mode="distance"):
    rows=[]
    for e in data["episodes"]:
        cat=e["object_category"]
        candidates=[]
        for n in nodes:
            if cat in n.get("categories",[]):
                d,route=cached_path(cache,e,n)
                if math.isfinite(d): candidates.append((d,n,route,float(n.get("category_scores",{}).get(cat,0))))
        if not candidates:
            optimal=float(e["info"]["geodesic_distance"])
            rows.append({"episode":f"{scene}_{cat}_{e['episode_id']}","scene":scene,"category":cat,"success":0.,"spl":0.,"dtg":optimal,"optimal_distance":optimal,"planned_distance":0.,"target_node":"","failure":"no_semantic_candidate"}); continue
        if rank_mode=="confidence_support":
            # Noisy-OR scores can round to exactly one for strongly supported
            # clusters. Break those ties with independent source-pose support
            # before path length, rather than letting floating-point saturation
            # discard the multi-view evidence used by fusion.
            planned,n,route,_=min(candidates,key=lambda x:(-x[3],-int(x[1].get("cluster_support",0)),x[0]))
        elif rank_mode=="confidence":
            planned,n,route,_=min(candidates,key=lambda x:(-x[3],x[0]))
        else:
            planned,n,route,_=min(candidates,key=lambda x:x[0])
        dtg=cached_goal_distance(goal_cache,n,cat)
        success=float(dtg<=success_distance); optimal=float(e["info"]["geodesic_distance"])
        rows.append({"episode":f"{scene}_{cat}_{e['episode_id']}","scene":scene,"category":cat,"success":success,
                     "spl":success*optimal/max(optimal,planned),"dtg":dtg,"optimal_distance":optimal,
                     "planned_distance":planned,"target_node":n["id"],"failure":"" if success else "wrong_semantic_candidate"})
    return rows


def summarize(name,rows):
    succ=sum(r["success"] for r in rows); spl=[r["spl"] for r in rows]
    finite_dtg=[r["dtg"] for r in rows if math.isfinite(r["dtg"])]
    bycat={}
    for c in CATEGORIES:
        rr=[r for r in rows if r["category"]==c]
        bycat[c]={"n":len(rr),"sr":float(np.mean([x["success"] for x in rr])) if rr else None,"spl":float(np.mean([x["spl"] for x in rr])) if rr else None}
    return {"method":name,"episodes":len(rows),"successes":int(succ),"sr":succ/len(rows),"sr_95ci":wilson(succ,len(rows)),
            "spl":float(np.mean(spl)),"spl_95ci":bootstrap_ci(spl),"dtg":float(np.mean(finite_dtg)) if finite_dtg else None,
            "failure_counts":dict(Counter(r["failure"] or "success" for r in rows)),"per_category":bycat}


def run(root,split="val_mini",scene_dir="minival",vlm_maps="outputs/hm3d_minival_vlm/maps",output="outputs/analysis",stress_seeds=20,include_vlm=True):
    root=Path(root).resolve(); dataset=root/"data/datasets/objectnav_hm3d_v2"/split; scenes=root/"data/hm3d"/scene_dir
    out=root/output; out.mkdir(parents=True,exist_ok=True)
    variant_specs=(("oracle","oracle","distance"),("vlm_all_nearest","vlm_all","distance"),
                   ("vlm_all_confidence","vlm_all","confidence"),("vlm_primary","vlm_primary","distance"))
    if not include_vlm: variant_specs=variant_specs[:1]
    variant_rows={name:[] for name,_,_ in variant_specs}
    stress_rows=defaultdict(list)
    for _,data in load_scene_episodes(dataset):
        _,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"]); sim=make_sim(base,nav)
        oracle=load_nodes(root,scene,"oracle",data); cache,goals=prepare_scene(sim,data,oracle)
        for backend,node_backend,rank_mode in variant_specs:
            nodes=oracle if backend=="oracle" else load_nodes(root,scene,node_backend,vlm_maps=vlm_maps)
            variant_rows[backend]+=evaluate_scene(data,nodes,cache,goals,scene,rank_mode=rank_mode)
        for keep in (1.0,.75,.5,.25):
          for drop in (0.,.1,.25,.5):
           for fp in (0.,.05,.15):
            for seed in range(stress_seeds):
                stress_rows[(keep,drop,fp,seed)]+=evaluate_scene(data,mutate_labels(oracle,drop,fp,keep,seed),cache,goals,scene)
        sim.close()
    variants={name:(rows,summarize(name,rows)) for name,rows in variant_rows.items()}
    stress=[]
    for (keep,drop,fp,seed),rows in stress_rows.items():
        s=summarize("stress",rows); stress.append({"keep_probability":keep,"semantic_dropout":drop,"false_positive":fp,"seed":seed,"sr":s["sr"],"spl":s["spl"],"dtg":s["dtg"],"failures":len(rows)-s["successes"]})
    # Detailed and aggregated machine-readable tables.
    for name,(rows,s) in variants.items():
        keys=rows[0].keys()
        with (out/f"episodes_{name}.csv").open("w",newline="") as f: w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
        (out/f"summary_{name}.json").write_text(json.dumps(s,indent=2))
    if stress:
      with (out/"stress_runs.csv").open("w",newline="") as f: w=csv.DictWriter(f,stress[0].keys());w.writeheader();w.writerows(stress)
    groups=defaultdict(list)
    for x in stress: groups[(x["keep_probability"],x["semantic_dropout"],x["false_positive"])].append(x)
    agg=[]
    for key,xs in groups.items():
        agg.append({"keep_probability":key[0],"semantic_dropout":key[1],"false_positive":key[2],
                    "sr_mean":np.mean([x["sr"] for x in xs]),"sr_std":np.std([x["sr"] for x in xs],ddof=1),
                    "spl_mean":np.mean([x["spl"] for x in xs]),"spl_std":np.std([x["spl"] for x in xs],ddof=1)})
    if agg:
      with (out/"stress_aggregate.csv").open("w",newline="") as f: w=csv.DictWriter(f,agg[0].keys());w.writeheader();w.writerows(agg)
    # Paper-ready main result table and compact sensitivity plot.
    summaries={k:v[1] for k,v in variants.items()}; (out/"all_summaries.json").write_text(json.dumps(summaries,indent=2))
    tex="\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Episodes & SR $\\uparrow$ & SPL $\\uparrow$ & DTG $\\downarrow$ \\\\\n\\midrule\n"
    for name,_,_ in variant_specs:
        s=summaries[name]; tex+=f"SSTG-Nav ({name}) & {s['episodes']} & {s['sr']:.3f} & {s['spl']:.3f} & {s['dtg'] if s['dtg'] is not None else float('nan'):.3f} \\\\\n"
    tex+="\\bottomrule\n\\end{tabular}\n"; (out/"ral_results_table.tex").write_text(tex)
    fig,ax=plt.subplots(figsize=(7,4))
    if agg:
      selected=[x for x in agg if x["keep_probability"]==1 and x["false_positive"]==0]
      selected.sort(key=lambda x:x["semantic_dropout"]); ax.errorbar([x["semantic_dropout"] for x in selected],[x["sr_mean"] for x in selected],yerr=[x["sr_std"] for x in selected],marker="o",label="SR")
      ax.set(xlabel="Semantic-node dropout probability",ylabel="Success rate",ylim=(0,1.05));ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(out/"semantic_dropout_sensitivity.pdf");fig.savefig(out/"semantic_dropout_sensitivity.png",dpi=200);plt.close(fig)
    else:plt.close(fig)
    print(json.dumps(summaries,indent=2))


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default=".");ap.add_argument("--split",default="val_mini")
    ap.add_argument("--scene-dir",default="minival");ap.add_argument("--vlm-maps",default="outputs/hm3d_minival_vlm/maps")
    ap.add_argument("--output",default="outputs/analysis");ap.add_argument("--stress-seeds",type=int,default=20)
    ap.add_argument("--skip-vlm",action="store_true")
    a=ap.parse_args();run(a.root,a.split,a.scene_dir,a.vlm_maps,a.output,a.stress_seeds,not a.skip_vlm)
if __name__=="__main__":main()
