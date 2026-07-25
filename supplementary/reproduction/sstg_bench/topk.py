"""Sequential top-K candidate evaluation with explicit oracle-feedback labeling.

This is a diagnostic upper bound: after visiting each candidate, the evaluator
checks distance to official ObjectNav goals. A deployable agent must replace
that oracle check with a visual/depth stop verifier.
"""
from __future__ import annotations

import argparse,csv,json,math
from pathlib import Path

import numpy as np

from .benchmark import load_scene_episodes,make_sim,scene_paths,shortest
from .experiments import bootstrap_ci,cached_goal_distance,load_nodes,prepare_scene,wilson


RANKING_STRATEGIES=("category_score","support_confidence","confidence_support")


def candidate_rank_key(node,category,distance,strategy="category_score"):
    """Return a stable query-time ordering key without evaluator signals."""
    score=float(node.get("category_scores",{}).get(category,0.))
    confidence=float(node.get("confidence",score))
    support=int(node.get("cluster_support",1))
    if strategy=="category_score":return (-score,distance)
    if strategy=="support_confidence":return (-support,-confidence,distance)
    if strategy=="confidence_support":return (-confidence,-support,distance)
    raise ValueError(f"unknown ranking strategy: {strategy}")


def diverse_candidates(ranked,max_k,min_separation):
    """Greedily keep candidates from different spatial hypotheses.

    Adjacent topology nodes often observe the same object (or the same visual
    false positive).  Counting them as separate attempts wastes the Top-K
    budget, so positive ``min_separation`` makes the diagnostic visit distinct
    semantic hypotheses.  Zero preserves the original confidence ranking.
    """
    selected=[]
    for _,_,node in ranked:
        position=np.asarray(node["position"],dtype=float)
        if min_separation>0 and any(
            np.linalg.norm(position-np.asarray(previous["position"],dtype=float))<min_separation
            for previous in selected
        ):
            continue
        selected.append(node)
        if len(selected)>=max_k:break
    return selected


def rank_nodes(sim,start,nodes,category,ranking_strategy):
    ranked=[]
    for node in nodes:
        if category not in node.get("categories_all",node.get("categories",[])):continue
        ok,distance,_=shortest(sim,start,node["position"])
        if ok and math.isfinite(distance):
            ranked.append((candidate_rank_key(node,category,distance,ranking_strategy),distance,node))
    ranked.sort(key=lambda item:item[0])
    return ranked


def append_distinct(selected,ranked,count,max_k,min_separation):
    added=0
    for _,_,node in ranked:
        position=np.asarray(node["position"],dtype=float)
        if any(np.linalg.norm(position-np.asarray(previous["position"],dtype=float))<max(1e-4,min_separation)
               for previous in selected):continue
        selected.append(node);added+=1
        if added>=count or len(selected)>=max_k:break
    return selected


def hierarchical_candidates(primary_ranked,fallback_ranked,max_k,min_separation,primary_k,secondary_ranked=None,secondary_k=0):
    """Select primary representatives, optional middle-tier candidates, then distinct residual standoffs."""
    selected=diverse_candidates(primary_ranked,min(primary_k,max_k),min_separation)
    if secondary_ranked and secondary_k>0:
        append_distinct(selected,secondary_ranked,secondary_k,max_k,min_separation)
    return append_distinct(selected,fallback_ranked,max_k-len(selected),max_k,min_separation)


def evaluate_episode(sim,data,episode,nodes,goal_cache,scene,max_k,success_distance,min_separation,ranking_strategy,primary_nodes=None,primary_k=1,secondary_nodes=None,secondary_k=0,secondary_ranking_strategy="category_score"):
    category=episode["object_category"];start=episode["start_position"];ranked=[]
    ranked=rank_nodes(sim,start,nodes,category,ranking_strategy)
    if primary_nodes is None:
        candidates=diverse_candidates(ranked,max_k,min_separation)
    else:
        primary_ranked=rank_nodes(sim,start,primary_nodes,category,"category_score")
        secondary_ranked=rank_nodes(sim,start,secondary_nodes,category,secondary_ranking_strategy) if secondary_nodes is not None else None
        candidates=hierarchical_candidates(primary_ranked,ranked,max_k,min_separation,primary_k,secondary_ranked,secondary_k)
    current=start;cumulative=0.;attempts=[];success_at=None
    for index,node in enumerate(candidates,1):
        ok,leg,route=shortest(sim,current,node["position"])
        if not ok or not math.isfinite(leg):continue
        cumulative+=leg;dtg=cached_goal_distance(goal_cache,node,category)
        success=dtg<=success_distance
        attempts.append({"rank":index,"node_id":node["id"],"confidence":float(node.get("category_scores",{}).get(category,0.)),
                         "leg_distance":leg,"cumulative_distance":cumulative,"dtg":dtg,"success":success,
                         "route":[point.tolist() for point in route]})
        current=node["position"]
        if success:success_at=index;break
    optimal=float(episode["info"]["geodesic_distance"])
    row={"episode":f"{scene}_{category}_{episode['episode_id']}","scene":scene,"category":category,
         "optimal_distance":optimal,"candidate_count":len(ranked),"success_at":success_at or 0,
         "attempted":len(attempts),"planned_distance":cumulative,"attempts":attempts}
    for k in range(1,max_k+1):
        success=float(success_at is not None and success_at<=k)
        if success:
            distance=next(x["cumulative_distance"] for x in attempts if x["rank"]==success_at)
            spl=optimal/max(optimal,distance)
        else:spl=0.
        row[f"success_at_{k}"]=success;row[f"spl_at_{k}"]=spl
    return row


def run(args):
    root=Path(args.root).resolve();dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split
    scenes=root/"data/hm3d"/args.scene_dir;output=root/args.output;output.mkdir(parents=True,exist_ok=True);rows=[]
    for _,data in load_scene_episodes(dataset):
        _,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"]);sim=make_sim(base,nav)
        nodes=load_nodes(root,scene,"vlm_all",vlm_maps=args.maps)
        primary_nodes=load_nodes(root,scene,"vlm_all",vlm_maps=args.primary_maps) if args.primary_maps else None
        secondary_nodes=load_nodes(root,scene,"vlm_all",vlm_maps=args.secondary_maps) if args.secondary_maps else None
        _,goal_cache=prepare_scene(sim,data,nodes)
        for episode in data["episodes"]:
            rows.append(evaluate_episode(sim,data,episode,nodes,goal_cache,scene,args.max_k,args.success_distance,args.min_separation,args.ranking_strategy,primary_nodes,args.primary_k,secondary_nodes,args.secondary_k,args.secondary_ranking_strategy))
        sim.close()
    flat=[]
    for row in rows:
        item={k:v for k,v in row.items() if k!="attempts"};item["attempts_json"]=json.dumps(row["attempts"],separators=(",",":"));flat.append(item)
    with (output/"episodes_topk.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=flat[0].keys());writer.writeheader();writer.writerows(flat)
    summary={"protocol":"oracle success feedback after each visited candidate; diagnostic upper bound","episodes":len(rows),"max_k":args.max_k,
             "ranking_strategy":args.ranking_strategy,
             "primary_maps":args.primary_maps,"primary_k":args.primary_k if args.primary_maps else 0,
             "secondary_maps":args.secondary_maps,"secondary_k":args.secondary_k if args.secondary_maps else 0,
             "secondary_ranking_strategy":args.secondary_ranking_strategy if args.secondary_maps else None,
             "candidate_min_separation_m":args.min_separation,"metrics":{}}
    for k in range(1,args.max_k+1):
        success=sum(row[f"success_at_{k}"] for row in rows);spl=[row[f"spl_at_{k}"] for row in rows]
        summary["metrics"][str(k)]={"successes":int(success),"sr":success/len(rows),"sr_95ci":wilson(success,len(rows)),
                                      "spl":float(np.mean(spl)),"spl_95ci":bootstrap_ci(spl,seed=k)}
    (output/"summary_topk.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--split",default="val_mini")
    p.add_argument("--scene-dir",default="minival");p.add_argument("--maps",default="outputs/hm3d_minival_uniform/qwen_maps")
    p.add_argument("--output",default="outputs/hm3d_minival_uniform/topk_analysis");p.add_argument("--max-k",type=int,default=3)
    p.add_argument("--success-distance",type=float,default=1.0)
    p.add_argument("--ranking-strategy",choices=RANKING_STRATEGIES,default="category_score")
    p.add_argument("--primary-maps",default=None,help="Optional representative map used for the first candidate(s).")
    p.add_argument("--primary-k",type=int,default=1)
    p.add_argument("--secondary-maps",default=None,help="Optional map supplying intermediate recovery candidates.")
    p.add_argument("--secondary-k",type=int,default=0)
    p.add_argument("--secondary-ranking-strategy",choices=RANKING_STRATEGIES,default="category_score")
    p.add_argument("--min-separation",type=float,default=0.,help="Greedy Euclidean separation between visited hypotheses (m).")
    run(p.parse_args())


if __name__=="__main__":main()
