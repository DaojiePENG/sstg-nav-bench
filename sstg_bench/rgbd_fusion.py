"""Back-project view-localized VLM detections and fuse them in 3D."""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .benchmark import load_scene_episodes,make_sim,scene_paths,shortest


def median_depth(path,center,max_depth):
    depth=np.load(path,mmap_mode="r");height,width=depth.shape
    x=int(round(center[0]*(width-1)));y=int(round(center[1]*(height-1)))
    patch=np.asarray(depth[max(0,y-3):min(height,y+4),max(0,x-3):min(width,x+4)])
    values=patch[np.isfinite(patch)&(patch>.2)&(patch<max_depth)]
    return float(np.median(values)) if values.size else None


def project(node,view,detection,depth):
    # Habitat depth is camera-forward Z depth. Camera frame is +X right,
    # +Y up and -Z forward; the sensor is 1.25 m above the agent state.
    width,height=view.get("resolution",[640,360]);hfov=math.radians(float(view.get("hfov_deg",90)))
    vfov=2*math.atan((height/width)*math.tan(hfov/2))
    x=(detection["center"][0]-.5)*2*math.tan(hfov/2)*depth
    y=-(detection["center"][1]-.5)*2*math.tan(vfov/2)*depth;z=-depth
    rotation=node.get("rotation",[0,0,0,1]);base_yaw=2*math.atan2(float(rotation[1]),float(rotation[3]))
    yaw=base_yaw+math.radians(view["yaw_offset_deg"]);c=math.cos(yaw);s=math.sin(yaw)
    px,py,pz=node["position"]
    return np.asarray([px+c*x+s*z,py+1.25+y,pz-s*x+c*z],dtype=np.float32)


def candidate_from_detection(sim,root,node,detection,standoff,max_depth):
    view=node["rgbd_views"][detection["view_index"]];depth=median_depth(root/view["depth_path"],detection["center"],max_depth)
    if depth is None:return None
    object_position=project(node,view,detection,depth);source=np.asarray(node["position"],dtype=np.float32)
    direction=source[[0,2]]-object_position[[0,2]];norm=float(np.linalg.norm(direction))
    if norm<.2:return None
    desired=object_position.copy();desired[[0,2]]+=direction/norm*standoff;desired[1]=source[1]
    snapped=np.asarray(sim.pathfinder.snap_point(desired),dtype=np.float32)
    if not np.isfinite(snapped).all():return None
    ok,_,_=shortest(sim,source,snapped)
    if not ok:return None
    return {"position":snapped.tolist(),"rotation":node.get("rotation",[0,0,0,1]),
            "object_estimate":object_position.tolist(),"source_topology_node":node["id"],"source_position":node["position"],
            "view_index":detection["view_index"],"center":detection["center"],"depth_m":depth,
            "bbox_norm":detection.get("bbox_norm"),"bbox_2d":detection.get("bbox_2d"),
            "source_rgb_path":view["rgb_path"],"source_depth_path":view["depth_path"],
            "category":detection["category"],"confidence":detection["confidence"],"observation_path":node["observation_path"]}


def components(sim,candidates,radius,vertical_tolerance,max_stop_geodesic):
    parent=list(range(len(candidates)))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        a,b=find(i),find(j)
        if a!=b:parent[b]=a
    for i,a in enumerate(candidates):
        pa=np.asarray(a["object_estimate"])
        for j in range(i):
            pb=np.asarray(candidates[j]["object_estimate"])
            if abs(pa[1]-pb[1])>vertical_tolerance or np.linalg.norm(pa[[0,2]]-pb[[0,2]])>radius:
                continue
            # Euclidean proximity alone can merge detections through walls or
            # across disconnected floor islands.  Such candidates must remain
            # independent navigation hypotheses.
            reachable,geodesic,_=shortest(sim,a["position"],candidates[j]["position"])
            if reachable and math.isfinite(geodesic) and geodesic<=max_stop_geodesic:
                union(i,j)
    groups=defaultdict(list)
    for i,item in enumerate(candidates):groups[find(i)].append(item)
    return list(groups.values())


def as_node(candidate,node_id,score,support,cluster_size):
    category=candidate["category"];result=dict(candidate);result.update({"id":node_id,"categories":[category],"categories_all":[category],
        "categories_primary":[category],"category_scores":{category:score},"cluster_support":support,"cluster_detections":cluster_size})
    return result


def run(args):
    root=Path(args.root).resolve();source=root/args.source;dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split
    scenes=root/"data/hm3d"/args.scene_dir;output=root/args.output;raw_root=output/"raw_maps";cluster_root=output/"clustered_maps"
    multi_root=output/"multi_standoff_maps"
    raw_root.mkdir(parents=True,exist_ok=True);cluster_root.mkdir(parents=True,exist_ok=True);multi_root.mkdir(parents=True,exist_ok=True);report={"scenes":{}}
    for _,data in load_scene_episodes(dataset):
        _,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"]);path=source/scene/"rgbd_semantic_map.json"
        if not path.exists():continue
        mapping=json.loads(path.read_text());sim=make_sim(base,nav);candidates=[]
        for node in mapping["nodes"]:
            for detection in node.get("localized_vlm",{}).get("detections",[]):
                item=candidate_from_detection(sim,root,node,detection,args.standoff,args.max_depth)
                if item:candidates.append(item)
        raw_nodes=[as_node(item,index,item["confidence"],1,1) for index,item in enumerate(candidates)]
        fused=[];multi_standoff=[]
        for category in sorted({item["category"] for item in candidates}):
            category_items=[item for item in candidates if item["category"]==category]
            for group in components(sim,category_items,args.cluster_radius,args.vertical_tolerance,args.max_stop_geodesic):
                support=len({item["source_topology_node"] for item in group});best=max(group,key=lambda x:x["confidence"])
                if support<args.min_support and best["confidence"]<args.keep_single_confidence:continue
                # Independent observations increase confidence without allowing
                # duplicate boxes from the same frame to inflate support.
                source_confidence={}
                for item in group:
                    source_node_id=item["source_topology_node"]
                    source_confidence[source_node_id]=max(source_confidence.get(source_node_id,0.),item["confidence"])
                score=1.-float(np.prod([1.-min(.99,value) for value in source_confidence.values()]))
                cluster_id=len(fused)
                representative=as_node(best,cluster_id,score,support,len(group));representative["fusion_cluster_id"]=cluster_id
                representative["is_cluster_representative"]=True;fused.append(representative)
                # Preserve the validated standoff alternatives of every kept
                # evidence cluster.  Query-time Top-K can therefore retain the
                # fused semantic score while choosing distinct reachable
                # stopping poses instead of collapsing all recovery geometry
                # into the single highest-confidence representative.
                for member in group:
                    node=as_node(member,len(multi_standoff),score,support,len(group))
                    node["fusion_cluster_id"]=cluster_id
                    node["is_cluster_representative"]=member is best
                    multi_standoff.append(node)
        metadata={"scene":mapping["scene"],"base_topology":str(path.relative_to(root)),"fusion":{"standoff_m":args.standoff,
                  "max_depth_m":args.max_depth,"cluster_radius_m":args.cluster_radius,
                  "max_stop_geodesic_m":args.max_stop_geodesic,"min_support":args.min_support}}
        for directory,nodes in ((raw_root,raw_nodes),(cluster_root,fused),(multi_root,multi_standoff)):
            scene_out=directory/scene;scene_out.mkdir(exist_ok=True);(scene_out/"vlm_topological_map.json").write_text(json.dumps(dict(metadata,nodes=nodes,edges=[]),indent=2))
        report["scenes"][scene]={"localized_detections":sum(len(n.get("localized_vlm",{}).get("detections",[])) for n in mapping["nodes"]),
                                 "depth_valid_candidates":len(raw_nodes),"fused_candidates":len(fused),
                                 "fused_standoff_candidates":len(multi_standoff),
                                 "fused_by_category":dict((c,sum(n["category"]==c for n in fused)) for c in sorted({n["category"] for n in fused}))}
        sim.close()
    report["raw_candidates"]=sum(x["depth_valid_candidates"] for x in report["scenes"].values());report["fused_candidates"]=sum(x["fused_candidates"] for x in report["scenes"].values())
    report["fused_standoff_candidates"]=sum(x["fused_standoff_candidates"] for x in report["scenes"].values())
    (output/"fusion_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--split",default="val_mini");p.add_argument("--scene-dir",default="minival")
    p.add_argument("--source",default="outputs/hm3d_minival_uniform/rgbd_semantics");p.add_argument("--output",default="outputs/hm3d_minival_uniform/rgbd_fusion")
    p.add_argument("--standoff",type=float,default=.8);p.add_argument("--max-depth",type=float,default=6.)
    p.add_argument("--cluster-radius",type=float,default=1.2);p.add_argument("--vertical-tolerance",type=float,default=1.)
    p.add_argument("--max-stop-geodesic",type=float,default=3.,help="Maximum navmesh distance for candidates fused into one object (m).")
    p.add_argument("--min-support",type=int,default=2);p.add_argument("--keep-single-confidence",type=float,default=.92);run(p.parse_args())


if __name__=="__main__":main()
