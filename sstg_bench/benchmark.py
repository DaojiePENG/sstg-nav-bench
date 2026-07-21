from __future__ import annotations

import argparse, csv, gzip, json, math, os, time
from pathlib import Path
from typing import Any

import habitat_sim
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import yaml
from habitat_sim.utils.common import quat_from_coeffs


def load_scene_episodes(root: Path):
    for p in sorted((root / "content").glob("*.json.gz")):
        with gzip.open(p, "rt") as f:
            yield p.stem.replace(".json", ""), json.load(f)


def scene_paths(scene_root: Path, scene_id: str):
    sid = scene_id.split("/")[-2]
    name = sid.split("-", 1)[-1]
    base = scene_root / sid / f"{name}.basis.glb"
    nav = scene_root / sid / f"{name}.basis.navmesh"
    return sid, name, base, nav


def make_sim(scene: Path, navmesh: Path, width=640, height=360):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene)
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [height, width]
    sensor.position = [0.0, 1.25, 0.0]
    sensor.hfov = 90
    agent_cfg.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim.pathfinder.load_nav_mesh(str(navmesh))
    return sim


def set_pose(sim, position, rotation=None):
    state = habitat_sim.AgentState()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_coeffs(rotation or [0, 0, 0, 1])
    sim.get_agent(0).set_state(state, reset_sensors=True)


def render(sim, position, rotation):
    set_pose(sim, position, rotation)
    return sim.get_sensor_observations()["rgb"][:, :, :3]


def shortest(sim, start, end):
    q = habitat_sim.ShortestPath()
    q.requested_start = np.asarray(start, np.float32)
    q.requested_end = np.asarray(end, np.float32)
    ok = sim.pathfinder.find_path(q)
    return ok, float(q.geodesic_distance), [np.asarray(x) for x in q.points]


def build_topology_edges(sim, nodes, neighbors=6, candidate_neighbors=14):
    """Connect each node to nearby geodesically reachable topology nodes."""
    if len(nodes)<2:return []
    points=np.asarray([n["position"] for n in nodes],dtype=np.float32);edges=[];seen=set()
    for i,point in enumerate(points):
        linked=0
        for j in np.argsort(np.linalg.norm(points-point,axis=1))[1:candidate_neighbors+1]:
            j=int(j);key=tuple(sorted((i,j)))
            if key in seen:continue
            ok,distance,route=shortest(sim,point,points[j])
            if not ok or not math.isfinite(distance):continue
            seen.add(key);linked+=1
            edges.append({"source":i,"target":j,"geodesic_distance":distance,
                          "route":[p.tolist() for p in route]})
            if linked>=neighbors:break
    return edges


def all_goal_nodes(data):
    nodes = []
    for key, goals in data["goals_by_category"].items():
        for goal in goals:
            cat = goal["object_category"]
            # One high-IoU pose per instance is the semantic topological node.
            if not goal.get("view_points"):
                continue
            vp = max(goal["view_points"], key=lambda x: x.get("iou", 0))
            st = vp["agent_state"]
            nodes.append({"id": len(nodes), "category": cat,
                          "object_id": goal.get("object_id"),
                          "object_position": goal["position"],
                          "position": st["position"], "rotation": st["rotation"],
                          "iou": vp.get("iou", 0), "source": "prebuild_viewpoint"})
    return nodes


def select_target(sim, start, nodes, category):
    candidates = []
    for n in nodes:
        if n["category"] != category:
            continue
        ok, dist, pts = shortest(sim, start, n["position"])
        if ok and math.isfinite(dist):
            candidates.append((dist, n, pts))
    return min(candidates, key=lambda x: x[0]) if candidates else None


def densify(points, spacing=0.25):
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        d = float(np.linalg.norm(b-a)); count = max(1, int(math.ceil(d/spacing)))
        out.extend(a+(b-a)*(i/count) for i in range(1, count+1))
    return out


def yaw_quat(a, b):
    d = b-a; yaw = math.atan2(-float(d[0]), -float(d[2]))
    return [0.0, math.sin(yaw/2), 0.0, math.cos(yaw/2)]


def topdown_base(sim, meters_per_pixel=.05):
    return np.asarray(sim.pathfinder.get_topdown_view(meters_per_pixel, 0.0), dtype=np.uint8)


def map_pixel(sim, p, shape):
    lo, hi = sim.pathfinder.get_bounds()
    x = int((p[0]-lo[0]) / max(1e-6, hi[0]-lo[0]) * (shape[1]-1))
    y = int((p[2]-lo[2]) / max(1e-6, hi[2]-lo[2]) * (shape[0]-1))
    return x, y


def draw_map(sim, nodes, route=None, current=None, title="Semantic topological map"):
    base = topdown_base(sim)
    fig, ax = plt.subplots(figsize=(8, 8)); ax.imshow(base, cmap="gray", origin="upper")
    colors = {c: plt.cm.tab10(i) for i,c in enumerate(sorted({n['category'] for n in nodes}))}
    for n in nodes:
        x,y=map_pixel(sim,n["position"],base.shape); ax.scatter(x,y,s=28,c=[colors[n['category']]])
    if route:
        xy=np.array([map_pixel(sim,p,base.shape) for p in route]); ax.plot(xy[:,0],xy[:,1],"#ff2d55",lw=3)
    if current is not None:
        x,y=map_pixel(sim,current,base.shape); ax.scatter(x,y,s=90,c="#00e5ff",edgecolors="black")
    ax.set_title(title); ax.axis("off"); fig.tight_layout()
    fig.canvas.draw(); img=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy(); plt.close(fig); return img


def save_video_artifacts(sim, out, ep, nodes, route, target, cfg):
    frames_dir=out/"frames"/ep; frames_dir.mkdir(parents=True,exist_ok=True)
    dense=densify(route, .30); ego=[]; top=[]; poses=[]
    for i,p in enumerate(dense):
        q=yaw_quat(p, dense[min(i+1,len(dense)-1)]) if i+1<len(dense) else target["rotation"]
        rgb=render(sim,p,q); ego.append(rgb)
        td=draw_map(sim,nodes,route,p,f"Episode {ep} | step {i+1}/{len(dense)}")
        top.append(td); poses.append({"frame":i,"position":p.tolist(),"rotation_xyzw":q})
        if i in (0,len(dense)-1): imageio.imwrite(frames_dir/f"rgb_{i:04d}.jpg",rgb)
    imageio.mimsave(out/"videos"/f"{ep}_first_person.mp4",ego,fps=cfg["fps"],quality=7)
    imageio.mimsave(out/"videos"/f"{ep}_topdown.mp4",top,fps=cfg["fps"],quality=7)
    (frames_dir/"poses.json").write_text(json.dumps(poses,indent=2))


def run(config_path):
    root=Path(config_path).resolve().parent.parent
    cfg=yaml.safe_load(Path(config_path).read_text())
    for k in ("dataset","scenes","semantic_annots","output"):
        cfg[k]=Path(cfg[k]) if Path(cfg[k]).is_absolute() else root/Path(cfg[k])
    out=cfg["output"]; (out/"videos").mkdir(parents=True,exist_ok=True); (out/"maps").mkdir(exist_ok=True)
    rows=[]; manifest={"config":str(Path(config_path).resolve()),"created":time.strftime('%F %T'),"scenes":{}}
    video_left=int(cfg.get("video_episodes",6))
    for scene_name,data in load_scene_episodes(cfg["dataset"]):
        sid,name,base,nav=scene_paths(cfg["scenes"],data["episodes"][0]["scene_id"])
        sim=make_sim(base,nav); nodes=all_goal_nodes(data)
        scene_dir=out/"maps"/name; scene_dir.mkdir(exist_ok=True)
        # Map construction observations: the exact image and pose attached to each semantic node.
        obs_dir=scene_dir/"observations"; obs_dir.mkdir(exist_ok=True)
        for n in nodes:
            imageio.imwrite(obs_dir/f"node_{n['id']:04d}_{n['category']}.jpg",render(sim,n['position'],n['rotation']))
        edges=build_topology_edges(sim,nodes)
        (scene_dir/"topological_map.json").write_text(json.dumps({"scene":sid,"nodes":nodes,"edges":edges},indent=2))
        imageio.imwrite(scene_dir/"semantic_map.png",draw_map(sim,nodes))
        manifest["scenes"][name]={"nodes":len(nodes),"episodes":len(data["episodes"])}
        for e in data["episodes"]:
            ep=f"{name}_{e['object_category']}_{e['episode_id']}"; chosen=select_target(sim,e["start_position"],nodes,e["object_category"])
            if not chosen:
                rows.append({"episode":ep,"scene":name,"category":e["object_category"],"success":0,"spl":0,"failure":"no_reachable_semantic_node"}); continue
            planned,target,route=chosen
            final=np.asarray(target["position"]); remaining=min(shortest(sim,final,n["position"])[1] for n in nodes if n["category"]==e["object_category"])
            success=float(remaining<=float(cfg["success_distance"])); optimal=float(e["info"]["geodesic_distance"])
            spl=success*optimal/max(optimal,planned)
            row={"episode":ep,"scene":name,"category":e["object_category"],"success":success,"spl":spl,
                 "optimal_distance":optimal,"planned_distance":planned,"final_goal_distance":remaining,
                 "target_node":target["id"],"target_iou":target["iou"],"path_points":len(route),"failure":""}
            rows.append(row)
            (out/"paths").mkdir(exist_ok=True); (out/"paths"/f"{ep}.json").write_text(json.dumps({"episode":e,"selected_node":target,"route":[x.tolist() for x in route],"metrics":row},indent=2))
            if video_left>0:
                save_video_artifacts(sim,out,ep,nodes,route,target,cfg); video_left-=1
        sim.close()
    keys=sorted({k for r in rows for k in r});
    with (out/"episodes.csv").open("w",newline="") as f: w=csv.DictWriter(f,keys); w.writeheader(); w.writerows(rows)
    sr=float(np.mean([r["success"] for r in rows])); spl=float(np.mean([r["spl"] for r in rows]))
    summary={"method":"SSTG-Nav (pre-mapped semantic topology, oracle labels)","dataset":"HM3D ObjectNav v2 val_mini","episodes":len(rows),"scenes":len(manifest["scenes"]),"success_rate":sr,"spl":spl,"success_distance":cfg["success_distance"],"semantic_backend":cfg["semantic_backend"]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)); (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    with (out/"summary.md").open("w") as f: f.write("| Method | Split | Episodes | SR | SPL |\n|---|---|---:|---:|---:|\n| %s | %s | %d | %.3f | %.3f |\n"%(summary["method"],summary["dataset"],len(rows),sr,spl))
    print(json.dumps(summary,indent=2)); return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/minival.yaml"); args=ap.parse_args(); run(args.config)
if __name__=="__main__": main()
