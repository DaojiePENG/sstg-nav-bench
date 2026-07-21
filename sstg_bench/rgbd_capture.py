"""Capture full-resolution cardinal RGB-D views on an existing topology."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path

import habitat_sim
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

from .benchmark import load_scene_episodes,scene_paths,set_pose
from .vlm_map import yaw_quaternion


def make_rgbd_sim(scene,navmesh,width=640,height=360,hfov=90):
    sim_cfg=habitat_sim.SimulatorConfiguration();sim_cfg.scene_id=str(scene);sim_cfg.enable_physics=False
    agent_cfg=habitat_sim.agent.AgentConfiguration();sensors=[]
    for uuid,sensor_type in (("rgb",habitat_sim.SensorType.COLOR),("depth",habitat_sim.SensorType.DEPTH)):
        sensor=habitat_sim.CameraSensorSpec();sensor.uuid=uuid;sensor.sensor_type=sensor_type
        sensor.resolution=[height,width];sensor.position=[0.,1.25,0.];sensor.hfov=hfov;sensors.append(sensor)
    agent_cfg.sensor_specifications=sensors
    sim=habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg,[agent_cfg]));sim.pathfinder.load_nav_mesh(str(navmesh));return sim


def run(args):
    root=Path(args.root).resolve();source=root/args.source;dataset=root/"data/datasets/objectnav_hm3d_v2"/args.split
    scenes=root/"data/hm3d"/args.scene_dir;output=root/args.output;output.mkdir(parents=True,exist_ok=True)
    report={"split":args.split,"source":args.source,"sensor":{"width":args.width,"height":args.height,"hfov":args.hfov,"height_m":1.25},"scenes":{}}
    for _,data in load_scene_episodes(dataset):
        sid,scene,base,nav=scene_paths(scenes,data["episodes"][0]["scene_id"]);source_map=source/scene/"vlm_topological_map.json"
        if not source_map.exists():continue
        mapping=json.loads(source_map.read_text());sim=make_rgbd_sim(base,nav,args.width,args.height,args.hfov);scene_out=output/scene;scene_out.mkdir(exist_ok=True)
        nodes=[]
        for node in mapping["nodes"]:
            item={k:v for k,v in node.items() if k not in ("vlm","categories","categories_all","categories_primary","category_scores")};views=[]
            canvas=Image.new("RGB",(args.width*2,args.height*2));draw=ImageDraw.Draw(canvas)
            for view_index,yaw in enumerate((0,90,180,270)):
                rotation=yaw_quaternion(item.get("rotation",[0,0,0,1]),yaw);set_pose(sim,item["position"],rotation)
                obs=sim.get_sensor_observations();rgb=obs["rgb"][:,:,:3];depth=np.asarray(obs["depth"],dtype=np.float32)
                rgb_path=scene_out/f"node_{item['id']:04d}_view{view_index}_rgb.jpg"
                depth_path=scene_out/f"node_{item['id']:04d}_view{view_index}_depth.npy"
                if not rgb_path.exists():imageio.imwrite(rgb_path,rgb,quality=92)
                if not depth_path.exists():np.save(depth_path,depth)
                canvas.paste(Image.fromarray(rgb),(view_index%2*args.width,view_index//2*args.height));draw.rectangle((view_index%2*args.width,view_index//2*args.height,view_index%2*args.width+145,view_index//2*args.height+24),fill="black")
                draw.text((view_index%2*args.width+5,view_index//2*args.height+5),f"view {view_index} yaw {yaw}",fill="white")
                views.append({"view_index":view_index,"yaw_offset_deg":yaw,"rotation":rotation,
                              "resolution":[args.width,args.height],"hfov_deg":args.hfov,
                              "rgb_path":str(rgb_path.relative_to(root)),"depth_path":str(depth_path.relative_to(root))})
            pano=scene_out/f"node_{item['id']:04d}_rgb_panorama.jpg";canvas.save(pano,quality=92)
            item["rgbd_views"]=views;item["observation_path"]=str(pano.relative_to(root));nodes.append(item)
        result={"scene":sid,"nodes":nodes,"edges":mapping.get("edges",[]),"capture":{"goal_annotations_used":False,"views_per_node":4,"full_resolution_per_view":[args.width,args.height],"hfov_deg":args.hfov}}
        (scene_out/"rgbd_topological_map.json").write_text(json.dumps(result,indent=2));report["scenes"][scene]={"nodes":len(nodes),"views":len(nodes)*4};sim.close()
    report["nodes"]=sum(x["nodes"] for x in report["scenes"].values());report["views"]=sum(x["views"] for x in report["scenes"].values())
    (output/"capture_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--split",default="val_mini")
    p.add_argument("--scene-dir",default="minival");p.add_argument("--source",default="outputs/hm3d_minival_uniform/source")
    p.add_argument("--output",default="outputs/hm3d_minival_uniform/rgbd_capture")
    p.add_argument("--width",type=int,default=640);p.add_argument("--height",type=int,default=360);p.add_argument("--hfov",type=float,default=90.)
    run(p.parse_args())


if __name__=="__main__":main()
