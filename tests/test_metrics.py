import math
from sstg_bench.benchmark import densify
from sstg_bench.experiments import bootstrap_ci, mutate_labels, wilson
import sstg_bench.experiments as experiments
from sstg_bench.uniform_map import farthest_cover
from sstg_bench.topk import diverse_candidates
from sstg_bench.camera_node_map import attach_detections_to_camera_nodes
import numpy as np

def test_densify_preserves_endpoints():
    p=densify([np.array([0.,0.,0.]),np.array([1.,0.,0.])],.25)
    assert len(p)==5 and np.allclose(p[0],[0,0,0]) and np.allclose(p[-1],[1,0,0])

def test_spl_definition():
    optimal, actual, success=5.,10.,1.
    assert math.isclose(success*optimal/max(optimal,actual),.5)

def test_wilson_and_bootstrap_are_bounded_and_deterministic():
    low,high=wilson(29,30)
    assert 0 <= low < 29/30 < high <= 1
    assert bootstrap_ci([0.,1.,1.],seed=7,n_boot=1000)==bootstrap_ci([0.,1.,1.],seed=7,n_boot=1000)

def test_label_mutation_is_seeded_and_non_destructive():
    nodes=[{"id":0,"categories":["chair"]},{"id":1,"categories":["bed"]}]
    a=mutate_labels(nodes,dropout=.2,false_positive=.5,keep_probability=.8,seed=11)
    b=mutate_labels(nodes,dropout=.2,false_positive=.5,keep_probability=.8,seed=11)
    assert a==b and nodes[0]["categories"]==["chair"]

def test_farthest_cover_respects_empirical_radius():
    pool=np.asarray([[0.,0.,0.],[.4,0.,0.],[1.,0.,0.],[1.4,0.,0.]])
    selected,radius=farthest_cover(pool,.5,10)
    assert len(selected)==2 and radius<=.5+1e-8

def test_path_cache_distinguishes_reused_episode_ids(monkeypatch):
    calls=[]
    def fake_shortest(sim,start,end):
        calls.append(tuple(start));return True,float(start[0]),[]
    monkeypatch.setattr(experiments,"shortest",fake_shortest)
    cache={"sim":object(),"paths":{}};node={"id":7,"position":[0,0,0]}
    a={"episode_id":"4","object_category":"plant","start_position":[1,0,0]}
    b={"episode_id":"4","object_category":"chair","start_position":[2,0,0]}
    assert experiments.cached_path(cache,a,node)[0]==1
    assert experiments.cached_path(cache,b,node)[0]==2
    assert len(calls)==2

def test_variant_caches_distinguish_same_node_id_at_different_positions(monkeypatch):
    calls=[]
    def fake_shortest(sim,start,end):
        calls.append((tuple(start),tuple(end)));return True,float(start[0]),[]
    monkeypatch.setattr(experiments,"shortest",fake_shortest)
    monkeypatch.setattr(experiments,"goal_viewpoints",lambda data,category:[[9,0,0]])
    episode={"episode_id":"1","object_category":"chair","start_position":[4,0,0]}
    first={"id":0,"position":[1,0,0]};second={"id":0,"position":[2,0,0]}
    path_cache={"sim":object(),"paths":{}}
    experiments.cached_path(path_cache,episode,first)
    experiments.cached_path(path_cache,episode,second)
    assert len(path_cache["paths"])==2
    goal_cache={"sim":object(),"data":{},"distances":{}}
    assert experiments.cached_goal_distance(goal_cache,first,"chair")==1
    assert experiments.cached_goal_distance(goal_cache,second,"chair")==2
    assert len(goal_cache["distances"])==2

def test_predicted_category_does_not_short_circuit_goal_distance(monkeypatch):
    monkeypatch.setattr(experiments,"goal_viewpoints",lambda data,category:[[9,0,0]])
    monkeypatch.setattr(experiments,"shortest",lambda sim,start,end:(True,3.5,[]))
    cache={"sim":object(),"data":{},"distances":{}}
    predicted={"id":0,"category":"chair","position":[1,0,0]}
    oracle={"id":1,"oracle_category":"chair","position":[2,0,0]}
    assert experiments.cached_goal_distance(cache,predicted,"chair")==3.5
    assert experiments.cached_goal_distance(cache,oracle,"chair")==0.0

def test_diverse_candidates_skip_redundant_local_hypotheses():
    ranked=[
        (-.9,1.,{"id":0,"position":[0,0,0]}),
        (-.8,2.,{"id":1,"position":[1,0,0]}),
        (-.7,3.,{"id":2,"position":[5,0,0]}),
    ]
    assert [n["id"] for n in diverse_candidates(ranked,2,2.)]==[0,2]

def test_camera_node_baseline_reuses_semantics_without_geometry():
    mapping={"nodes":[{"id":3,"position":[1,0,2],"localized_vlm":{"detections":[
        {"category":"chair","confidence":.6},
        {"category":"chair","confidence":.8},
        {"category":"plant","confidence":.7},
    ]}}],"edges":[]}
    result=attach_detections_to_camera_nodes(mapping);node=result["nodes"][0]
    assert node["position"]==[1,0,2]
    assert node["categories_all"]==["chair","plant"]
    assert node["categories_primary"]==["chair"]
    assert node["category_scores"]=={"chair":.8,"plant":.7}
    assert result["baseline"]["depth_used"] is False
