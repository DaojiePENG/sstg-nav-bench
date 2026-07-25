"""Audit benchmark counts, metric consistency, artifacts, and bibliography."""
from __future__ import annotations
import csv,json,re,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def metric_check(csv_path,summary_path):
    rows=list(csv.DictReader((ROOT/csv_path).open()));summary=json.loads((ROOT/summary_path).read_text())
    sr=sum(float(x["success"]) for x in rows)/len(rows);spl=sum(float(x["spl"]) for x in rows)/len(rows)
    return {"rows":len(rows),"unique_episode_keys":len({x["episode"] for x in rows}),"sr":sr,"spl":spl,
            "sr_matches":abs(sr-summary["sr"])<1e-9,"spl_matches":abs(spl-summary["spl"])<1e-9}

def topk_metric_check(csv_path,summary_path):
    rows=list(csv.DictReader((ROOT/csv_path).open()));summary=json.loads((ROOT/summary_path).read_text())
    metrics={}
    for k in range(1,int(summary["max_k"])+1):
        sr=sum(float(row[f"success_at_{k}"]) for row in rows)/len(rows)
        spl=sum(float(row[f"spl_at_{k}"]) for row in rows)/len(rows)
        expected=summary["metrics"][str(k)]
        metrics[str(k)]={"sr":sr,"spl":spl,"sr_matches":abs(sr-expected["sr"])<1e-9,
                         "spl_matches":abs(spl-expected["spl"])<1e-9}
    return {"rows":len(rows),"unique_episode_keys":len({row["episode"] for row in rows}),"metrics":metrics}

def main():
    result={"minival_gpt":metric_check("outputs/analysis/episodes_vlm_all_confidence.csv","outputs/analysis/summary_vlm_all_confidence.json"),
            "full_val_oracle":metric_check("outputs/hm3d_val_oracle_analysis/episodes_oracle.csv","outputs/hm3d_val_oracle_analysis/summary_oracle.json")}
    qcsv=ROOT/"outputs/hm3d_val_qwen_analysis/episodes_vlm_all_confidence.csv"
    if qcsv.exists():result["full_val_qwen"]=metric_check(qcsv.relative_to(ROOT),"outputs/hm3d_val_qwen_analysis/summary_vlm_all_confidence.json")
    new_metrics={
        "independent_qwen":"outputs/hm3d_minival_uniform/analysis_fixed",
        "qwen_rgbd_camera_90":"outputs/hm3d_minival_uniform/qwen_rgbd_90_analysis_camera",
        "qwen_rgbd_raw_90":"outputs/hm3d_minival_uniform/qwen_rgbd_90_analysis_raw",
        "qwen_rgbd_fused_90":"outputs/hm3d_minival_uniform/qwen_rgbd_90_analysis_fused",
        "gpt54_rgbd_wide120":"outputs/hm3d_minival_uniform/peterai_rgbd_wide120_analysis_soft",
        "gpt54_rgbd_wide120_raw":"outputs/hm3d_minival_uniform/peterai_rgbd_wide120_analysis_raw",
        "mimo_rgbd_90":"outputs/hm3d_minival_uniform/mimo_rgbd_analysis_reparsed_soft",
        "mimo_rgbd_wide120":"outputs/hm3d_minival_uniform/mimo_rgbd_wide120_analysis_soft",
        "gpt54_camera_90":"outputs/hm3d_minival_uniform/peterai_camera_node_analysis",
        "gpt54_camera_120":"outputs/hm3d_minival_uniform/peterai_camera_node_wide120_analysis",
        "mimo_camera_90":"outputs/hm3d_minival_uniform/mimo_camera_node_analysis",
        "mimo_camera_120":"outputs/hm3d_minival_uniform/mimo_camera_node_wide120_analysis",
    }
    for name,directory in new_metrics.items():
        result[name]=metric_check(f"{directory}/episodes_vlm_all_confidence.csv",f"{directory}/summary_vlm_all_confidence.json")
    full_metrics={
        "full_gpt54_camera":"outputs/hm3d_val_uniform/gpt54_camera_node_analysis",
        "full_gpt54_raw":"outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw",
        "full_gpt54_fused":"outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused",
    }
    for name,directory in full_metrics.items():
        if (ROOT/directory/"summary_vlm_all_confidence.json").exists():
            result[name]=metric_check(f"{directory}/episodes_vlm_all_confidence.csv",f"{directory}/summary_vlm_all_confidence.json")
    arrival_eval="outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict"
    if (ROOT/arrival_eval/"summary_arrival_verified.json").exists():
        result["full_gpt54_arrival_verified"]=metric_check(
            f"{arrival_eval}/episodes_arrival_verified.csv", f"{arrival_eval}/summary_arrival_verified.json"
        )
        arrival_summary=json.loads((ROOT/arrival_eval/"summary_arrival_verified.json").read_text())
        result["full_gpt54_arrival_verified"].update({
            "success_at_1":arrival_summary["metrics"]["1"]["sr"],
            "success_at_2":arrival_summary["metrics"]["2"]["sr"],
            "success_at_3":arrival_summary["metrics"]["3"]["sr"],
            "precision":arrival_summary["verifier_attempt_confusion"]["precision"],
            "recall":arrival_summary["verifier_attempt_confusion"]["recall"],
        })
    arrival_capture_path=ROOT/"outputs/hm3d_val_uniform/gpt54_arrival_capture_top3/capture_report.json"
    arrival_report_path=ROOT/"outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict/verifier_report.json"
    if arrival_capture_path.exists() and arrival_report_path.exists():
        capture_report=json.loads(arrival_capture_path.read_text())
        verifier_report=json.loads(arrival_report_path.read_text())
        result["full_arrival_capture"]={
            "episodes":capture_report["episodes"],"candidates":capture_report["unique_candidates"],
            "views":capture_report["views"],"goal_annotations_used":capture_report["goal_annotations_used"],
        }
        result["full_arrival_api"]={
            "model":verifier_report["model"],"candidates":verifier_report["candidates"],
            "api_successes":verifier_report["api_successes"],"decision_policy":verifier_report["decision_policy"],
        }
    for backend,label in (("peterai","gpt54"),("mimo","mimo")):
        report=json.loads((ROOT/f"outputs/hm3d_minival_uniform/{backend}_rgbd_semantics_wide120/semantic_report.json").read_text())
        result[f"{label}_wide_api"]={"nodes":report["nodes"],"api_successes":report["api_successes"],
                                        "detections":report["detections"],"parser_version":report.get("parser_version")}
    capture=json.loads((ROOT/"outputs/hm3d_minival_uniform/rgbd_capture_wide120/capture_report.json").read_text())
    mappings=[json.loads(path.read_text()) for path in (ROOT/"outputs/hm3d_minival_uniform/rgbd_capture_wide120").glob("*/rgbd_topological_map.json")]
    result["independent_wide_capture"]={"nodes":capture["nodes"],"views":capture["views"],
        "goal_annotations_used":any(mapping["capture"]["goal_annotations_used"] for mapping in mappings)}
    qwen_report_path=ROOT/"outputs/hm3d_minival_uniform/qwen_rgbd_semantics_90/semantic_report.json"
    if qwen_report_path.exists():
        qwen_report=json.loads(qwen_report_path.read_text())
        result["qwen_rgbd_90_inference"]={key:qwen_report[key] for key in (
            "model","nodes","inference_success","detections","panorama_nodes","view_local_repair_nodes"
        )}
    qwen_pair_specs={
        "qwen_camera_to_raw":(0,2,-0.06666666666666667),
        "qwen_raw_to_fused":(8,3,0.16666666666666666),
        "qwen_camera_to_fused":(6,3,0.1),
    }
    for label,(expected_gains,expected_losses,expected_delta_sr) in qwen_pair_specs.items():
        path=ROOT/f"outputs/hm3d_minival_uniform/{label}.json"
        if path.exists():
            paired=json.loads(path.read_text())
            result[label]={key:paired[key] for key in (
                "episodes","gains","losses","delta_sr","mcnemar_exact_two_sided_p","delta_spl","delta_spl_95ci"
            )}
            result[label]["expected"]={"gains":expected_gains,"losses":expected_losses,"delta_sr":expected_delta_sr}
    full_capture_path=ROOT/"outputs/hm3d_val_uniform/rgbd_capture_wide120/capture_report.json"
    if full_capture_path.exists():
        full_capture=json.loads(full_capture_path.read_text())
        full_mappings=[json.loads(path.read_text()) for path in (ROOT/"outputs/hm3d_val_uniform/rgbd_capture_wide120").glob("*/rgbd_topological_map.json")]
        result["full_independent_capture"]={"scenes":len(full_capture["scenes"]),"nodes":full_capture["nodes"],"views":full_capture["views"],
            "goal_annotations_used":any(mapping["capture"]["goal_annotations_used"] for mapping in full_mappings)}
    full_semantic_path=ROOT/"outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120/semantic_report.json"
    if full_semantic_path.exists():
        report=json.loads(full_semantic_path.read_text())
        result["full_gpt54_api"]={"model":report["model"],"nodes":report["nodes"],"api_successes":report["api_successes"],
                                   "detections":report["detections"],"parser_version":report.get("parser_version")}
    full_oracle_path=ROOT/"outputs/hm3d_val_uniform/oracle_geometry/density_ablation.json"
    if full_oracle_path.exists():
        summary=json.loads(full_oracle_path.read_text())["summaries"]["1.0"]
        result["full_independent_oracle"]={key:summary[key] for key in ("episodes","successes","sr","spl","dtg")}
        rows=list(csv.DictReader((full_oracle_path.parent/"episodes_uniform_density_1.csv").open()))
        result["full_independent_oracle"].update({
            "rows":len(rows),"unique_episode_keys":len({row["episode"] for row in rows}),
            "sr_matches":abs(sum(float(row["success"]) for row in rows)/len(rows)-summary["sr"])<1e-9,
            "spl_matches":abs(sum(float(row["spl"]) for row in rows)/len(rows)-summary["spl"])<1e-9,
        })
    topk=json.loads((ROOT/"outputs/hm3d_minival_uniform/peterai_rgbd_wide120_topk_0m/summary_topk.json").read_text())
    result["gpt54_wide_topk"]={"success_at_1":topk["metrics"]["1"]["sr"],"success_at_2":topk["metrics"]["2"]["sr"],
                                "spl_at_2":topk["metrics"]["2"]["spl"],"protocol":topk["protocol"]}
    full_topk_path=ROOT/"outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_3m/summary_topk.json"
    if full_topk_path.exists():
        full_topk=json.loads(full_topk_path.read_text())
        result["full_gpt54_topk"]={
            "episodes":full_topk["episodes"],"min_separation_m":full_topk["candidate_min_separation_m"],
            "success_at_1":full_topk["metrics"]["1"]["sr"],
            "success_at_2":full_topk["metrics"]["2"]["sr"],
            "success_at_3":full_topk["metrics"]["3"]["sr"],
            "spl_at_3":full_topk["metrics"]["3"]["spl"],"protocol":full_topk["protocol"],
        }
    topk_specs={
        "full_gpt54_raw_topk":("gpt54_rgbd_wide120_raw_topk_2p0m",0.975),
        "full_gpt54_fused_topk":("gpt54_rgbd_wide120_fused_topk_2p0m",0.964),
        "full_gpt54_fusion_aware_topk":("gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m",0.975),
    }
    for label,(directory,expected_s3) in topk_specs.items():
        base=f"outputs/hm3d_val_uniform/{directory}"
        check=topk_metric_check(f"{base}/episodes_topk.csv",f"{base}/summary_topk.json")
        summary=json.loads((ROOT/base/"summary_topk.json").read_text())
        check.update({"success_at_1":summary["metrics"]["1"]["sr"],
                      "success_at_2":summary["metrics"]["2"]["sr"],
                      "success_at_3":summary["metrics"]["3"]["sr"],
                      "spl_at_3":summary["metrics"]["3"]["spl"],"expected_success_at_3":expected_s3})
        result[label]=check
    fusion_aware_capture_path=ROOT/"outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3/capture_report.json"
    if fusion_aware_capture_path.exists():
        report=json.loads(fusion_aware_capture_path.read_text())
        result["full_fusion_aware_arrival_capture"]={
            "episodes":report["episodes"],"candidates":report["unique_candidates"],"views":report["views"],
            "goal_annotations_used":report["goal_annotations_used"],"ranking_strategy":report["ranking_strategy"],
            "min_separation_m":report["candidate_min_separation_m"],
        }
    sampling_path=ROOT/"outputs/hm3d_val_uniform/source/sampling_report.json"
    if sampling_path.exists():
        sampling=json.loads(sampling_path.read_text())
        result["full_topology"]={
            "scenes":len(sampling["scenes"]),"nodes":sampling["total_nodes"],
            "edges":sum(scene["edges"] for scene in sampling["scenes"].values()),
            "max_empirical_pool_cover_radius_m":max(scene["empirical_pool_cover_radius_m"] for scene in sampling["scenes"].values()),
        }
    identity_path=ROOT/"outputs/hm3d_val_uniform/model_identity_audit.json"
    if identity_path.exists():
        result["full_model_identity"]=json.loads(identity_path.read_text())
    maps=list((ROOT/"outputs/hm3d_val_qwen/maps").glob("*/vlm_topological_map.json"))
    if maps:
        mappings=[json.loads(p.read_text()) for p in maps]
        result["full_val_maps"]={"scenes":len(maps),"nodes":sum(len(x["nodes"]) for x in mappings),
                                 "edges":sum(len(x.get("edges",[])) for x in mappings)}
    bib=(ROOT/"paper/baselines.bib").read_text();rows=list(csv.DictReader((ROOT/"paper/baselines.csv").open()))
    keys={r["cite_key"] for r in rows if r["cite_key"]!="none"}
    result["bibliography"]={"comparison_rows":len(rows),"entries":len(re.findall(r"^@",bib,re.M)),
                            "missing_keys":sorted(k for k in keys if not re.search(r"@[A-Za-z]+\{"+re.escape(k)+r",",bib))}
    result["visuals"]={"minival_videos":len(list((ROOT/"outputs/hm3d_minival_vlm/videos").glob("*.mp4"))),
                       "full_val_videos":len(list((ROOT/"outputs/hm3d_val_qwen/visuals/videos").glob("*.mp4"))),
                       "gpt54_wide_videos":len(list((ROOT/"outputs/hm3d_minival_uniform/peterai_rgbd_wide120_visuals").glob("**/*.mp4"))),
                       "gpt54_detection_overlays":len(list((ROOT/"outputs/hm3d_minival_uniform/peterai_rgbd_wide120_visuals/detection_overlays").glob("*.jpg"))),
                       "full_gpt54_videos":len(list((ROOT/"outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals").glob("**/*.mp4"))),
                       "full_gpt54_detection_overlays":len(list((ROOT/"outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals/detection_overlays").glob("*.jpg"))),
                       "full_gpt54_semantic_maps":len(list((ROOT/"outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals/semantic_maps").glob("*.png"))),
                       "arrival_verification_overlays":len(list((ROOT/"outputs/hm3d_val_uniform/gpt54_arrival_verified_visuals/verification_overlays").glob("*.jpg")))}
    failures=[]
    for name,check in result.items():
        if isinstance(check,dict):
            if check.get("rows")!=check.get("unique_episode_keys"):failures.append(f"{name}: duplicate episode keys")
            if check.get("sr_matches") is False or check.get("spl_matches") is False:failures.append(f"{name}: metric mismatch")
            if "metrics" in check and any(not item["sr_matches"] or not item["spl_matches"] for item in check["metrics"].values()):
                failures.append(f"{name}: Top-K metric mismatch")
    if result["bibliography"]["missing_keys"]:failures.append("missing bibliography keys")
    for label in ("gpt54","mimo"):
        check=result[f"{label}_wide_api"]
        if check["nodes"]!=339 or check["api_successes"]!=339:failures.append(f"{label}: incomplete wide RGB-D API cache")
    if result["independent_wide_capture"]!={"nodes":339,"views":1356,"goal_annotations_used":False}:
        failures.append("wide RGB-D capture provenance/count mismatch")
    if "full_independent_capture" in result and result["full_independent_capture"]!={"scenes":36,"nodes":6642,"views":26568,"goal_annotations_used":False}:
        failures.append("full wide RGB-D capture provenance/count mismatch")
    if "full_gpt54_api" in result:
        check=result["full_gpt54_api"]
        if check["model"]!="gpt-5.4" or check["nodes"]!=6642 or check["api_successes"]!=6642:
            failures.append("full GPT-5.4 RGB-D API cache is incomplete or model-mismatched")
    if "full_model_identity" in result and (result["full_model_identity"]["requested_model"]!="gpt-5.4" or result["full_model_identity"]["response_model"]!="gpt-5.4"):
        failures.append("full requested/response model identity mismatch")
    if "full_topology" in result and (result["full_topology"]["scenes"]!=36 or result["full_topology"]["nodes"]!=6642 or result["full_topology"]["edges"]!=21845 or result["full_topology"]["max_empirical_pool_cover_radius_m"]>0.8):
        failures.append("full topology count or coverage mismatch")
    if "full_gpt54_topk" in result and (result["full_gpt54_topk"]["episodes"]!=1000 or abs(result["full_gpt54_topk"]["success_at_3"]-0.975)>1e-9 or abs(result["full_gpt54_topk"]["spl_at_3"]-0.6159428497207337)>1e-9):
        failures.append("full GPT-5.4 diverse Top-3 regression")
    for label in ("full_gpt54_raw_topk","full_gpt54_fused_topk","full_gpt54_fusion_aware_topk"):
        check=result[label]
        if check["rows"]!=1000 or check["unique_episode_keys"]!=1000 or abs(check["success_at_3"]-check["expected_success_at_3"])>1e-9:
            failures.append(f"{label}: candidate recovery regression")
    if result.get("full_fusion_aware_arrival_capture")!={"episodes":1000,"candidates":720,"views":2880,
            "goal_annotations_used":False,"ranking_strategy":"confidence_support","min_separation_m":2.0}:
        failures.append("fusion-aware arrival capture provenance/count mismatch")
    if "full_gpt54_arrival_verified" in result and (result["full_gpt54_arrival_verified"]["rows"]!=1000 or abs(result["full_gpt54_arrival_verified"]["success_at_3"]-0.912)>1e-9):
        failures.append("full GPT-5.4 autonomous arrival-verifier regression")
    if "full_arrival_capture" in result and result["full_arrival_capture"]!={"episodes":1000,"candidates":656,"views":2624,"goal_annotations_used":False}:
        failures.append("full arrival capture provenance/count mismatch")
    if "full_arrival_api" in result and result["full_arrival_api"]!={"model":"gpt-5.4","candidates":656,"api_successes":656,"decision_policy":"strict_dual_geometry"}:
        failures.append("full arrival verifier API cache or policy mismatch")
    if result["visuals"]["arrival_verification_overlays"] not in (0,656):
        failures.append("incomplete fresh-arrival verification overlays")
    if abs(result["gpt54_rgbd_wide120"]["sr"]-1.0)>1e-9:failures.append("GPT-5.4 wide fused SR regression")
    if "qwen_rgbd_90_inference" in result:
        check=result["qwen_rgbd_90_inference"]
        if check["model"]!="Qwen/Qwen2.5-VL-3B-Instruct" or check["nodes"]!=339 or check["inference_success"]!=1 or check["detections"]!=1064:
            failures.append("Qwen localized RGB-D inference cache is incomplete or model-mismatched")
    if abs(result["qwen_rgbd_camera_90"]["sr"]-.4)>1e-9 or abs(result["qwen_rgbd_raw_90"]["sr"]-1/3)>1e-9 or abs(result["qwen_rgbd_fused_90"]["sr"]-.5)>1e-9:
        failures.append("Qwen same-response camera/raw/fused regression")
    for label in ("qwen_camera_to_raw","qwen_raw_to_fused","qwen_camera_to_fused"):
        if label not in result:
            failures.append(f"{label}: missing paired analysis")
            continue
        check=result[label];expected=check["expected"]
        if (check["episodes"]!=30 or check["gains"]!=expected["gains"] or check["losses"]!=expected["losses"]
                or abs(check["delta_sr"]-expected["delta_sr"])>1e-9):
            failures.append(f"{label}: paired-analysis regression")
    if abs(result["gpt54_wide_topk"]["success_at_2"]-1.0)>1e-9:failures.append("GPT-5.4 wide Success@2 regression")
    if result["visuals"]["gpt54_wide_videos"]<14:failures.append("insufficient GPT-5.4 wide navigation videos")
    full_visual_report=ROOT/"outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals/visual_report.json"
    if full_visual_report.exists():
        if result["visuals"]["full_gpt54_detection_overlays"]<1000 or result["visuals"]["full_gpt54_semantic_maps"]!=72:
            failures.append("incomplete full GPT-5.4 detection or semantic-map visuals")
    result["failures"]=failures;(ROOT/"outputs/release_audit.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return 1 if failures else 0

if __name__=="__main__":sys.exit(main())
