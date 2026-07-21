import json

import numpy as np

from sstg_bench.arrival_vlm import depth_median, parse_output
from sstg_bench.arrival_evaluate import verifier_accept


def test_arrival_parser_accept_shape_and_1000_coordinates():
    raw = json.dumps({
        "target_category": "sofa",
        "target_visible": True,
        "stop_geometry": "valid",
        "view_index": 0,
        "bbox_norm": [100, 200, 900, 800],
        "confidence": 0.94,
        "reason_code": "confirmed",
    })
    result = parse_output(raw, "sofa")
    assert result["target_visible"] is True
    assert result["view_index"] == 0
    assert result["bbox_norm"] == [0.1, 0.2, 0.9, 0.8]


def test_arrival_parser_rejects_wrong_returned_category():
    raw = json.dumps({
        "target_category": "chair",
        "target_visible": True,
        "stop_geometry": "valid",
        "view_index": 1,
        "bbox_norm": [0.1, 0.2, 0.8, 0.9],
        "confidence": 0.98,
        "reason_code": "confirmed",
    })
    assert parse_output(raw, "bed")["target_visible"] is False


def test_depth_median_uses_box_center(tmp_path):
    depth = np.full((20, 20), 8.0, dtype=np.float32)
    depth[7:13, 7:13] = 0.9
    path = tmp_path / "depth.npy"
    np.save(path, depth)
    value = depth_median(path, [0.2, 0.2, 0.8, 0.8])
    assert value == np.float32(0.9)


def test_arrival_evaluator_can_reparse_cached_signals():
    response = {
        "status": "ok",
        "decision": {
            "accept": False,
            "target_visible": True,
            "depth_m": 0.9,
        },
    }
    assert verifier_accept(response, "cached", 0.25, 2.5) is False
    assert verifier_accept(response, "visible", 0.25, 2.5) is True
    assert verifier_accept(response, "visible_depth", 0.25, 2.5) is True
    response["decision"]["depth_m"] = 3.0
    assert verifier_accept(response, "visible_depth", 0.25, 2.5) is False
