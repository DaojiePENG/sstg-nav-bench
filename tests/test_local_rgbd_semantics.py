from sstg_bench.local_rgbd_semantics import parse, parse_view


def test_qwen_grounding_parser_accepts_empty_list():
    detections, description = parse("```json\n[]\n```")
    assert detections == []
    assert description == ""


def test_qwen_grounding_parser_maps_panorama_box_to_view_local_center():
    raw = '[{"bbox_2d":[700,400,900,600],"label":"chair","confidence":0.8}]'
    detections, _ = parse(raw)
    assert len(detections) == 1
    detection = detections[0]
    assert detection["view_index"] == 3
    assert detection["center"] == [0.25, 140 / 360]


def test_qwen_view_repair_parser_normalizes_1000_coordinates():
    detections = parse_view('[{"bbox_2d":[100,200,500,800],"label":"plant","confidence":0.9}]', 2)
    assert detections == [{
        "category": "plant", "confidence": 0.9, "view_index": 2,
        "center": [0.3, 0.5], "bbox_norm": [0.1, 0.2, 0.5, 0.8],
    }]
