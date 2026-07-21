from sstg_bench.topk import candidate_rank_key, hierarchical_candidates


def test_fused_support_ranking_breaks_saturated_scores():
    near_low_support = {
        "category_scores": {"chair": 1.0},
        "confidence": 0.99,
        "cluster_support": 2,
    }
    far_high_support = {
        "category_scores": {"chair": 1.0},
        "confidence": 0.97,
        "cluster_support": 12,
    }
    assert candidate_rank_key(far_high_support, "chair", 8.0, "support_confidence") < candidate_rank_key(
        near_low_support, "chair", 2.0, "support_confidence"
    )
    assert candidate_rank_key(near_low_support, "chair", 2.0, "confidence_support") < candidate_rank_key(
        far_high_support, "chair", 8.0, "confidence_support"
    )


def test_hierarchical_candidates_keep_primary_then_distinct_fallbacks():
    primary = [((0,), 1.0, {"id": 10, "position": [0.0, 0.0, 0.0]})]
    fallback = [
        ((0,), 1.0, {"id": 20, "position": [0.0, 0.0, 0.0]}),
        ((1,), 2.0, {"id": 21, "position": [3.2, 0.0, 0.0]}),
        ((2,), 3.0, {"id": 22, "position": [6.5, 0.0, 0.0]}),
    ]
    selected = hierarchical_candidates(primary, fallback, 3, 3.0, 1)
    assert [item["id"] for item in selected] == [10, 21, 22]
