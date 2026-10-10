"""B against A on val from the caches (session 07, A1): paired AUCs over clips and the list-price cost."""

import numpy as np
import pytest

from sentinel.eval.b_val import compare, usage_stats


def test_paired_comparison_resamples_whole_clips():
    rng = np.random.default_rng(0)
    groups = []
    for label in (1, 0) * 20:
        y = np.full(5, label)
        groups.append({"y": y, "b": y * 0.3 + rng.random(5), "a": y * 0.6 + rng.random(5)})
    c = compare(groups)
    assert c["clips"] == 40 and c["windows"] == 200 and c["fight_windows"] == 100
    assert c["auc_a"] > c["auc_b"] and c["b_minus_a"] == pytest.approx(c["auc_b"] - c["auc_a"])
    assert c["b_minus_a_ci"][0] < c["b_minus_a"] < c["b_minus_a_ci"][1]


def test_cost_at_list_price_counts_input_and_output_tokens_of_real_calls():
    records = [{"usage": {"input": 1000, "output": 200, "reasoning": 150}, "latency_s": 1.0}] * 3 + [None]
    u = usage_stats(records, price_in=0.15, price_out=0.60)
    assert u["calls"] == 3 and u["tokens_total"] == {"input": 3000, "output": 600, "reasoning": 450}
    assert u["cost_usd_list_price"] == pytest.approx(round(3000 * 0.15e-6 + 600 * 0.60e-6, 4))
    assert u["latency_p50_s"] == 1.0
