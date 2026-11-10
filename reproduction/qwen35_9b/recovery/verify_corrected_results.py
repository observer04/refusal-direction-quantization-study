"""Audit fresh corrected-run artifacts without model execution or local Torch.

The first-token KL check recomputes the mean of saved per-prompt terms. It does
not independently rerun inference or derive those terms from saved logits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import evaluation as ev
from verify_recovery_results import read_jsonl, read_own_saved_float_arrays, unit, verify_saved_adapter_B


def load(path):
    return json.loads(path.read_text())


def verify(out, *, pilot_results=None):
    summary = load(out / "summary.json")
    manifest = load(out / "run_manifest.json")
    prompts = read_jsonl(out / "prompts.jsonl")
    prompt_sets = {}
    for row in prompts:
        prompt_sets.setdefault(row["split"], []).append(row)
    expected_counts = {"good_train": 400, "bad_train": 400, "good_test": 100, "bad_test": 100}
    assert {k: len(v) for k, v in prompt_sets.items()} == expected_counts
    for rows in prompt_sets.values():
        assert [r["index"] for r in rows] == list(range(len(rows)))
        assert all(r["system"] == manifest["effective_settings"]["system_prompt"] for r in rows)

    report = {"scope": "Independent artifact calculations without model inference",
              "prompt_counts": expected_counts, "generation_summaries": {}}
    pinned_sources = load(Path(__file__).resolve().parent.parent / "sources.json")
    for key in ("base_model", "reference_model", "datasets", "compatible_source_commit"):
        assert manifest["sources"][key] == pinned_sources[key]
    report["pinned_provenance"] = {key: manifest["sources"][key] for key in
                                   ("base_model", "reference_model", "datasets", "compatible_source_commit")}
    if pilot_results is not None:
        hashes = load(pilot_results / "split_manifest.json")["prompt_hashes"]
        for condition, hash_key in [("good_train", "extract_good"), ("bad_train", "extract_bad"),
                                    ("good_test", "historical_good"), ("bad_test", "historical_bad")]:
            assert [ev.digest(r["prompt"]) for r in prompt_sets[condition]] == hashes[hash_key]
        report["pilot_prompt_provenance"] = "All 400+400 extraction and 100+100 historical prompts match the pinned pilot manifest in order"
    all_rows = {}
    for condition, summary_key in [("baseline", "baseline"), ("edited", "edited"), ("reloaded", "reloaded")]:
        rows = read_jsonl(out / (condition + "_responses.jsonl"))
        assert [r["index"] for r in rows] == list(range(100))
        assert [r["prompt"] for r in rows] == [r["prompt"] for r in prompt_sets["bad_test"]]
        assert all(r["condition"] == condition for r in rows)
        for r in rows:
            assert ev.marker_refusal(r["response"]) == r["refusal_marker_match"]
            assert 0 < r["generated_tokens"] <= manifest["max_new_tokens"] == 100
            if r["hit_token_limit"]:
                assert r["generated_tokens"] == 100
        measured = {"refusals": sum(r["refusal_marker_match"] for r in rows), "n": len(rows),
                    "hit_token_limit": sum(r["hit_token_limit"] for r in rows)}
        assert all(summary[summary_key][key] == value for key, value in measured.items())
        report["generation_summaries"][condition] = measured
        all_rows[condition] = rows
    exact_matches = sum(a["response"] == b["response"] for a, b in zip(all_rows["edited"], all_rows["reloaded"]))
    assert exact_matches == summary["reloaded"]["exact_response_matches"] == 100
    report["serialization_exact_response_matches"] = exact_matches

    terms = load(out / "benign_first_token_kl_terms.json")
    values = terms["per_prompt_kl"]
    assert len(values) == 100 and all(math.isfinite(x) and x >= -1e-5 for x in values)
    assert terms["dataset"] == "mlabonne/harmless_alpaca" and terms["split"] == "test[:100]"
    kl_mean = math.fsum(values) / len(values)
    assert abs(kl_mean - terms["mean"]) < 1e-5
    assert abs(kl_mean - summary["edited"]["benign_first_token_kl"]) < 1e-5
    report["benign_kl"] = {"n": len(values), "mean_of_saved_terms": kl_mean,
        "reported_batchmean": summary["edited"]["benign_first_token_kl"],
        "scope": "Saved per-prompt mean independently recomputed; no independent inference/logit recomputation"}

    good, bad, projected = read_own_saved_float_arrays(out / "refusal_directions.pt")
    if pilot_results is not None:
        old_good, old_bad, old_raw = read_own_saved_float_arrays(pilot_results / "raw_refusal_directions.pt")
        _, _, old_projected = read_own_saved_float_arrays(pilot_results / "projected_refusal_directions.pt")
        report["fresh_vs_cached_means"] = {"good_max_coordinate_difference": max(abs(a - b) for x, y in zip(good, old_good) for a, b in zip(x, y)),
                                          "bad_max_coordinate_difference": max(abs(a - b) for x, y in zip(bad, old_bad) for a, b in zip(x, y)),
                                          "projected_max_coordinate_difference": max(abs(a - b) for x, y in zip(projected, old_projected) for a, b in zip(x, y)),
                                          "scope": "Comparison of saved means; fresh extraction execution is established by the runner/log, not by equality alone"}
    errors = []
    for g, b, saved in zip(good, bad, projected):
        raw = unit([y - x for x, y in zip(g, b)])
        normalized_good = unit(g)
        coefficient = sum(x * y for x, y in zip(raw, normalized_good))
        recomputed = unit([x - coefficient * y for x, y in zip(raw, normalized_good)])
        errors.append(max(abs(x - y) for x, y in zip(recomputed, saved)))
    assert max(errors) < 2e-6
    assert manifest["orthogonalize_direction"] is True
    assert manifest["effective_settings"]["orthogonalize_direction"] is True
    assert manifest["historical_test_exposed"] is True
    report["direction_derivation"] = {"slots": 33, "max_coordinate_error": max(errors),
        "scope": "Per-slot difference-of-means, good-mean projection and normalization independently recomputed before interpolation"}
    # Reuse only the actual adapter parser; this points at adapter/, not pilot selected_adapter/.
    report["saved_adapter"] = verify_saved_adapter_B(out, projected, manifest["published_parameters"], adapter_subdir="adapter")

    fixture = load(out / "paired_long_prompts.json")
    expected_hashes = [ev.digest(r["prompt"]) for r in fixture]
    assert len(expected_hashes) == len(set(expected_hashes)) == 12
    assert expected_hashes == manifest["paired_long_prompt_hashes"]
    paired_rows = []
    for condition, summary_key in [("original", "original"), ("corrected_reloaded", "corrected")]:
        rows = read_jsonl(out / (condition + "_paired_long_responses.jsonl"))
        assert [ev.digest(r["prompt"]) for r in rows] == expected_hashes
        for row in rows:
            assert row["condition"] == condition and row["evaluation_split"] == "paired_long_diagnostic"
            assert row["max_new_tokens"] == 512 and 0 < row["generated_tokens"] <= 512
            assert row["prompt_sha256"] == ev.digest(row["prompt"])
            assert ev.marker_refusal(row["response"]) == row["refusal_marker_match"]
            assert row["hit_token_limit"] == (row["stop_reason"] == "length")
            if row["hit_token_limit"]:
                assert row["generated_tokens"] == 512
        measured = ev.summarize_rows(rows)
        assert measured == summary["paired_long_diagnostic"][summary_key]
        report["generation_summaries"][condition + "_paired_long"] = measured
        paired_rows.extend(rows)
    queue, mapping = ev.make_blind_review_queue(paired_rows, n_prompts=12)
    assert queue == read_jsonl(out / "paired_long_blind_review_queue.jsonl")
    assert mapping == read_jsonl(out / "paired_long_blind_review_mapping.jsonl")
    assert len(queue) == len({r["review_id"] for r in queue}) == 24
    report["paired_queue"] = {"n_prompts": 12, "n_responses": 24, "unique_ids": True,
                              "labels_unreviewed_in_original_queue": True}
    report["source_artifact_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.glob("*.json*"))
        if p.name != "corrected_evaluation_verification.json"
    }
    report["verified"] = True
    (out / "corrected_evaluation_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--pilot-results", type=Path, help="Also compare pinned prompt order and fresh versus previously saved means")
    args = parser.parse_args()
    print(json.dumps(verify(args.results, pilot_results=args.pilot_results), indent=2))
