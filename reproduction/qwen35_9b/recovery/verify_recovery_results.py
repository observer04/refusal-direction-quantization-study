"""Independently check saved recovery evidence with stdlib, without inference.

Usage: python3 verify_recovery_results.py /path/to/qwen35_recovery/results
KL inference cannot be rerun from scalar summaries; its calculation path was
reviewed before launch. This checker never claims an independent logit rerun.
"""
from __future__ import annotations

import argparse
import json
import math
import pickletools
import re
import struct
import zipfile
from pathlib import Path

import evaluation as ev


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def read_own_saved_float_arrays(path):
    """Inspect known own-output PT zip arrays without executing pickle."""
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        prefix = metadata_name.rsplit("/", 1)[0]
        byteorder = archive.read(prefix + "/byteorder").decode().strip()
        if byteorder != "little":
            raise ValueError(f"Unexpected byte order: {byteorder}")
        ops = list(pickletools.genops(archive.read(metadata_name)))
        strings = [arg for op, arg, _ in ops if op.name == "BINUNICODE"]
        if strings[:7] != ["good_means", "storage", "0", "cpu", "bad_means", "1", "directions"]:
            raise ValueError(f"Unexpected own-output storage layout: {strings}")
        arrays = []
        for key in ("0", "1", "2"):
            blob = archive.read(prefix + "/data/" + key)
            if len(blob) != 33 * 4096 * 4:
                raise ValueError("Unexpected saved float-array size")
            values = [item[0] for item in struct.iter_unpack("<f", blob)]
            if not all(math.isfinite(x) for x in values):
                raise ValueError("Non-finite saved direction/mean")
            arrays.append([values[i * 4096:(i + 1) * 4096] for i in range(33)])
    return arrays


def unit(row):
    norm = math.sqrt(sum(x * x for x in row))
    return [x / norm for x in row] if norm else [0.0 for _ in row]


def verify_direction_derivation(out):
    good, bad, raw = read_own_saved_float_arrays(out / "raw_refusal_directions.pt")
    projected_good, projected_bad, projected = read_own_saved_float_arrays(out / "projected_refusal_directions.pt")
    if good != projected_good or bad != projected_bad:
        raise ValueError("Raw/projected conditions did not reuse identical means")
    raw_errors, projected_errors = [], []
    for mean_good, mean_bad, saved_raw, saved_projected in zip(good, bad, raw, projected):
        recomputed_raw = unit([b - a for a, b in zip(mean_good, mean_bad)])
        good_direction = unit(mean_good)
        coeff = sum(a * b for a, b in zip(saved_raw, good_direction))
        recomputed_projected = unit([a - coeff * b for a, b in zip(saved_raw, good_direction)])
        raw_errors.append(max(abs(a - b) for a, b in zip(recomputed_raw, saved_raw)))
        projected_errors.append(max(abs(a - b) for a, b in zip(recomputed_projected, saved_projected)))
    if max(raw_errors + projected_errors) > 2e-6:
        raise ValueError("Saved vectors disagree with independently calculated derivation")
    return {"raw_max_coordinate_error": max(raw_errors),
            "projected_max_coordinate_error": max(projected_errors),
            "means_identical_between_conditions": True}, raw, projected


def verify_saved_adapter_B(out, vectors, published, *, adapter_subdir="selected_adapter"):
    path = out / adapter_subdir / "adapter_model.safetensors"
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size))
        data_start = 8 + header_size
        index = published["direction_index"] + 1
        lo, fraction = int(index), index % 1
        selected = unit([(1 - fraction) * a + fraction * b for a, b in zip(vectors[lo], vectors[lo + 1])])
        results = []
        for name, entry in header.items():
            if ".lora_B." not in name:
                continue
            layer = int(re.search(r"\.layers\.(\d+)\.", name).group(1))
            component = "mlp.down_proj" if ".mlp.down_proj." in name else "attn.o_proj"
            p = published["parameters"][component]
            distance = abs(layer - p["max_weight_position"])
            weight = (p["max_weight"] + distance / p["min_weight_distance"] *
                      (p["min_weight"] - p["max_weight"])) if distance <= p["min_weight_distance"] else 0.0
            if entry["dtype"] != "F32" or entry["shape"] != [4096, 1]:
                raise ValueError(f"Unexpected adapter B encoding: {name}, {entry}")
            start, end = entry["data_offsets"]
            handle.seek(data_start + start)
            values = [item[0] for item in struct.iter_unpack("<f", handle.read(end - start))]
            error = max(abs(actual + weight * v) for actual, v in zip(values, selected))
            if error > 2e-6:
                raise ValueError(f"Adapter B disagrees with selected vector/layer profile: {name}: {error}")
            results.append({"layer": layer, "component": component, "expected_weight": weight,
                            "max_coordinate_error": error})
    if len(results) != 64:
        raise ValueError(f"Expected64 adapter B tensors, found{len(results)}")
    config = json.loads((out / adapter_subdir / "adapter_config.json").read_text())
    if config["r"] != 1 or config["lora_alpha"] != 1:
        raise ValueError("Unexpected adapter scaling")
    return {"checked_B_tensors": len(results), "active_B_tensors": sum(r["expected_weight"] != 0 for r in results),
            "max_coordinate_error": max(r["max_coordinate_error"] for r in results),
            "scope": "Actual saved B vectors/masks/scaling verified; A checks were executed on Kaggle"}


def verify(out, *, generation_only=False):
    summary = json.loads((out / "summary.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    split_manifest = json.loads((out / "split_manifest.json").read_text())
    report = {"generation_summaries": {},
              "kl_verification_scope": "Calculation route reviewed in runner; no independent inference or logit recomputation",
              "historical_exposed": manifest["historical_test_exposed"],
              "fresh_scope": split_manifest["fresh_scope"]}
    if not generation_only:
        vectors_check, raw, projected = verify_direction_derivation(out)
        report["direction_derivation"] = vectors_check
    else:
        report["weight_verification_scope"] = "Not checked in this generation-only audit"

    def check_rows(stem, recorded, expected_hashes, max_tokens=100):
        rows = read_jsonl(out / (stem + "_responses.jsonl"))
        if [ev.digest(r["prompt"]) for r in rows] != expected_hashes:
            raise ValueError(f"Prompt order/provenance mismatch: {stem}")
        for row in rows:
            if ev.marker_refusal(row["response"]) != row["refusal_marker_match"]:
                raise ValueError(f"Incorrect marker label: {stem}, {row['id']}")
            if row["max_new_tokens"] != max_tokens:
                raise ValueError(f"Incorrect token budget: {stem}")
            if row["hit_token_limit"] != (row["stop_reason"] == "length"):
                raise ValueError(f"Inconsistent stopping metadata: {stem}")
        metrics = ev.summarize_rows(rows)
        for key in ("n", "marker_refusals", "marker_refusal_rate", "hit_token_limit", "empty_responses"):
            if metrics[key] != recorded[key]:
                raise ValueError(f"Summary mismatch: {stem}, {key}")
        report["generation_summaries"][stem] = metrics
        return rows

    hashes = split_manifest["prompt_hashes"]
    dev_hashes = hashes["dev_bad_native"] + hashes["dev_bad_external"]
    check_rows("original_development", summary["baseline_development"], dev_hashes)
    dev_rows = {}
    for trial in summary["trials"]:
        dev_rows[trial["candidate_id"]] = check_rows(trial["candidate_id"] + "_development", trial, dev_hashes)
    winner = ev.choose_development_candidate(summary["trials"])
    if winner != summary["frozen_selection"]:
        raise ValueError("Winner not selected by declared development rule")
    if winner != json.loads((out / "frozen_selection.json").read_text()):
        raise ValueError("Frozen selection file differs from summary")
    report["selection_verified"] = winner["candidate_id"]
    if not generation_only:
        vectors = projected if winner["extraction_mode"] == "projected" else raw
        report["saved_adapter"] = verify_saved_adapter_B(out, vectors, manifest["published_parameters"])
    if "selected_historical_exposed" in summary:
        merged = check_rows("selected_merged_development", summary["selected_merged_development"], dev_hashes)
        exact = sum(a["response"] == b["response"] for a, b in zip(merged, dev_rows[winner["candidate_id"]]))
        if exact != summary["merge_comparison"]["exact_response_matches"]:
            raise ValueError("Incorrect merged/unmerged exact-match count")
        report["merge_exact_response_matches"] = exact
        check_rows("selected_merged_historical_exposed", summary["selected_historical_exposed"], hashes["historical_bad"])
        check_rows("selected_merged_fresh_standard_subset", summary["selected_fresh_subset"], hashes["fresh_bad"])
        check_rows("selected_merged_fresh_long_subset", summary["selected_long_subset"], hashes["fresh_bad"][:12], 512)
    report["verified"] = True
    filename = "generation_verification.json" if generation_only else "evaluation_verification.json"
    (out / filename).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--generation-only", action="store_true", help="Check response rows, selection and merge counts without direction/adapter files")
    args = parser.parse_args()
    print(json.dumps(verify(args.results, generation_only=args.generation_only), indent=2))
