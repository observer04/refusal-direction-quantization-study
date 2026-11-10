"""Persist reviewed small artifacts and audit existing remote checkpoint metadata.

This script never downloads model weights or executes a model. The corrected
Kaggle notebook must already have uploaded and evaluated the full checkpoint.
Credentials are read from the process or the workspace .env without printing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT_REPO = "Observer04/Qwen3.5-9B-heretic-corrected-20261109"


def token_from_workspace():
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    for line in (ROOT / ".env").read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key == "HF_TOKEN":
            return value.strip().strip("\"'")
    raise RuntimeError("HF_TOKEN missing")


def checkpoint_metadata(api, repo_id, token):
    info = api.model_info(repo_id, files_metadata=True)
    assert info.private, "Recovery checkpoint must remain private"
    shards = {}
    for item in info.siblings:
        name = item.rfilename
        if "/" not in name and name.endswith(".safetensors"):
            assert item.lfs is not None, (name, "Missing LFS content metadata")
            shards[name] = {"size": item.size, "sha256": item.lfs.sha256}
    assert shards, "No full-model weight shards found"
    url = f"https://huggingface.co/{repo_id}/resolve/{info.sha}/model.safetensors.index.json"
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=60) as response:
        index_blob = response.read(2_000_001)
    assert len(index_blob) <= 2_000_000, "Unexpectedly large weight index"
    index = json.loads(index_blob)
    assert set(index["weight_map"].values()) == set(shards), "Weight index/shard mismatch"
    return {"repo_id": repo_id, "private": info.private, "revision": info.sha,
            "weight_shards": shards,
            "weight_index_sha256": hashlib.sha256(index_blob).hexdigest(),
            "weight_index_total_size": index["metadata"]["total_size"],
            "scope": "Remote index and LFS content metadata; no local weight download"}


def collect_files(corrected_results):
    files = {"README.md": HERE / "CORRECTED_MODEL_CARD.md"}
    for name in ["Reproduce_Qwen3_5_9B_Heretic_Corrected.ipynb",
                 "Reproduce_Qwen3_5_9B_Heretic.ipynb", "STATE.md"]:
        files["reproduction/" + name] = ROOT / name
    for name in ["README.md", "sources.json", "PHASE1_RESULTS.md"]:
        path = HERE.parent / name
        if path.exists():
            files["reproduction/" + name] = path
    # Deliberately omit observer/download logs, credentials, caches and full weights.
    for path in sorted(HERE.iterdir()):
        if path.is_file() and path.suffix in {".py", ".md", ".json"}:
            files["reproduction/recovery/" + path.name] = path
    for name in ["evaluation.py", "verify_corrected_results.py", "verify_recovery_results.py",
                 "semantic_review_rubric.md"]:
        files["experiments/" + name] = HERE / name
    for pattern in ["*semantic_review*.json*", "*evaluation_verification.json"]:
        for path in corrected_results.glob(pattern):
            files["experiments/" + path.name] = path
    for path in (HERE / "runs/corrected-v1/source").glob("*"):
        if path.is_file() and path.suffix in {".json", ".ipynb"}:
            files["reproduction/kaggle_submitted_source/" + path.name] = path
    pilot = HERE / "runs/pilot-v1/qwen35_recovery/results"
    assert json.loads((pilot / "evaluation_verification.json").read_text())["verified"]
    assert json.loads((corrected_results / "corrected_evaluation_verification.json").read_text())["verified"]
    for label, folder in [("pilot-v1", pilot), ("corrected-v1", corrected_results)]:
        for path in sorted(folder.rglob("*")):
            if (path.is_file() and "__pycache__" not in path.parts
                    and path.suffix in {".json", ".jsonl", ".pt", ".safetensors", ".txt"}
                    and path.name != "baseline_logprobs.pt"):
                assert path.stat().st_size < 20_000_000, (path, "Unexpected large artifact")
                files[f"recovery_evidence/{label}/{path.relative_to(folder)}"] = path
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corrected_results", type=Path)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    args = parser.parse_args()
    from huggingface_hub import HfApi

    token = token_from_workspace()
    api = HfApi(token=token)
    before = checkpoint_metadata(api, args.repo, token)
    files = collect_files(args.corrected_results)
    contents = {}
    with tempfile.TemporaryDirectory(prefix="qwen35-recovery-evidence-") as task_dir:
        staging = Path(task_dir)
        for relative, source in files.items():
            blob = source.read_bytes()
            assert token.encode() not in blob, (source, "Credential detected")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            contents[relative] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
        (staging / "recovery_evidence/artifact_manifest.json").write_text(
            json.dumps({"checkpoint_before_evidence_commit": before,
                        "files": contents}, indent=2) + "\n")
        api.upload_folder(repo_id=args.repo, folder_path=str(staging),
                          commit_message="Persist corrected reproduction source and independently reviewed evidence")
    after = checkpoint_metadata(api, args.repo, token)
    assert after["weight_shards"] == before["weight_shards"], "Checkpoint weights changed during evidence upload"
    after["small_artifact_count"] = len(contents)
    after["small_artifact_bytes"] = sum(item["bytes"] for item in contents.values())
    output = HERE / "corrected_persistence.json"
    output.write_text(json.dumps(after, indent=2) + "\n")
    print(json.dumps(after, indent=2))


if __name__ == "__main__":
    main()
