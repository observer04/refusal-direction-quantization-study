"""Fetch only small corrected-run evidence from the already-persisted private repo."""
import concurrent.futures
import json
import urllib.parse
import urllib.request
from pathlib import Path

from persist_recovery_evidence import DEFAULT_REPO, HERE, checkpoint_metadata, token_from_workspace


def main():
    from huggingface_hub import HfApi
    token = token_from_workspace()
    api = HfApi(token=token)
    info = api.model_info(DEFAULT_REPO, files_metadata=True)
    assert info.private
    selected = []
    for item in info.siblings:
        name = item.rfilename
        if not name.startswith("experiments/"):
            continue
        relative = name[len("experiments/"):]
        if (("/" not in relative and (relative.endswith((".json", ".jsonl"))
                                     or relative in {"refusal_directions.pt", "environment.txt"}))
                or relative in {"adapter/adapter_config.json", "adapter/adapter_model.safetensors"}):
            assert item.size < 20_000_000, (name, item.size)
            selected.append((name, relative, item.size))
    destination = HERE / "runs/corrected-v1/qwen35_corrected/results"
    destination.mkdir(parents=True, exist_ok=True)

    def fetch(entry):
        name, relative, size = entry
        url = f"https://huggingface.co/{DEFAULT_REPO}/resolve/{info.sha}/{urllib.parse.quote(name)}"
        request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
        with urllib.request.urlopen(request, timeout=60) as response:
            blob = response.read(20_000_001)
        assert len(blob) == size < 20_000_000, (name, len(blob), size)
        assert token.encode() not in blob, "Credential appeared in evidence"
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(blob)
        return relative

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        downloaded = sorted(executor.map(fetch, selected))
    metadata = checkpoint_metadata(api, DEFAULT_REPO, token)
    metadata["downloaded_revision"] = info.sha
    metadata["small_evidence_files"] = downloaded
    (HERE / "corrected_download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"revision": info.sha, "small_files_downloaded": len(downloaded),
                      "results": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
