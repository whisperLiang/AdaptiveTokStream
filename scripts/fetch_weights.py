"""Download pinned official weights atomically; never invent or reshape parameters."""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

revision = "c514b22615a140c06f4c9792e93cbef9e5c79f32"
out = Path(__file__).resolve().parents[1] / "artifacts/weights/AdapTok"
out.mkdir(parents=True, exist_ok=True)
for filename in ["config.json", "model.safetensors"]:
    source = hf_hub_download("yeahhhh326/AdapTok", filename, revision=revision)
    temporary = out / (filename + ".partial")
    shutil.copyfile(source, temporary)
    temporary.replace(out / filename)
print(out)
