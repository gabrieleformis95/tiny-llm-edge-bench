"""P1-1 quality-only re-run: recompute instruction_follow with chat templates.

Reloads each model that has an existing instruction_follow result, re-runs ONLY
the quality benchmark (30 prompts) with the model's chat_template applied, and
rewrites the `quality` field in place. Throughput/energy are preserved untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.benchmarks.quality import run_quality_benchmark
from src.inference.llama_cpp_backend import LlamaCppBackend
from src.registry.downloader import download_gguf
from src.registry.models import get_model
from src.registry.tasks import get_task
from src.tasks.instruction_follow import InstructionFollowTask

RESULTS = Path("results")
spec = get_task("instruction_follow")

cells = []
for f in RESULTS.glob("*.json"):
    try:
        d = json.loads(f.read_text())
    except Exception:
        continue
    if not isinstance(d, dict):
        continue
    t = d.get("task")
    if not isinstance(t, dict) or t.get("id") != "instruction_follow":
        continue
    cells.append((f, d))

cells.sort(key=lambda x: (x[1]["model"]["id"], x[1]["quant"]["name"]))
print(f"Re-patching {len(cells)} instruction_follow cells\n")

for f, d in cells:
    model_id = d["model"]["id"]
    quant = d["quant"]["name"]
    old = d["quality"]["primary_metric_value"]
    print(f"[{model_id} {quant}] old={old:.3f} ...", flush=True)

    model = get_model(model_id)
    gguf = download_gguf(model_id, quant)
    b = LlamaCppBackend()
    with b:
        b.load(gguf, n_ctx=2048, chat_template=model.chat_template)
        task = InstructionFollowTask(spec.dataset_path)
        res = run_quality_benchmark(b, task, spec)

    d["quality"] = res.model_dump(mode="json")
    f.write_text(json.dumps(d, indent=2))
    print(f"[{model_id} {quant}] new={res.primary_metric_value:.3f}  (was {old:.3f})\n", flush=True)

print("Done.")
