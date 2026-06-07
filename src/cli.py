"""CLI entry-point (typer)."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="tiny-bench", help="Tiny LLM Edge Benchmark CLI")
console = Console()


@app.command(name="list-models")
def list_models() -> None:
    """List all models in the registry."""
    from src.registry.models import load_models

    table = Table(title="Model Registry")
    table.add_column("ID")
    table.add_column("Params (B)")
    table.add_column("Context")
    table.add_column("HF Repo")
    for m in load_models():
        table.add_row(m.id, str(m.params_b), str(m.context_len), m.hf_gguf_repo)
    console.print(table)


@app.command(name="list-quants")
def list_quants() -> None:
    """List all quantization levels in the registry."""
    from src.registry.models import load_quants

    table = Table(title="Quantization Registry")
    table.add_column("Name")
    table.add_column("Bits")
    table.add_column("File Suffix")
    for q in load_quants():
        table.add_row(q.name, str(q.bits), q.file_suffix)
    console.print(table)


@app.command(name="list-tasks")
def list_tasks() -> None:
    """List all benchmark tasks in the registry."""
    from src.registry.tasks import load_tasks

    table = Table(title="Task Registry")
    table.add_column("ID")
    table.add_column("Kind")
    table.add_column("Metric")
    table.add_column("Samples")
    for t in load_tasks():
        table.add_row(t.id, t.kind, t.metric, str(t.n_samples))
    console.print(table)


@app.command()
def fingerprint_cmd(
    ambient_temp: float = typer.Option(None, "--ambient-temp", help="Ambient temperature in Celsius (optional)"),
) -> None:
    """Print hardware + software fingerprint for the current host."""
    from src.registry.fingerprint import capture_fingerprint
    import json

    fp = capture_fingerprint(ambient_temp_c=ambient_temp)
    console.print_json(json.dumps(fp.model_dump()))


@app.command()
def download(
    model: str = typer.Option(..., help="Model id from registry"),
    quant: str = typer.Option(..., help="Quantization name (e.g. Q4_K_M)"),
) -> None:
    """Download a GGUF model from HuggingFace."""
    from src.registry.downloader import download_gguf

    path = download_gguf(model, quant)
    console.print(f"[green]Downloaded:[/green] {path}")


@app.command()
def run(
    model: str = typer.Option(..., help="Model id from registry"),
    quant: str = typer.Option(..., help="Quantization name"),
    task: str = typer.Option("none", help="Task id or 'none' for perf-only"),
    backend: str = typer.Option("llama_cpp", help="Inference backend: llama_cpp | mlx"),
) -> None:
    """Run a single benchmark (model x quant x task) and save results JSON."""
    from scripts.run_bench import run_benchmark

    task_id = task if task != "none" else None
    out = run_benchmark(model, quant, task_id, backend)
    console.print(f"[green]Result saved:[/green] {out}")


@app.command()
def aggregate() -> None:
    """Aggregate all results/*.json into results/aggregated.parquet."""
    from src.analysis.aggregate import aggregate_results
    from src.config import settings

    df = aggregate_results(settings.results_dir)
    console.print(f"[green]Aggregated {len(df)} runs.[/green]")


@app.command()
def report() -> None:
    """Generate report artifacts (plots, HTML, README update)."""
    from src.analysis.aggregate import aggregate_results
    from src.analysis.report import generate_report
    from src.config import settings

    df = aggregate_results(settings.results_dir)
    generate_report(df, settings.reports_dir)
    console.print(f"[green]Report generated in {settings.reports_dir}/[/green]")


if __name__ == "__main__":
    app()
