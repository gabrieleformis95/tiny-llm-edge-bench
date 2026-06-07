"""Central Pydantic Settings and domain model dataclasses."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gguf_cache_dir: Path = Path.home() / ".cache" / "tiny-llm-edge-bench" / "models"
    groq_api_key: str = ""
    hf_token: str = ""
    skip_power_measurement: bool = False
    results_dir: Path = Path("results")
    reports_dir: Path = Path("reports")
    data_dir: Path = Path("data")


settings = Settings()


class ModelSpec(BaseModel):
    id: str
    hf_gguf_repo: str
    params_b: float
    context_len: int
    chat_template: str = "chatml"


class QuantSpec(BaseModel):
    name: str
    scheme: str = "gguf"  # "gguf" | "awq" | "gptq"
    bits: float
    file_suffix: Optional[str] = None
    extra: dict = Field(default_factory=dict)


class HardwareFingerprint(BaseModel):
    """Full hardware + software snapshot captured at the start of every run."""
    host_id: str
    family: str  # "apple_silicon" | "rpi" | "x86_linux" | "cortex_m4_simulated"
    cores: int
    ram_gb: float
    cpu_freq_mhz: Optional[float] = None
    cpu_governor: Optional[str] = None
    omp_threads: int
    ac_powered: Optional[bool] = None
    ambient_temp_c: Optional[float] = None
    llama_cpp_sha: Optional[str] = None
    python_version: str
    os_release: str


class HardwareProfile(BaseModel):
    id: str
    family: str  # "apple_silicon" | "rpi" | "x86_linux"
    cores: int
    ram_gb: float
    can_run_powermetrics: bool


class TaskSpec(BaseModel):
    id: str
    kind: str  # "qa" | "json" | "classification" | "perplexity"
    dataset_path: Path
    metric: str
    n_samples: int
    description: str = ""


class ThroughputResult(BaseModel):
    n_warmup: int
    n_measured: int
    median_tok_per_s: float
    iqr_tok_per_s: float
    std_tok_per_s: Optional[float] = None
    min_tok_per_s: float
    max_tok_per_s: float
    median_ttft_ms: Optional[float] = None
    median_tpot_ms: float
    raw_samples: list[float]  # tok/s per measured run, for plot regeneration


class MemoryResult(BaseModel):
    peak_rss_mb: float
    peak_unified_mb: Optional[float] = None  # Apple Silicon only


class PowerResult(BaseModel):
    avg_watts: float
    joules_per_query: float
    duration_s: float


class EnergyResult(BaseModel):
    """At least one of measured / estimated must be populated."""
    # Measured via powermetrics with baseline subtraction (macOS only, sudo required)
    measured_joules_per_query: Optional[float] = None
    measured_tokens_per_joule: Optional[float] = None
    # Analytical estimate: MAC count × per-MAC energy (Lai et al. 2018)
    estimated_joules_per_query: Optional[float] = None
    estimated_tokens_per_joule: Optional[float] = None
    estimation_method: Optional[str] = None  # "powermetrics_mac" | "cmsis_nn_lai2018"
    notes: str = ""


class QualityResult(BaseModel):
    task_id: str
    n_samples: int
    primary_metric_value: float
    per_sample_results: list[dict]


class BenchmarkRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: ModelSpec
    quant: QuantSpec
    hardware: HardwareProfile
    fingerprint: HardwareFingerprint
    task: Optional[TaskSpec] = None
    throughput: ThroughputResult
    memory: MemoryResult
    power: Optional[PowerResult] = None  # legacy; prefer energy
    energy: Optional[EnergyResult] = None
    quality: Optional[QualityResult] = None
