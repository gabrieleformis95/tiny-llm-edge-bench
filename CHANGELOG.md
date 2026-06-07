# Changelog

## [1.0.0] - 2026-05-21

### Added
- Phase 0: Project scaffold, pyproject.toml, CI, stubs
- Phase 1: Cross-project integration - predictive-maintenance-copilot pipeline runs with LLM_PROVIDER=tinyllm_local (Phi-3.5-mini GGUF in place of Mistral-7B); rigorous faithfulness comparison deferred (see README Limitations)
- Phase 2: Model registry (4 models), quantization registry (GGUF + AWQ + GPTQ), hardware auto-detection, HardwareFingerprint per run
- Phase 3: Throughput + memory benchmark (N=50 measured, 5 warmup, median+IQR), BenchmarkRun JSON output
- Phase 4: Quality eval - RAG-industrial (ROUGE-L), MMLU-200 (exact match), JSON-following (schema validity), IFEval-style (constraint compliance)
- Phase 5: GGUF vs AWQ design comparison (spec-level only; AWQ-INT4 not benchmarked - requires CUDA, conversion path in scripts/quantize_awq.py)
- Phase 6: Energy measurement - macOS powermetrics (measured, baseline-subtracted) + CMSIS-NN analytical (Lai et al. 2018); both labeled explicitly
- Phase 7: Operator-level profiling - memory-bandwidth-based breakdown; FFN/attention/KV-cache fractions; FP16 vs Q4_K_M stacked-bar plot
- Phase 8: MCU deployment - P1 LSTM autoencoder ported to a real FP32 C firmware on STM32F4, run on Renode functional sim; MEASURED 23.4M executed instructions (~139 ms @ 168 MHz, instruction-count proxy), recon error 0.27381 matches PyTorch; INT8 export MSE 0.005 (real FD001 calibration)
- Phase 9: Speculative decoding - Qwen2.5-0.5B draft + Qwen2.5-3B target, acceptance rate 53.1% (Leviathan et al. 2023)
- Phase 10: Human calibration of the ROUGE-L faithfulness metric - Spearman r=0.702, N=8, p=0.052 (Phi-3.5-mini Q4_K_M, industrial RAG domain)
- Phase 11: Full matrix run infrastructure (run_full_matrix.sh idempotent), aggregation to aggregated.parquet, Pareto front computation
- Phase 12: Report generation - 6 plots (Pareto quality/latency/RAM/energy, throughput, quality degradation, MCU comparison), HTML report, README auto-update
- Phase 13: Final README pass, References section, CHANGELOG, v1.0.0 tag

### Infrastructure
- llama-cpp-python with Metal acceleration (Apple Silicon)
- Pydantic + YAML config registries (models, quants, hardware, tasks)
- ruff + mypy + pytest CI (GitHub Actions, ubuntu-latest)
- Streamlit explorer (`make ui`)
- Makefile with install, test, bench-all, mcu-bench, report, lock, clean targets
