"""Analytical energy estimation via MAC counting and Lai et al. 2018 per-MAC figures.

Method:
  MACs_per_token  ≈  2 × N_params  (standard transformer approximation;
                                     ignores embedding table and KV cache reads)
  joules_per_token = MACs_per_token × PJ_PER_MAC_CORTEX_A × 1e-12

Reference:
  Lai et al. 2018 - "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs"
  Per-MAC energy figures:
    Cortex-M4  @ 168 MHz : ~20 pJ/MAC
    Cortex-A57 @ 1.9 GHz :  ~4.6 pJ/MAC  <-- used here for Apple Silicon
  Apple Firestorm/Avalanche cores are Cortex-A class (higher IPC); using 4.6 pJ
  as a conservative upper-bound for compute-only energy. DRAM access is not
  included (typically 65 pJ per 8-byte DRAM read) - so this estimate is a
  lower bound on true energy consumption.

ALL results from this module are labeled ESTIMATED in output JSON.
"""

from __future__ import annotations

from src.config import EnergyResult

# Lai et al. 2018 - Cortex-A57 figure. Apple M-series likely lower.
_PJ_PER_MAC_CORTEX_A = 4.6


def estimate_energy_analytical(
    params_b: float,
    n_tokens: int,
    pj_per_mac: float = _PJ_PER_MAC_CORTEX_A,
) -> EnergyResult:
    """Estimate inference energy from parameter count and token count.

    params_b: model size in billions of parameters
    n_tokens: total tokens generated (output tokens only)
    pj_per_mac: per-MAC energy in picojoules (default: Lai et al. 2018, Cortex-A)
    """
    # Standard approximation: MACs per forward pass ≈ 2 × N_params
    # (covers attention + FFN projections; excludes embeddings ~1% of params)
    macs_per_token = 2.0 * params_b * 1e9
    total_macs = macs_per_token * n_tokens
    joules = total_macs * pj_per_mac * 1e-12
    tpj = n_tokens / joules if joules > 0 else None

    return EnergyResult(
        estimated_joules_per_query=joules,
        estimated_tokens_per_joule=tpj,
        estimation_method="cmsis_nn_lai2018",
        notes=(
            f"ESTIMATED (analytical). "
            f"MACs/token = 2 × {params_b}B = {2*params_b:.1f}B; "
            f"n_tokens = {n_tokens}; "
            f"pJ/MAC = {pj_per_mac} (Lai et al. 2018, Cortex-A57). "
            "Excludes DRAM access energy (~65 pJ/8-byte read) - "
            "actual energy is higher. "
            "Apple Firestorm/Avalanche efficiency may differ from Cortex-A57."
        ),
    )


def merge_energy_results(
    measured: EnergyResult | None,
    estimated: EnergyResult,
) -> EnergyResult:
    """Merge measured (powermetrics) and analytical estimates into one EnergyResult."""
    if measured is None:
        return estimated
    return EnergyResult(
        measured_joules_per_query=measured.measured_joules_per_query,
        measured_tokens_per_joule=measured.measured_tokens_per_joule,
        estimated_joules_per_query=estimated.estimated_joules_per_query,
        estimated_tokens_per_joule=estimated.estimated_tokens_per_joule,
        estimation_method=f"{measured.estimation_method}+{estimated.estimation_method}",
        notes=f"[measured] {measured.notes} | [estimated] {estimated.notes}",
    )
