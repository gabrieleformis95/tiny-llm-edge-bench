"""CMSIS-NN analytical energy estimation for the P1 LSTM autoencoder.

Methodology:
  energy = total_MACs x pJ_per_MAC
  pJ_per_MAC = 20 pJ  (Lai et al. 2018, Cortex-M4 @ 168 MHz figure)

Reference:
  Lai et al. 2018 - "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs"
  Per-MAC figures:
    Cortex-M4  @ 168 MHz : ~20 pJ/MAC
    Cortex-A57 @ 1.9 GHz :  ~4.6 pJ/MAC (used in Mac estimation)

IMPORTANT: This is an ANALYTICAL ESTIMATE, not a measurement.
  - Does not include DRAM access energy (~65 pJ per 8-byte read)
  - Does not include OS overhead, cache misses, or peripheral activity
  - SIMD pipelining on M4F (2-4 INT8 MACs/cycle) can halve effective energy
  - Treat as a lower-bound on true energy consumption

All outputs are labeled ESTIMATED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Lai et al. 2018 - Cortex-M4 @ 168MHz
_PJ_PER_MAC_CORTEX_M4 = 20.0
# CMSIS-NN SIMD SIMD32 factor for INT8 (4 MACs per instruction on M4F)
# Conservative: assume 2x effective throughput (not all ops SIMD-pipelined)
_SIMD_THROUGHPUT_FACTOR = 2.0
# Derived: pJ/cycle = pJ/MAC * SIMD_factor (each cycle executes SIMD_factor MACs)
_PJ_PER_CYCLE_CORTEX_M4 = _PJ_PER_MAC_CORTEX_M4 * _SIMD_THROUGHPUT_FACTOR  # 40 pJ/cycle
# Cortex-M4 clock speed (STM32F4 target)
_CLOCK_HZ = 168_000_000
# Correction factor for LSTM activations (sigmoid/tanh are non-trivial vs pure MACs)
# Lai et al. pitfall: sigmoid/tanh underestimate; +20% correction per ROADMAP guidance
_ACTIVATION_CORRECTION = 1.20


@dataclass
class MCUEnergyEstimate:
    """Analytical energy estimate for one inference pass on Cortex-M4."""
    total_macs: int
    pj_per_mac: float
    raw_joules: float                    # before correction
    joules_per_inference_estimated: float  # after activation correction
    cycles_estimated: int               # at _SIMD_THROUGHPUT_FACTOR
    latency_us_estimated: float         # at _CLOCK_HZ
    estimation_method: str
    notes: str

    def to_dict(self) -> dict:
        return {
            "total_macs": self.total_macs,
            "pj_per_mac": self.pj_per_mac,
            "joules_per_inference_estimated": self.joules_per_inference_estimated,
            "cycles_estimated": self.cycles_estimated,
            "latency_us_estimated": self.latency_us_estimated,
            "estimation_method": self.estimation_method,
            "notes": self.notes,
        }


def count_lstm_macs(
    n_features: int,
    hidden_dim: int,
    latent_dim: int,
    window_size: int,
    num_layers: int = 1,
) -> int:
    """Count multiply-accumulate operations for one forward pass of the P1 LSTM autoencoder.

    Architecture: encoder LSTM + to_latent + latent_to_h + latent_to_c + decoder LSTM + output_layer.

    For a single-layer LSTM per time step:
      MACs = (n_features + hidden_dim) x 4 x hidden_dim
    (4 gates: i, f, g, o; each gate = full input+hidden dot product)
    """
    # Per time step, per LSTM layer
    macs_per_step = (n_features + hidden_dim) * 4 * hidden_dim * num_layers

    # Encoder LSTM: window_size steps
    encoder_macs = window_size * macs_per_step

    # to_latent: hidden_dim -> latent_dim
    to_latent_macs = hidden_dim * latent_dim

    # latent_to_h and latent_to_c: latent_dim -> hidden_dim each
    latent_proj_macs = 2 * latent_dim * hidden_dim

    # Decoder LSTM: window_size steps (n_features input = zeros, but weight shape unchanged)
    decoder_macs = window_size * macs_per_step

    # output_layer: hidden_dim -> n_features per step
    output_macs = window_size * hidden_dim * n_features

    return (
        encoder_macs + to_latent_macs + latent_proj_macs + decoder_macs + output_macs
    )


def energy_from_cycles(
    cycles: int,
    pj_per_cycle: float = _PJ_PER_CYCLE_CORTEX_M4,
    clock_hz: int = _CLOCK_HZ,
) -> float:
    """Estimate inference energy from a Renode cycle-accurate cycle count.

    Uses cycles x pJ/cycle (more accurate than MACs x pJ/MAC when cycle count
    is available from simulation, as it captures control overhead, branches, etc.)
    Applies the same LSTM activation correction as the MAC-based estimate.
    """
    raw = cycles * pj_per_cycle * 1e-12
    return raw * _ACTIVATION_CORRECTION


def estimate_mcu_energy(
    n_features: int = 17,
    hidden_dim: int = 128,
    latent_dim: int = 32,
    window_size: int = 30,
    num_layers: int = 1,
    pj_per_mac: float = _PJ_PER_MAC_CORTEX_M4,
    clock_hz: int = _CLOCK_HZ,
    simd_factor: float = _SIMD_THROUGHPUT_FACTOR,
) -> MCUEnergyEstimate:
    """Estimate inference energy for the P1 LSTM autoencoder on a Cortex-M4.

    Returns MCUEnergyEstimate with joules, cycles, and latency.
    All values are ANALYTICAL ESTIMATES.
    """
    total_macs = count_lstm_macs(
        n_features=n_features,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        window_size=window_size,
        num_layers=num_layers,
    )

    raw_joules = total_macs * pj_per_mac * 1e-12

    # Apply LSTM activation correction (+20% for sigmoid/tanh overhead)
    corrected_joules = raw_joules * _ACTIVATION_CORRECTION

    # Cycles: each MAC takes 1/simd_factor cycles (SIMD parallelism)
    cycles = int(total_macs / simd_factor)
    latency_us = (cycles / clock_hz) * 1e6

    return MCUEnergyEstimate(
        total_macs=total_macs,
        pj_per_mac=pj_per_mac,
        raw_joules=raw_joules,
        joules_per_inference_estimated=corrected_joules,
        cycles_estimated=cycles,
        latency_us_estimated=latency_us,
        estimation_method="cmsis_nn_lai2018_cortex_m4",
        notes=(
            f"ESTIMATED (analytical). "
            f"MACs = {total_macs:,} (LSTM encoder+decoder + projections). "
            f"pJ/MAC = {pj_per_mac} (Lai et al. 2018, Cortex-M4 @ 168 MHz). "
            f"+{int((_ACTIVATION_CORRECTION-1)*100)}% correction for LSTM sigmoid/tanh overhead. "
            f"SIMD factor = {simd_factor}x (SIMD32 VMLA, conservative). "
            f"Clock = {clock_hz//1_000_000} MHz. "
            "DRAM access energy (~65 pJ/8-byte read) excluded: true energy is higher."
        ),
    )
