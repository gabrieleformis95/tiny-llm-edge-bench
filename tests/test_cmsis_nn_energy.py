"""Tests for CMSIS-NN analytical energy estimation."""

from __future__ import annotations

from src.mcu.cmsis_nn_energy import MCUEnergyEstimate, count_lstm_macs, estimate_mcu_energy


def test_mac_count_positive():
    macs = count_lstm_macs(n_features=17, hidden_dim=128, latent_dim=32, window_size=30)
    assert macs > 0


def test_mac_count_known_architecture():
    # Encoder: 30 × (17+128) × 4 × 128 = 30 × 74240 = 2,227,200
    # to_latent: 128 × 32 = 4,096
    # latent_to_h + latent_to_c: 2 × 32 × 128 = 8,192
    # Decoder: same as encoder = 2,227,200
    # output_layer: 30 × 128 × 17 = 65,280
    expected = 2_227_200 + 4_096 + 8_192 + 2_227_200 + 65_280
    macs = count_lstm_macs(n_features=17, hidden_dim=128, latent_dim=32, window_size=30)
    assert macs == expected, f"Expected {expected}, got {macs}"


def test_energy_estimate_returns_dataclass():
    est = estimate_mcu_energy()
    assert isinstance(est, MCUEnergyEstimate)
    assert est.joules_per_inference_estimated > 0
    assert est.cycles_estimated > 0
    assert est.latency_us_estimated > 0


def test_energy_estimate_labeled_estimated():
    est = estimate_mcu_energy()
    assert "ESTIMATED" in est.notes
    assert "Lai et al. 2018" in est.notes


def test_energy_to_dict():
    est = estimate_mcu_energy()
    d = est.to_dict()
    assert "joules_per_inference_estimated" in d
    assert "cycles_estimated" in d
    assert "latency_us_estimated" in d
