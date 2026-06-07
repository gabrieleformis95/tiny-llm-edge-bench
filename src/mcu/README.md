# MCU Benchmark

Runs the P1 LSTM autoencoder (165K params) on a virtual STM32F4 (Cortex-M4 @ 168 MHz).
The firmware executes a **real FP32 forward pass** (hand-rolled C from the checkpoint
weights, `model_weights.h`) on a real FD001 window; Renode reports the measured
executed-instruction count via `cpu ExecutedInstructions`.

Renode is a **functional** simulator: the instruction count is a ~1 instr/cycle proxy,
not silicon-cycle-accurate.

## Running

```bash
make mcu-bench
```

This runs `src/mcu/export.py` (extracts weights -> `model_weights.h` + TFLite INT8),
compiles the firmware (`arm-none-eabi-gcc`), and runs `src/mcu/renode_runner.py`
(Renode simulation, or analytical fallback if Renode is unavailable).

## Renode simulation

The runner tries, in order:
1. **Native `renode` binary** in PATH
2. **Docker** (`antmicro/renode:1.16.1`) if Docker is running and the image is present
3. **Analytical fallback** (labeled `ESTIMATED`) if neither is available

### Installing Renode (native)

The brew cask was removed. Install from the official releases:
```
https://github.com/renode/renode/releases
```
Download the macOS `.pkg` (universal binary, works on Apple Silicon).
After install, `renode` should be in `/Applications/Renode.app/Contents/MacOS/`.
Add it to PATH or symlink: `sudo ln -s /Applications/Renode.app/Contents/MacOS/renode /usr/local/bin/renode`.

### Docker

```bash
docker pull antmicro/renode:1.16.1
```

**Note**: On Apple Silicon the image runs as `linux/amd64` via emulation. Despite the
emulation overhead, the full run (load + ~23M-instruction inference + readout) completes
in a few seconds of wall time - practical for CI.

## Current status

- Hardware: Apple Silicon
- Renode native: not installed; Docker (`antmicro/renode:1.16.1`) used
- Result: `renode_functional_instruction_count` - MEASURED 23,419,939 executed
  instructions (~139 ms @ 168 MHz, proxy), recon error 0.27381 (matches PyTorch)

When Renode is available the runner uses `energy_from_cycles()` (count x pJ/cycle) and
sets `estimation_method = renode_functional+cmsis_nn_lai2018`. Renode is functional, so
this is an instruction-count proxy, not a silicon-cycle-accurate measurement.
