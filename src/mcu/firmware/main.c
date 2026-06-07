/* LSTM Autoencoder REAL inference on STM32F4 (Cortex-M4F @ 168 MHz).
 *
 * Ports the P1 LSTM autoencoder (165,297 params) from predictive-maintenance-copilot
 * and runs an actual FP32 forward pass on one real FD001 input window. The executed
 * instruction count is measured via the DWT cycle counter and reported through ARM
 * semihosting, captured by Renode.
 *
 * Fidelity note: the deployed TFLite model uses dynamic-range quantization (INT8
 * weights, float32 activations), so this FP32 activation path is representative of
 * the deployed compute. Weights are embedded FP32 (see model_weights.h).
 *
 * Timing note: Renode is a FUNCTIONAL simulator. DWT_CYCCNT here counts executed
 * instructions, not silicon-accurate pipeline cycles. Reported as instruction-count
 * timing, NOT cycle-accurate. Not measured on physical hardware.
 */

#include <stdint.h>
#include "semihosting.h"
#include "model_weights.h"

/* Renode does not model the DWT cycle counter. The executed-instruction count is
 * read from the host side via `cpu ExecutedInstructions`; the firmware enters WFI
 * after inference so the counter freezes (no spin-loop inflation). */

/* --- Math (freestanding: no libm) --- */
static float my_expf(float x) {
    if (x > 88.0f)  x = 88.0f;
    if (x < -88.0f) x = -88.0f;
    const float LN2 = 0.6931471805599453f;
    const float INV_LN2 = 1.4426950408889634f;
    int k = (int)(x * INV_LN2 + (x >= 0 ? 0.5f : -0.5f));
    float r = x - (float)k * LN2;
    /* exp(r) on [-ln2/2, ln2/2] via degree-5 Taylor */
    float p = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666667f
              + r * (0.041666668f + r * 0.008333334f))));
    union { float f; uint32_t u; } v;
    int e = k + 127;
    if (e < 1)   e = 1;
    if (e > 254) e = 254;
    v.u = ((uint32_t)e) << 23;   /* 2^k */
    return p * v.f;
}
static float sigmoidf(float x) { return 1.0f / (1.0f + my_expf(-x)); }
static float tanhf_(float x)   { return 2.0f * sigmoidf(2.0f * x) - 1.0f; }

/* --- Working buffers (SRAM) --- */
static float h[HID], c[HID];
static float g[4 * HID];
static float lat[LAT];
static float dh[HID], dc[HID], h2[HID], c2[HID];

/* One LSTM step given precomputed gate preactivations g[4H] -> updates h_, c_ */
static void lstm_gates(const float *gv, float *h_, float *c_) {
    for (int j = 0; j < HID; j++) {
        float gi = gv[j];
        float gf = gv[HID + j];
        float gg = gv[2 * HID + j];
        float go = gv[3 * HID + j];
        float cj = sigmoidf(gf) * c_[j] + sigmoidf(gi) * tanhf_(gg);
        c_[j] = cj;
        h_[j] = sigmoidf(go) * tanhf_(cj);
    }
}

/* Run full autoencoder forward pass, return mean squared reconstruction error. */
static float run_inference(void) {
    /* Encoder LSTM over the input window */
    for (int j = 0; j < HID; j++) { h[j] = 0.0f; c[j] = 0.0f; }
    for (int t = 0; t < WIN; t++) {
        const float *xt = &input_window[t * N_FEAT];
        for (int j = 0; j < 4 * HID; j++) {
            float acc = enc_b[j];
            const float *wih = &enc_wih[j * N_FEAT];
            for (int f = 0; f < N_FEAT; f++) acc += wih[f] * xt[f];
            const float *whh = &enc_whh[j * HID];
            for (int k = 0; k < HID; k++) acc += whh[k] * h[k];
            g[j] = acc;
        }
        lstm_gates(g, h, c);
    }

    /* Projections: to_latent, latent_to_h, latent_to_c */
    for (int l = 0; l < LAT; l++) {
        float acc = b_lat[l];
        const float *wl = &w_lat[l * HID];
        for (int k = 0; k < HID; k++) acc += wl[k] * h[k];
        lat[l] = acc;
    }
    for (int j = 0; j < HID; j++) {
        float ah = b_h[j], ac = b_c[j];
        const float *wh = &w_h[j * LAT];
        const float *wc = &w_c[j * LAT];
        for (int l = 0; l < LAT; l++) { ah += wh[l] * lat[l]; ac += wc[l] * lat[l]; }
        dh[j] = ah; dc[j] = ac;
    }

    /* Decoder LSTM (zero input) + per-step output, accumulate MSE vs input */
    for (int j = 0; j < HID; j++) { h2[j] = dh[j]; c2[j] = dc[j]; }
    float sse = 0.0f;
    for (int t = 0; t < WIN; t++) {
        for (int j = 0; j < 4 * HID; j++) {
            float acc = dec_b[j];          /* input is zero -> skip dec_wih term */
            const float *whh = &dec_whh[j * HID];
            for (int k = 0; k < HID; k++) acc += whh[k] * h2[k];
            g[j] = acc;
        }
        lstm_gates(g, h2, c2);
        const float *xt = &input_window[t * N_FEAT];
        for (int f = 0; f < N_FEAT; f++) {
            float o = b_out[f];
            const float *wo = &w_out[f * HID];
            for (int k = 0; k < HID; k++) o += wo[k] * h2[k];
            float d = o - xt[f];
            sse += d * d;
        }
    }
    return sse / (float)(WIN * N_FEAT);
}

int main(void) {
    sh_uart_init();
    float recon = run_inference();

    sh_write0("=== LSTM Autoencoder MCU Benchmark (REAL inference) ===\n");
    sh_write0("reconstruction_error="); sh_print_float(recon);            sh_writec('\n');
    sh_write0("ref_recon_error=");      sh_print_float(REF_RECON_ERROR);  sh_writec('\n');
    sh_write0("anomaly_threshold=");    sh_print_float(ANOMALY_THRESHOLD);sh_writec('\n');
    sh_write0("is_anomaly=");
    sh_writec(recon > ANOMALY_THRESHOLD ? '1' : '0');                     sh_writec('\n');
    sh_write0("done\n");

    /* Idle: WFI freezes the executed-instruction counter (no spin inflation). */
    for (;;) __asm__ volatile("wfi");
    return 0;
}
