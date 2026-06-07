/* USART2 MMIO output for Renode capture (headless, version-stable).
 *
 * Renode's ARM semihosting enable API changed across versions; writing chars to
 * the STM32 USART2 data register and capturing via `showAnalyzer sysbus.usart2`
 * with `--disable-xwt` is the robust headless path. Function names kept as sh_*
 * so the firmware code is output-mechanism agnostic.
 *
 * STM32F4 USART2: base 0x40004400, SR @ +0x00 (TXE = bit 7), DR @ +0x04.
 */
#pragma once

#include <stdint.h>
#include <stddef.h>

#define USART2_SR  (*(volatile uint32_t *)0x40004400)
#define USART2_DR  (*(volatile uint32_t *)0x40004404)
#define USART2_CR1 (*(volatile uint32_t *)0x4000440C)
#define USART_UE   (1u << 13)
#define USART_TE   (1u << 3)

/* Enable USART2 transmitter; Renode drops DR writes unless TE is set. */
static inline void sh_uart_init(void) {
    USART2_CR1 |= USART_UE | USART_TE;
}

static inline void sh_writec(char ch) {
    /* Renode's STM32_UART transmits on DR write; no TXE poll needed. */
    USART2_DR = (uint32_t)(uint8_t)ch;
}

static inline void sh_write0(const char *s) {
    while (*s) sh_writec(*s++);
}

static inline void sh_exit(int code) {
    (void)code;
    while (1) { }   /* halt; the Renode .resc quits after RunFor */
}

/* Minimal unsigned-int to decimal (no stdlib). */
static inline void sh_print_uint(uint32_t v) {
    char buf[12];
    int i = 10;
    buf[11] = '\0';
    if (v == 0) { sh_writec('0'); return; }
    while (v > 0 && i >= 0) {
        buf[i--] = '0' + (v % 10);
        v /= 10;
    }
    sh_write0(&buf[i + 1]);
}

/* Print float as "INT.FFFFFF" (6 decimals), no stdlib. */
static inline void sh_print_float(float f) {
    if (f < 0) { sh_writec('-'); f = -f; }
    uint32_t int_part = (uint32_t)f;
    uint32_t frac_part = (uint32_t)((f - (float)int_part) * 1000000.0f + 0.5f);
    sh_print_uint(int_part);
    sh_writec('.');
    char frac_buf[7];
    frac_buf[6] = '\0';
    for (int i = 5; i >= 0; i--) {
        frac_buf[i] = '0' + (frac_part % 10);
        frac_part /= 10;
    }
    sh_write0(frac_buf);
}
