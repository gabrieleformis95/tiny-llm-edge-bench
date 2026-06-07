/* Minimal startup for STM32F4 (Cortex-M4) — Renode simulation only.
 * Not intended for physical hardware programming.
 *
 * Provides: Reset_Handler, vector table, weak Default_Handler.
 * Semihosting is used for output (bkpt 0xAB) — Renode handles it.
 */

#include <stdint.h>

extern int main(void);

/* Linker-defined symbols from stm32f4.ld */
extern uint32_t _estack;
extern uint32_t _sdata;
extern uint32_t _edata;
extern uint32_t _sbss;
extern uint32_t _ebss;
extern uint32_t _sidata;

void Default_Handler(void) { while (1) {} }

void Reset_Handler(void) {
    /* Enable the Cortex-M4F FPU (CP10/CP11 full access) before any FP op,
     * otherwise the first float instruction faults and the core locks up. */
    *((volatile uint32_t *)0xE000ED88) |= (0xFu << 20);  /* CPACR */
    __asm__ volatile("dsb");
    __asm__ volatile("isb");

    /* Copy .data section from Flash to SRAM */
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;
    while (dst < &_edata) { *dst++ = *src++; }

    /* Zero .bss section */
    dst = &_sbss;
    while (dst < &_ebss) { *dst++ = 0; }

    main();
    while (1) {}
}

/* Weak aliases for all interrupt vectors */
#define WEAK_ALIAS(x) \
    void x##_Handler(void) __attribute__((weak, alias("Default_Handler")));
WEAK_ALIAS(NMI)
WEAK_ALIAS(HardFault)
WEAK_ALIAS(MemManage)
WEAK_ALIAS(BusFault)
WEAK_ALIAS(UsageFault)
WEAK_ALIAS(SVC)
WEAK_ALIAS(DebugMon)
WEAK_ALIAS(PendSV)
WEAK_ALIAS(SysTick)

/* Vector table (must be at 0x08000000 for STM32F4) */
__attribute__((section(".vectors")))
void (*const g_pfnVectors[])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    NMI_Handler,
    HardFault_Handler,
    MemManage_Handler,
    BusFault_Handler,
    UsageFault_Handler,
    0, 0, 0, 0,
    SVC_Handler,
    DebugMon_Handler,
    0,
    PendSV_Handler,
    SysTick_Handler,
};
