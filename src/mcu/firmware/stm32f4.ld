/* Linker script for STM32F407 (Cortex-M4)
 * Flash: 1MB @ 0x08000000
 * SRAM:  192KB @ 0x20000000
 * Simulation target: Renode STM32F4 platform
 */

MEMORY {
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    SRAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 192K
}

_estack = ORIGIN(SRAM) + LENGTH(SRAM);

SECTIONS {
    .vectors : {
        KEEP(*(.vectors))
    } > FLASH

    .text : {
        *(.text*)
        *(.rodata*)
    } > FLASH

    _sidata = LOADADDR(.data);

    .data : {
        _sdata = .;
        *(.data*)
        _edata = .;
    } > SRAM AT > FLASH

    .bss : {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > SRAM

    /DISCARD/ : {
        *(.ARM.exidx*)
    }
}
