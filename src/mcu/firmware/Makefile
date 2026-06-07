# Cross-compile LSTM autoencoder inference firmware for STM32F4 (Cortex-M4)
# Target: Renode simulation only — not intended for physical hardware flashing.
#
# Usage:
#   make          - build firmware.elf
#   make clean    - remove build artifacts
#
# Prerequisites (install via brew):
#   brew install arm-none-eabi-gcc

CC      := arm-none-eabi-gcc
OBJCOPY := arm-none-eabi-objcopy
SIZE    := arm-none-eabi-size

# STM32F407 Cortex-M4F
CPU     := cortex-m4
FPU     := fpv4-sp-d16
FLOAT   := hard

CFLAGS  := -mcpu=$(CPU) -mthumb -mfpu=$(FPU) -mfloat-abi=$(FLOAT)
CFLAGS  += -O2 -g -Wall -Wextra
CFLAGS  += -ffreestanding -fno-common -ffunction-sections -fdata-sections
CFLAGS  += -nostdlib -nostartfiles

LDFLAGS := -mcpu=$(CPU) -mthumb -mfpu=$(FPU) -mfloat-abi=$(FLOAT)
LDFLAGS += -T stm32f4.ld -Wl,--gc-sections
LDFLAGS += -nostdlib -nostartfiles

SRCS    := main.c startup_stm32f4.c
OBJS    := $(SRCS:.c=.o)
TARGET  := firmware

.PHONY: all clean size

all: $(TARGET).elf

$(TARGET).elf: $(OBJS) stm32f4.ld
	$(CC) $(LDFLAGS) $(OBJS) -o $@
	$(SIZE) $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET).bin: $(TARGET).elf
	$(OBJCOPY) -O binary $< $@

clean:
	rm -f $(OBJS) $(TARGET).elf $(TARGET).bin

size: $(TARGET).elf
	$(SIZE) $(TARGET).elf
