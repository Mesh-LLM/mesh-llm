/*
 * Copyright 2024, UNSW
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

#include <stdint.h>
#include <stdbool.h>
#include <microkit.h>

/*
 * This is a very simple timer driver with the intention of showing
 * how to do MMIO and handle interrupts in Microkit.
 */

uintptr_t timer_regs;

#define TIMER_IRQ_CH 0

#define TIMER_REG_START   0x140

#define TIMER_A_INPUT_CLK 0
#define TIMER_E_INPUT_CLK 8
#define TIMER_A_EN      (1 << 16)
#define TIMER_A_MODE    (1 << 12)

#define TIMESTAMP_TIMEBASE_SYSTEM   0b000
#define TIMESTAMP_TIMEBASE_1_US     0b001
#define TIMESTAMP_TIMEBASE_10_US    0b010
#define TIMESTAMP_TIMEBASE_100_US   0b011
#define TIMESTAMP_TIMEBASE_1_MS     0b100

#define TIMEOUT_TIMEBASE_1_US   0b00
#define TIMEOUT_TIMEBASE_10_US  0b01
#define TIMEOUT_TIMEBASE_100_US 0b10
#define TIMEOUT_TIMEBASE_1_MS   0b11

#define NS_IN_US    1000ULL
#define NS_IN_MS    1000000ULL

typedef struct {
    uint32_t mux;
    uint32_t timer_a;
    uint32_t timer_b;
    uint32_t timer_c;
    uint32_t timer_d;
    uint32_t unused[13];
    uint32_t timer_e;
    uint32_t timer_e_hi;
    uint32_t mux1;
    uint32_t timer_f;
    uint32_t timer_g;
    uint32_t timer_h;
    uint32_t timer_i;
} meson_timer_reg_t;

typedef struct {
    volatile meson_timer_reg_t *regs;
    bool disable;
} meson_timer_t;

meson_timer_t timer;

static char hexchar(unsigned int v)
{
    return v < 10 ? '0' + v : ('a' - 10) + v;
}

static void puthex64(uint64_t val)
{
    char buffer[16 + 3];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[16 + 3 - 1] = 0;
    for (unsigned i = 16 + 1; i > 1; i--) {
        buffer[i] = hexchar(val & 0xf);
        val >>= 4;
    }
    microkit_dbg_puts(buffer);
}

uint64_t meson_get_time()
{
    uint64_t initial_high = timer.regs->timer_e_hi;
    uint64_t low = timer.regs->timer_e;
    uint64_t high = timer.regs->timer_e_hi;
    if (high != initial_high) {
        low = timer.regs->timer_e;
    }

    uint64_t ticks = (high << 32) | low;
    uint64_t time = ticks * NS_IN_US;
    return time;
}

void meson_set_timeout(uint16_t timeout, bool periodic)
{
    if (periodic) {
        timer.regs->mux |= TIMER_A_MODE;
    } else {
        timer.regs->mux &= ~TIMER_A_MODE;
    }

    timer.regs->timer_a = timeout;

    if (timer.disable) {
        timer.regs->mux |= TIMER_A_EN;
        timer.disable = false;
    }
}

void meson_stop_timer()
{
    timer.regs->mux &= ~TIMER_A_EN;
    timer.disable = true;
}

void init()
{
    timer.regs = (void *)(timer_regs + TIMER_REG_START);

    timer.regs->mux = TIMER_A_EN | (TIMESTAMP_TIMEBASE_1_US << TIMER_E_INPUT_CLK) |
                      (TIMEOUT_TIMEBASE_1_MS << TIMER_A_INPUT_CLK);

    timer.regs->timer_e = 0;

    // Have a timeout of 1 second, and have it be periodic so that it will keep recurring.
    microkit_dbg_puts("Setting a timeout of 1 second.\n");
    meson_set_timeout(1000, true);
}

void notified(microkit_channel ch)
{
    switch (ch) {
    case TIMER_IRQ_CH:
        microkit_dbg_puts("Got timer interrupt!\n");
        microkit_irq_ack(ch);
        microkit_dbg_puts("Current time is: ");
        puthex64(meson_get_time());
        microkit_dbg_puts("\n");
        break;
    default:
        microkit_dbg_puts("TIMER|ERROR: unexpected channel!\n");
    }
}
