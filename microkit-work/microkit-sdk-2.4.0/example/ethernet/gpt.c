/*
 * Copyright 2021, Breakaway Consulting Pty. Ltd.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdbool.h>
#include <stdint.h>
#include <microkit.h>

#define IRQ_CH 3

uintptr_t gpt_regs;
uintptr_t gpt_regs_clk;
static volatile uint32_t *gpt;
static volatile uint32_t *lpcg;

static uint64_t timeouts[MICROKIT_MAX_CHANNELS];
static microkit_channel active_channel = -1;
static bool timeout_active;
static uint64_t current_timeout;
static uint32_t overflow_count;
static uint8_t pending_timeouts;

#define CR 0
#define PR 1
#define SR 2
#define IR 3
#define OCR1 4
#define OCR2 5
#define OCR3 6
#define ICR1 7
#define ICR2 8
#define CNT 9

static char
hexchar(unsigned int v)
{
    return v < 10 ? '0' + v : ('a' - 10) + v;
}

static void
puthex32(uint32_t x)
{
    char buffer[11];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[2] = hexchar((x >> 28) & 0xf);
    buffer[3] = hexchar((x >> 24) & 0xf);
    buffer[4] = hexchar((x >> 20) & 0xf);
    buffer[5] = hexchar((x >> 16) & 0xf);
    buffer[6] = hexchar((x >> 12) & 0xf);
    buffer[7] = hexchar((x >> 8) & 0xf);
    buffer[8] = hexchar((x >> 4) & 0xf);
    buffer[9] = hexchar(x & 0xf);
    buffer[10] = 0;
    microkit_dbg_puts(buffer);
}

void
init(void)
{
    microkit_dbg_puts(microkit_name);
    microkit_dbg_puts(": gpt PD init function running\n");
    gpt = (volatile uint32_t *) gpt_regs;
    lpcg = (volatile uint32_t *) gpt_regs_clk;
    microkit_dbg_puts("LPCG: ");
    puthex32(lpcg[0]);
    microkit_dbg_puts("\n");


    uint32_t cr = (
        (1 << 9) | // Free run mode
        (1 << 6) | // Peripheral clocks
        (1) // Enable
    );
    gpt[CR] = cr;

    gpt[IR] = (
        (1 << 5) // rollover interrupt
    );

    microkit_dbg_puts("CR: ");
    puthex32(gpt[0]);
    microkit_dbg_puts("\n");
    microkit_dbg_puts("PR: ");
    puthex32(gpt[1]);
    microkit_dbg_puts("\n");
}

void
notified(microkit_channel ch)
{
    switch (ch) {

        case IRQ_CH: {
            uint32_t sr = gpt[SR];
            gpt[SR] = sr;
            microkit_irq_ack(ch);

            if (sr & (1 << 5)) {
                overflow_count++;
                /* FIXME: set the next timeout if required */
            }
            if (sr & 1) {
                gpt[IR] &= ~1;
                timeout_active = false;
                microkit_channel microkit_current_channel = active_channel;
                timeouts[microkit_current_channel] = 0;
                /* FIXME: set the next timeout if any are available */
#if 0
            microkit_dbg_puts("GPT: irq sr=");
            puthex32(sr);
            microkit_dbg_puts(" cnt=");
            puthex32(gpt[0x24 / 4]);
            microkit_dbg_puts("\n");
#endif
                microkit_notify(microkit_current_channel);
            }

            if (pending_timeouts && !timeout_active) {
                /* find next timeout */
                uint64_t next_timeout = UINT64_MAX;
                microkit_channel ch = -1;
                for (unsigned i = 0; i < MICROKIT_MAX_CHANNELS; i++) {
                    if (timeouts[i] != 0 && timeouts[i] < next_timeout) {
                        next_timeout = timeouts[i];
                        ch = i;
                    }
                }
                /* FIXME: Is there a race here?  -- Probably! Fix it later. */
                if (ch != -1 && overflow_count == (next_timeout >> 32)) {
                    pending_timeouts--;
                    gpt[OCR1] = next_timeout;
                    gpt[IR] |= 1;
                    timeout_active = true;
                    current_timeout = next_timeout;
                    active_channel = ch;
                }
            }

            break;
        }
        default:
            microkit_dbg_puts("gpt: received notification on unexpected channel\n");
            break;
    }
}

static uint64_t get_ticks(void) {
    /* FIXME: If an overflow interrupt happens in the middle here we are in trouble */
    uint64_t overflow = overflow_count;
    uint32_t sr1 = gpt[SR];
    uint32_t cnt = gpt[CNT];
    uint32_t sr2 = gpt[SR];
    if ((sr2 & (1 << 5)) && (!(sr1 & (1 << 5)))) {
        /* rolled-over during - 64-bit time must be the overflow */
        cnt = gpt[CNT];
        overflow++;
    }
    return (overflow << 32) | cnt;
}

seL4_MessageInfo_t
protected(microkit_channel ch, microkit_msginfo msginfo)
{
    switch (microkit_msginfo_get_label(msginfo)) {
        case 0:

            seL4_SetMR(0, get_ticks());
            return microkit_msginfo_new(0, 1);
        case 1: {
            /* FIXME: There is a race here, if there are higher priority
             * protection domains it is possible arbitrary amount of time
             * could elapse between any of these instructions. The code
             * should be made robust against such a possibility.
             */
            uint64_t rel_timeout = seL4_GetMR(0);
            uint64_t cur_ticks = get_ticks();
            uint64_t abs_timeout = cur_ticks + rel_timeout;
            timeouts[ch] = abs_timeout;
            if ((!timeout_active || abs_timeout < current_timeout) && (cur_ticks >> 32 == abs_timeout >> 32)) {
                if (timeout_active) {
                    /* there current timeout is now treated as pending */
                    pending_timeouts++;
                }
                gpt[OCR1] = abs_timeout;
                gpt[IR] |= 1;
                timeout_active = true;
                current_timeout = abs_timeout;
                active_channel = ch;
            } else {
                pending_timeouts++;
            }
#if 0
            microkit_dbg_puts("GPT: set timeout ch = ");
            puthex32(ch);
            microkit_dbg_puts(" - " );
            puthex32(timeout);
            microkit_dbg_puts("\n");
#endif
            return microkit_msginfo_new(0, 1);
        }
    }

    return microkit_msginfo_new(0, 0);
}