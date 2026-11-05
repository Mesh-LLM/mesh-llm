/*
 * Copyright 2021, Breakaway Consulting Pty. Ltd.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdint.h>
#include <microkit.h>

#define GPT_CH 0
#define OUTER_INPUT_CH 1
#define OUTER_OUTPUT_CH 2
#define INNER_INPUT_CH 3
#define INNER_OUTPUT_CH 4

#define BUFFER_SIZE (2 * 1024)
#define DATA_OFFSET 64

#define OUTER_INPUT outer_input_vaddr
#define OUTER_OUTPUT outer_output_vaddr
#define INNER_INPUT inner_input_vaddr
#define INNER_OUTPUT inner_output_vaddr


#define BUFFER_MAX 1024

unsigned outer_input_index = 0;
unsigned inner_input_index = 0;

unsigned outer_output_index = 0;
unsigned inner_output_index = 0;

uintptr_t outer_input_vaddr;
uintptr_t outer_output_vaddr;
uintptr_t inner_input_vaddr;
uintptr_t inner_output_vaddr;

struct buffer_descriptor {
    uint16_t data_length;
    uint16_t flags;
};


volatile uint64_t *shared_counter = (uint64_t *)(uintptr_t)0x1800000;

static char
hexchar(unsigned int v)
{
    return v < 10 ? '0' + v : ('a' - 10) + v;
}

static void
puthex16(uint16_t x)
{
    char buffer[7];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[2] = hexchar((x >> 12) & 0xf);
    buffer[3] = hexchar((x >> 8) & 0xf);
    buffer[4] = hexchar((x >> 4) & 0xf);
    buffer[5] = hexchar(x & 0xf);
    buffer[6] = 0;
    microkit_dbg_puts(buffer);
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


static void
puthex64(uint64_t x)
{
    char buffer[19];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[2] = hexchar((x >> 60) & 0xf);
    buffer[3] = hexchar((x >> 56) & 0xf);
    buffer[4] = hexchar((x >> 52) & 0xf);
    buffer[5] = hexchar((x >> 48) & 0xf);
    buffer[6] = hexchar((x >> 44) & 0xf);
    buffer[7] = hexchar((x >> 40) & 0xf);
    buffer[8] = hexchar((x >> 36) & 0xf);
    buffer[9] = hexchar((x >> 32) & 0xf);
    buffer[10] = hexchar((x >> 28) & 0xf);
    buffer[11] = hexchar((x >> 24) & 0xf);
    buffer[12] = hexchar((x >> 20) & 0xf);
    buffer[13] = hexchar((x >> 16) & 0xf);
    buffer[14] = hexchar((x >> 12) & 0xf);
    buffer[15] = hexchar((x >> 8) & 0xf);
    buffer[16] = hexchar((x >> 4) & 0xf);
    buffer[17] = hexchar(x & 0xf);
    buffer[18] = 0;
    microkit_dbg_puts(buffer);
}

static void
mycpy(volatile void *dst, volatile void *src, unsigned int length)
{
    volatile uint64_t *d = dst;
    volatile uint64_t *s = src;
    int i = 0;
    int l = length / 64;
    if (length % 64) {
        l++;
    }
    while (l) {
        d[i] = s[i];
        d[i + 1] = s[i + 1];
        d[i + 2] = s[i + 2];
        d[i + 3] = s[i + 3];
        d[i + 4] = s[i + 4];
        d[i + 5] = s[i + 5];
        d[i + 6] = s[i + 6];
        d[i + 7] = s[i + 7];
        l--;
        i += 8;
    }
}


static void
dump_hex(const uint8_t *d, unsigned int length)
{
    unsigned int i = 0;
    while (length) {
        puthex16(i);
        microkit_dbg_puts(": ");
        while (length) {
            microkit_dbg_putc(hexchar((d[i] >> 4) & 0xf));
            microkit_dbg_putc(hexchar(d[i] & 0xf));
            length--;
            i++;
            if (i % 16 == 0) {
                microkit_dbg_putc('\n');
                break;
            } else {
                microkit_dbg_putc(' ');
            }
        }
    }
    if (i % 16) {
        microkit_dbg_putc('\n');
    }
}

#define GPT_CHANNEL 0

static inline uint64_t
gpt_ticks(void)
{
    (void) microkit_ppcall(GPT_CHANNEL, microkit_msginfo_new(0, 0));
    return microkit_mr_get(0);
}

static inline void
gpt_timer(uint64_t timeout)
{
    microkit_mr_set(0, timeout);
    (void) microkit_ppcall(GPT_CHANNEL, microkit_msginfo_new(1, 1));
}


void
init(void)
{
    microkit_dbg_puts("pass protection domain init function running\n");

    /* Example calling a PP */
    microkit_dbg_puts("ticks: ");
    puthex32(gpt_ticks());
    microkit_dbg_puts("\n");

    gpt_timer(0x1000000);
}

void
notified(microkit_channel ch)
{
    switch (ch) {
        case GPT_CH:
            microkit_dbg_puts("tick! ticks=");
            puthex64(gpt_ticks());
            microkit_dbg_puts("\n");
            gpt_timer(0x1000000);

        case OUTER_INPUT_CH:

            for (;;) {
                volatile struct buffer_descriptor *bd = (void *)(uintptr_t)(OUTER_INPUT + (BUFFER_SIZE * outer_input_index));
                volatile void *pkt = (void *)(uintptr_t)(OUTER_INPUT + (BUFFER_SIZE * outer_input_index) + DATA_OFFSET);
                if (bd->flags == 0) {
                    break;
                }

                outer_input_index++;
                if (outer_input_index == BUFFER_MAX) {
                    outer_input_index = 0;
                }

                volatile struct buffer_descriptor *obd = (void *)(uintptr_t)(INNER_OUTPUT + (BUFFER_SIZE * inner_output_index));
                volatile void *opkt = (void *)(uintptr_t)(INNER_OUTPUT + (BUFFER_SIZE * inner_output_index) + DATA_OFFSET);
                if (obd->flags == 1) {
                    microkit_dbg_puts("PASS: outer can't pass buffer (no space for inner)\n");
                } else {
                    obd->data_length = bd->data_length;

                    mycpy(opkt, pkt, bd->data_length);
                    obd->flags = 1;

                    microkit_notify(INNER_OUTPUT_CH);

                    inner_output_index++;
                    if (inner_output_index == BUFFER_MAX) {
                        inner_output_index = 0;
                    }
                }
                bd->flags = 0;
            }

            break;

        case OUTER_OUTPUT_CH:
            microkit_dbg_puts("outer output\n");
            break;

        case INNER_INPUT_CH:
            for (;;) {
                volatile struct buffer_descriptor *bd = (void *)(uintptr_t)(INNER_INPUT + (BUFFER_SIZE * inner_input_index));
                volatile void *pkt = (void *)(uintptr_t)(INNER_INPUT + (BUFFER_SIZE * inner_input_index) + DATA_OFFSET);
                if (bd->flags == 0) {
                    break;
                }

                inner_input_index++;
                if (inner_input_index == BUFFER_MAX) {
                    inner_input_index = 0;
                }

                volatile struct buffer_descriptor *obd = (void *)(uintptr_t)(OUTER_OUTPUT + (BUFFER_SIZE * outer_output_index));
                volatile void *opkt = (void *)(uintptr_t)(OUTER_OUTPUT + (BUFFER_SIZE * outer_output_index) + DATA_OFFSET);
                if (obd->flags == 1) {
                    microkit_dbg_puts("PASS: inner can't pass buffer (no space for outer)\n");
                } else {
                    obd->data_length = bd->data_length;
                    mycpy(opkt, pkt, bd->data_length);
                    obd->flags = 1;

                    microkit_notify(OUTER_OUTPUT_CH);

                    outer_output_index++;
                    if (outer_output_index == BUFFER_MAX) {
                        outer_output_index = 0;
                    }
                }

                bd->flags = 0;

            }
            break;

        case INNER_OUTPUT_CH:
            microkit_dbg_puts("inner output\n");
            break;

        default:
            microkit_dbg_puts("foo: received notification on unexpected channel\n");
            break;
        /* ignore any other channels */
    }
}