/*
 * Copyright 2021, Breakaway Consulting Pty. Ltd.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdint.h>
#include <microkit.h>

static uint8_t restart_count = 0;

static char decchar(unsigned int v)
{
    return '0' + v;
}

static void put8(uint8_t x)
{
    char tmp[4];
    unsigned i = 3;
    tmp[3] = 0;
    do {
        uint8_t c = x % 10;
        tmp[--i] = decchar(c);
        x /= 10;
    } while (x);
    microkit_dbg_puts(&tmp[i]);
}

void init(void)
{
    microkit_dbg_puts("restarter: starting\n");
}

void notified(microkit_channel ch)
{
}

seL4_MessageInfo_t protected(microkit_channel ch, microkit_msginfo msginfo)
{
    microkit_dbg_puts("restarter: received protected message\n");

    return microkit_msginfo_new(0, 0);
}

seL4_Bool fault(microkit_child child, microkit_msginfo msginfo, microkit_msginfo *reply_msginfo)
{
    microkit_dbg_puts("restarter: received fault message for child pd: ");
    put8(child);
    microkit_dbg_puts("\n");
    restart_count++;
    if (restart_count < 10) {
        microkit_pd_restart(child, 0x200000);
        microkit_dbg_puts("restarter: restarted\n");
    } else {
        microkit_pd_stop(child);
        microkit_dbg_puts("restarter: too many restarts - PD stopped\n");
    }

    /* We explicitly restart the thread so we do not need to 'reply' to the fault. */
    return seL4_False;
}
