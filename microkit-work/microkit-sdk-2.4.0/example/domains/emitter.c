/*
 * Copyright 2026, UNSW
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdint.h>
#include <microkit.h>

void init(void)
{
    int i;

    microkit_dbg_puts("emitter: starting to emit events...\n");
    i = 0;
    while (i < 100000) {
        i++;
        microkit_notify(0);

        if (i % 10000 == 0) {
            microkit_dbg_puts("emitter: still emitting events...\n");
        }
    }

    microkit_dbg_puts("emitter: done emitting events\n");
}

void notified(microkit_channel ch) { }
