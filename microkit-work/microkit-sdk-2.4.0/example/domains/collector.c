/*
 * Copyright 2026, UNSW
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdint.h>
#include <microkit.h>

void init(void)
{
    microkit_dbg_puts("collector: Waiting for an event\n");
}

void notified(microkit_channel ch)
{
    microkit_dbg_puts("collector: Got an event\n");
}
