/*
 * Copyright 2021, Breakaway Consulting Pty. Ltd.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdint.h>
#include <microkit.h>

void init(void)
{
    int *x = 0;
    microkit_dbg_puts("crasher, starting\n");
    /* Crash! */
    *x = 1;
}

void notified(microkit_channel ch)
{
}