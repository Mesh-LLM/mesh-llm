/*
 * Copyright 2021, Breakaway Consulting Pty. Ltd.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
PHDRS
{
    text PT_LOAD;
    data PT_LOAD;
}

ENTRY(_start);

SECTIONS
{
    . = 0x200000;

    .text :
    {
        _text = .;
        *(.text.start)
        *(.text*)
        *(.rodata)
        _text_end = .;
    } :text

    . = ALIGN(0x1000);

    /* For some reason the ABI puts init array
     * into writable memory, so we have to follow suit */
    .init_array :
    {
        PROVIDE(__init_array_start = .);
        KEEP (*(SORT(.init_array.*)))
        KEEP (*(.init_array*))
        PROVIDE(__init_array_end = .);
    } :data

    .data.rel.ro :
    {
        . = ALIGN(8);
        __data_rel_ro_start = .;
        *(.data.rel.ro.local*) *(.data.rel.ro*) *(.data.rel*)
        __data_rel_ro_end = .;
        . = ALIGN(8);
    } :data

    .data :
    {
        _data = .;
        *(.data)
        . = ALIGN(8);
        __global_pointer$ = . + 0x800;
        *(.srodata)
        *(.sdata)
        _data_end = .;
    } :data

    .bss :
    {
        _bss = .;
        *(.sbss)
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } :data
}
