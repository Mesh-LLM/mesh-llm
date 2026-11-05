<!--
     Copyright 2024, UNSW
     SPDX-License-Identifier: CC-BY-SA-4.0
-->
# Example - Timer

This example shows a basic timer driver for the Odroid-C4
platform. The timer driver initialises the device and then
sets a regular 1 second timeout and prints the current time
whenever the timeout expires.

## Building

```sh
mkdir build
make BUILD_DIR=build MICROKIT_BOARD=odroidc4 MICROKIT_CONFIG=<debug/release/benchmark> MICROKIT_SDK=/path/to/sdk
```

## Running

See instructions for your board in the manual.
