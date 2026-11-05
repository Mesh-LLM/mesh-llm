<!--
     Copyright 2021, Breakaway Consulting Pty. Ltd.
     SPDX-License-Identifier: CC-BY-SA-4.0
-->
# Example - Ethernet

This example shows an ethernet system for the TQMa8XQP platform.
It also includes a driver for the general purpose timer on the platform.

## Building

```sh
mkdir build
make BUILD_DIR=build MICROKIT_BOARD=tqma8xqp1gb MICROKIT_CONFIG=<debug/release/benchmark> MICROKIT_SDK=/path/to/sdk
```

## Running

See instructions for your board in the manual.
