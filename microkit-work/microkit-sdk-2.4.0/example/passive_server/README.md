<!--
     Copyright 2024, UNSW
     SPDX-License-Identifier: CC-BY-SA-4.0
-->
# Example - Passive Server

This example shows a client and server PD communicating with each
other where the server does not have an active scheduling context
and is therefore a 'passive' server.

When the client PPCs into the server, the server is executing on
the client's budget. See the manual for more details on passive
PDs.

All supported platforms are supported in this example.

## Building

```sh
mkdir build
make BUILD_DIR=build MICROKIT_BOARD=<board> MICROKIT_CONFIG=<debug/release/benchmark> MICROKIT_SDK=/path/to/sdk
```

## Running

See instructions for your board in the manual.

You should see the following output:

```
MON|INFO: Microkit Monitor started!
SERVER|INFO: init function running
MON|INFO: PD 'server' is now passive!
CLIENT|INFO: init function running
SERVER|INFO: running on clients scheduling context
CLIENT|INFO: server replied to our message
```
