<!--
     Copyright 2024, UNSW
     SPDX-License-Identifier: CC-BY-SA-4.0
-->
# Example - Hierarchy

This example shows off the parent/child PD concept in Microkit as
well as fault handling. The parent 'restarter' PD receives faults
from the 'crasher' PD that is intentionally crashing and then
resets the crasher's program counter.

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
INFO  [sel4_capdl_initializer::initialize] Starting CapDL initializer
INFO  [sel4_capdl_initializer::initialize] Starting threads
MON|INFO: Microkit Monitor started!
restarter: starting
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: restarted
crasher, starting
restarter: received fault message for child pd: 1
restarter: too many restarts - PD stopped
hello, world
```
