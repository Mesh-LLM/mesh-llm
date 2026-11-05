<!--
     Copyright 2026, UNSW
     SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Example - Domains

This is a basic example that uses the seL4 domain scheduler.

It is similar to the [CAmkES](https://github.com/seL4/camkes/tree/camkes-3.12.x-compatible/apps/domains)
example application 'Domains'.

It uses three domains: monitor (0), emitter (1), and collector (2).
Domain 0 runs the Microkit monitor. Domain 1 contains the emitter PD, and
Domain 2 the collector PD.

The emitter will be able to do a large number of notifies before the collector
has time to run. Because notifications are coalesced by seL4, one should not see the
100,000 notifies that the emitter produces, but only a smaller number, once
per time the collector runs. A shorter duration for the domain 1 (emitter)
will cause domain 2 (collector) to see more events.

Look at the `domains.system` file for a commented example of the syntax.

## Example Output

```
INFO  [sel4_capdl_initializer::initialize] Starting threads
MON|INFO: Microkit Monitor started!
emitter: starting to emit events...
collector: Waiting for an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
emitter: still emitting collector: Got an event
collector: Got an event
events...
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
emitter: still emitting events...
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
emitter: still emitting events...
...
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
collector: Got an event
emitter: still emitting events...
emitter: done emitting events
collector: Got an event
collector: Got an event
```

## Building

```sh
make MICROKIT_BOARD=<board> MICROKIT_CONFIG=debug MICROKIT_SDK=/path/to/sdk
```

## Running

See instructions for your board in the manual.
