# skippy-hardware-profile

`skippy-hardware-profile` detects the local operating system, architecture,
GPU labels, and compatible native runtime flavors used by Mesh LLM native
runtime selection.

The crate is intentionally small and publishable. It avoids depending on the
host application runtime so the SDK, installer, updater, and CLI can share the
same flavor ranking input without pulling in the full Mesh LLM app graph.


`model_capacity` supplies the OS-reported memory budget for model discovery in
both CLIs without loading a native runtime. It uses Metal's recommended working
set, NVIDIA capacity minus driver reservations, ROCm/Intel device capacity, or
a RAM budget on CPU-only hosts. Runtime-selection profiles remain distinct
from this model-fit estimate.
