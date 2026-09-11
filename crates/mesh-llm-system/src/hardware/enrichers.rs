use super::GpuFacts;

#[cfg(target_os = "linux")]
mod linux {
    use super::GpuFacts;
    use libc::{c_char, c_int, c_uint, c_void};
    use std::ffi::CStr;

    #[derive(Clone, Debug, Default, PartialEq)]
    struct NvidiaDeviceInfo {
        name: Option<String>,
        pci_bdf: Option<String>,
        total_bytes: Option<u64>,
        reserved_bytes: Option<u64>,
        uuid: Option<String>,
    }

    struct DlLibrary(*mut c_void);

    impl DlLibrary {
        fn open(name: &'static [u8]) -> Option<Self> {
            let handle = unsafe { libc::dlopen(name.as_ptr().cast(), libc::RTLD_LAZY) };
            if handle.is_null() {
                None
            } else {
                Some(Self(handle))
            }
        }

        unsafe fn symbol<T: Copy>(&self, name: &'static [u8]) -> Option<T> {
            let symbol = unsafe { libc::dlsym(self.0, name.as_ptr().cast()) };
            if symbol.is_null() {
                None
            } else {
                Some(unsafe { std::mem::transmute_copy(&symbol) })
            }
        }
    }

    impl Drop for DlLibrary {
        fn drop(&mut self) {
            unsafe {
                libc::dlclose(self.0);
            }
        }
    }

    type CuDevice = c_int;
    type CuResult = c_int;
    type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> CuResult;
    type CuDeviceGet = unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult;
    type CuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;
    type CuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CuDevice) -> CuResult;
    type CuDeviceGetPciBusId = unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult;

    type NvmlDevice = *mut c_void;
    type NvmlReturn = c_int;
    type NvmlInit = unsafe extern "C" fn() -> NvmlReturn;
    type NvmlShutdown = unsafe extern "C" fn() -> NvmlReturn;
    type NvmlDeviceGetCount = unsafe extern "C" fn(*mut c_uint) -> NvmlReturn;
    type NvmlDeviceGetHandleByIndex = unsafe extern "C" fn(c_uint, *mut NvmlDevice) -> NvmlReturn;
    type NvmlDeviceGetUuid = unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> NvmlReturn;
    type NvmlDeviceGetName = unsafe extern "C" fn(NvmlDevice, *mut c_char, c_uint) -> NvmlReturn;
    type NvmlDeviceGetPciInfo = unsafe extern "C" fn(NvmlDevice, *mut NvmlPciInfo) -> NvmlReturn;
    type NvmlDeviceGetMemoryInfo = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> NvmlReturn;
    type NvmlDeviceGetMemoryInfoV2 =
        unsafe extern "C" fn(NvmlDevice, *mut NvmlMemoryV2) -> NvmlReturn;

    #[repr(C)]
    #[derive(Default)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        used: u64,
    }

    #[repr(C)]
    #[derive(Default)]
    struct NvmlMemoryV2 {
        version: c_uint,
        total: u64,
        reserved: u64,
        free: u64,
        used: u64,
    }

    /// `nvmlPciInfo_t` as `nvmlDeviceGetPciInfo_v3` writes it. `_reserved` is
    /// slack, not a field: a driver whose struct grew past the v3 layout still
    /// writes inside this allocation instead of past the end of it.
    #[repr(C)]
    struct NvmlPciInfo {
        bus_id_legacy: [c_char; 16],
        domain: c_uint,
        bus: c_uint,
        device: c_uint,
        pci_device_id: c_uint,
        pci_sub_system_id: c_uint,
        bus_id: [c_char; 32],
        _reserved: [u8; 64],
    }

    const NVML_SUCCESS: NvmlReturn = 0;
    const CUDA_SUCCESS: CuResult = 0;

    pub(crate) fn enrich_gpu_facts(gpus: &mut [GpuFacts]) {
        let mut infos = cuda_device_infos();
        merge_device_infos(&mut infos, &nvml_device_infos());
        if infos.is_empty() {
            return;
        }

        let unidentified = enrich_nvidia_gpu_facts(gpus, &infos);
        if !unidentified.is_empty() {
            tracing::warn!(
                devices = ?unidentified,
                "no NVIDIA driver device matched by PCI address or UUID; \
                 reporting backend VRAM without driver enrichment"
            );
        }
    }

    /// Applies driver facts to every GPU whose identity is present in `infos`.
    /// Returns the display names of the NVIDIA GPUs that matched nothing, so
    /// the caller can report a survey that degraded rather than one that
    /// silently borrowed another device's numbers.
    fn enrich_nvidia_gpu_facts(gpus: &mut [GpuFacts], infos: &[NvidiaDeviceInfo]) -> Vec<String> {
        let mut unidentified = Vec::new();
        for gpu in gpus {
            let Some(info) = match_nvidia_device(gpu, infos) else {
                if looks_like_nvidia(gpu) {
                    unidentified.push(gpu.display_name.clone());
                }
                continue;
            };
            if let Some(total_bytes) = info.total_bytes {
                gpu.vram_bytes = total_bytes;
            }
            if info.reserved_bytes.is_some() {
                gpu.reserved_bytes = info.reserved_bytes;
            }
            if let Some(uuid) = &info.uuid {
                gpu.vendor_uuid = Some(uuid.clone());
                if gpu.stable_id.as_deref().is_none_or(|stable_id| {
                    stable_id.starts_with("index:")
                        || stable_id.starts_with("cuda")
                        || stable_id.starts_with("vulkan")
                }) {
                    gpu.stable_id = Some(format!("uuid:{uuid}"));
                }
            }
            if let Some(pci_bdf) = &info.pci_bdf {
                gpu.pci_bdf = Some(pci_bdf.clone());
                if !super::super::is_placeholder_pci_bdf(pci_bdf) {
                    gpu.stable_id = Some(format!("pci:{pci_bdf}"));
                }
            }
        }
        unidentified
    }

    fn looks_like_nvidia(gpu: &GpuFacts) -> bool {
        gpu.display_name.to_ascii_lowercase().contains("nvidia")
            || gpu.vendor_uuid.is_some()
            || gpu
                .backend_device
                .as_deref()
                .is_some_and(|name| name.starts_with("CUDA") || name.starts_with("Vulkan"))
    }

    /// Resolves a GPU to a driver device by identity alone.
    ///
    /// There is deliberately no positional fallback. The backend device list
    /// and the driver device list are produced by separate enumerations with
    /// different visibility rules, so equal indices do not imply the same
    /// card: under `CUDA_VISIBLE_DEVICES=1` the only visible backend device
    /// sits at index 0 while the driver still reports the hidden card there.
    fn match_nvidia_device<'a>(
        gpu: &GpuFacts,
        infos: &'a [NvidiaDeviceInfo],
    ) -> Option<&'a NvidiaDeviceInfo> {
        if !looks_like_nvidia(gpu) {
            return None;
        }

        let pci_match = gpu
            .pci_bdf
            .as_deref()
            .and_then(normalize_pci_bdf)
            .and_then(|pci_bdf| {
                infos
                    .iter()
                    .find(|info| info.pci_bdf.as_deref() == Some(pci_bdf.as_str()))
            });
        if pci_match.is_some() {
            return pci_match;
        }

        gpu.vendor_uuid
            .as_deref()
            .and_then(|uuid| infos.iter().find(|info| info.uuid.as_deref() == Some(uuid)))
    }

    fn cuda_device_infos() -> Vec<NvidiaDeviceInfo> {
        let Some(lib) = DlLibrary::open(b"libcuda.so.1\0") else {
            return Vec::new();
        };
        let Some(cu_init) = (unsafe { lib.symbol::<CuInit>(b"cuInit\0") }) else {
            return Vec::new();
        };
        let Some(cu_device_get_count) =
            (unsafe { lib.symbol::<CuDeviceGetCount>(b"cuDeviceGetCount\0") })
        else {
            return Vec::new();
        };
        let Some(cu_device_get) = (unsafe { lib.symbol::<CuDeviceGet>(b"cuDeviceGet\0") }) else {
            return Vec::new();
        };
        let cu_device_total_mem =
            unsafe { lib.symbol::<CuDeviceTotalMem>(b"cuDeviceTotalMem_v2\0") };
        let cu_device_get_name = unsafe { lib.symbol::<CuDeviceGetName>(b"cuDeviceGetName\0") };
        let cu_device_get_pci_bus_id =
            unsafe { lib.symbol::<CuDeviceGetPciBusId>(b"cuDeviceGetPCIBusId\0") };

        if unsafe { cu_init(0) } != CUDA_SUCCESS {
            return Vec::new();
        }

        let mut count = 0;
        if unsafe { cu_device_get_count(&mut count) } != CUDA_SUCCESS || count <= 0 {
            return Vec::new();
        }

        let mut infos = Vec::new();
        for index in 0..count {
            let mut device = 0;
            if unsafe { cu_device_get(&mut device, index) } != CUDA_SUCCESS {
                continue;
            }

            let mut info = NvidiaDeviceInfo::default();
            if let Some(device_name) = cu_device_get_name {
                let mut buf = [0 as c_char; 256];
                if unsafe { device_name(buf.as_mut_ptr(), buf.len() as c_int, device) }
                    == CUDA_SUCCESS
                {
                    info.name = unsafe { c_string(buf.as_ptr()) };
                }
            }
            if let Some(total_mem) = cu_device_total_mem {
                let mut total = 0usize;
                if unsafe { total_mem(&mut total, device) } == CUDA_SUCCESS {
                    info.total_bytes = Some(total as u64);
                }
            }
            if let Some(pci_bus_id) = cu_device_get_pci_bus_id {
                let mut buf = [0 as c_char; 32];
                if unsafe { pci_bus_id(buf.as_mut_ptr(), buf.len() as c_int, device) }
                    == CUDA_SUCCESS
                {
                    info.pci_bdf = unsafe { c_string(buf.as_ptr()) }
                        .as_deref()
                        .and_then(normalize_pci_bdf);
                }
            }
            infos.push(info);
        }

        infos
    }

    /// Joins driver facts onto the CUDA-visible device list by identity.
    ///
    /// NVML ignores `CUDA_VISIBLE_DEVICES` and libcuda honours it, so the two
    /// lists have neither the same length nor the same order. Matching on PCI
    /// address (or UUID, when only that is available) keeps a hidden card's
    /// memory out of a visible card's slot. An NVML device that matches nothing
    /// is appended, which is what keeps hosts with NVML but no usable libcuda
    /// enumerating at all.
    fn merge_device_infos(cuda: &mut Vec<NvidiaDeviceInfo>, nvml: &[NvidiaDeviceInfo]) {
        for info in nvml {
            match cuda.iter_mut().find(|visible| same_device(visible, info)) {
                Some(visible) => apply_nvml_facts(visible, info),
                None => cuda.push(info.clone()),
            }
        }
    }

    /// PCI address decides identity when both sides report one; it is stable
    /// across driver restarts and is the key the backend device list also
    /// carries. UUID is the fallback for devices that report no address.
    fn same_device(left: &NvidiaDeviceInfo, right: &NvidiaDeviceInfo) -> bool {
        if let (Some(left_bdf), Some(right_bdf)) =
            (left.pci_bdf.as_deref(), right.pci_bdf.as_deref())
        {
            return left_bdf == right_bdf;
        }
        match (left.uuid.as_deref(), right.uuid.as_deref()) {
            (Some(left_uuid), Some(right_uuid)) => left_uuid == right_uuid,
            _ => false,
        }
    }

    fn apply_nvml_facts(target: &mut NvidiaDeviceInfo, source: &NvidiaDeviceInfo) {
        if source.uuid.is_some() {
            target.uuid = source.uuid.clone();
        }
        if source.total_bytes.is_some() {
            target.total_bytes = source.total_bytes;
        }
        if source.reserved_bytes.is_some() {
            target.reserved_bytes = source.reserved_bytes;
        }
        if target.name.is_none() {
            target.name = source.name.clone();
        }
        if target.pci_bdf.is_none() {
            target.pci_bdf = source.pci_bdf.clone();
        }
    }

    fn nvml_device_infos() -> Vec<NvidiaDeviceInfo> {
        let Some(lib) = DlLibrary::open(b"libnvidia-ml.so.1\0") else {
            return Vec::new();
        };
        let Some(nvml_init) = (unsafe { lib.symbol::<NvmlInit>(b"nvmlInit_v2\0") }) else {
            return Vec::new();
        };
        let Some(nvml_device_get_count) =
            (unsafe { lib.symbol::<NvmlDeviceGetCount>(b"nvmlDeviceGetCount_v2\0") })
        else {
            return Vec::new();
        };
        let Some(nvml_device_get_handle_by_index) = (unsafe {
            lib.symbol::<NvmlDeviceGetHandleByIndex>(b"nvmlDeviceGetHandleByIndex_v2\0")
        }) else {
            return Vec::new();
        };
        let nvml_shutdown = unsafe { lib.symbol::<NvmlShutdown>(b"nvmlShutdown\0") };
        let nvml_device_get_uuid =
            unsafe { lib.symbol::<NvmlDeviceGetUuid>(b"nvmlDeviceGetUUID\0") };
        let nvml_device_get_name =
            unsafe { lib.symbol::<NvmlDeviceGetName>(b"nvmlDeviceGetName\0") };
        let nvml_device_get_pci_info =
            unsafe { lib.symbol::<NvmlDeviceGetPciInfo>(b"nvmlDeviceGetPciInfo_v3\0") };
        let nvml_device_get_memory_info =
            unsafe { lib.symbol::<NvmlDeviceGetMemoryInfo>(b"nvmlDeviceGetMemoryInfo\0") };
        let nvml_device_get_memory_info_v2 =
            unsafe { lib.symbol::<NvmlDeviceGetMemoryInfoV2>(b"nvmlDeviceGetMemoryInfo_v2\0") };

        if unsafe { nvml_init() } != NVML_SUCCESS {
            return Vec::new();
        }

        let mut infos = Vec::new();
        let mut count = 0;
        if unsafe { nvml_device_get_count(&mut count) } == NVML_SUCCESS {
            for index in 0..count {
                let mut device = std::ptr::null_mut();
                if unsafe { nvml_device_get_handle_by_index(index, &mut device) } != NVML_SUCCESS {
                    continue;
                }

                let mut info = NvidiaDeviceInfo::default();
                if let Some(get_uuid) = nvml_device_get_uuid {
                    let mut buf = [0 as c_char; 96];
                    if unsafe { get_uuid(device, buf.as_mut_ptr(), buf.len() as c_uint) }
                        == NVML_SUCCESS
                    {
                        info.uuid = unsafe { c_string(buf.as_ptr()) };
                    }
                }
                if let Some(get_name) = nvml_device_get_name {
                    let mut buf = [0 as c_char; 96];
                    if unsafe { get_name(device, buf.as_mut_ptr(), buf.len() as c_uint) }
                        == NVML_SUCCESS
                    {
                        info.name = unsafe { c_string(buf.as_ptr()) };
                    }
                }
                if let Some(get_pci_info) = nvml_device_get_pci_info {
                    let mut pci: NvmlPciInfo = unsafe { std::mem::zeroed() };
                    if unsafe { get_pci_info(device, &mut pci) } == NVML_SUCCESS {
                        info.pci_bdf = unsafe { c_string(pci.bus_id.as_ptr()) }
                            .or_else(|| unsafe { c_string(pci.bus_id_legacy.as_ptr()) })
                            .as_deref()
                            .and_then(normalize_pci_bdf);
                    }
                }
                if let Some(get_memory_v2) = nvml_device_get_memory_info_v2 {
                    let mut memory = NvmlMemoryV2 {
                        version: (std::mem::size_of::<NvmlMemoryV2>() as c_uint) | (2 << 24),
                        ..NvmlMemoryV2::default()
                    };
                    if unsafe { get_memory_v2(device, &mut memory) } == NVML_SUCCESS {
                        info.total_bytes = Some(memory.total);
                        info.reserved_bytes = Some(round_up_to_mib(memory.reserved));
                    }
                } else if let Some(get_memory) = nvml_device_get_memory_info {
                    let mut memory = NvmlMemory::default();
                    if unsafe { get_memory(device, &mut memory) } == NVML_SUCCESS {
                        info.total_bytes = Some(memory.total);
                    }
                }

                infos.push(info);
            }
        }

        if let Some(shutdown) = nvml_shutdown {
            unsafe {
                shutdown();
            }
        }

        infos
    }

    unsafe fn c_string(ptr: *const c_char) -> Option<String> {
        if ptr.is_null() {
            return None;
        }
        let value = unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .trim()
            .to_string();
        if value.is_empty() { None } else { Some(value) }
    }

    /// Canonicalises a PCI address to lowercase `00000000:bb:dd.f`.
    ///
    /// Case has to be folded here because the two sides that get compared
    /// disagree on it: NVML documents its bus id as `%08X:%02X:%02X.0`, while
    /// ggml lowercases the CUDA `device_id` it exports. Any domain, bus, or
    /// device containing `A`-`F` would otherwise compare unequal and miss
    /// either the CUDA/NVML merge or the backend-device match, which is the
    /// same failure this module exists to prevent.
    fn normalize_pci_bdf(value: &str) -> Option<String> {
        let trimmed = value.trim();
        let (domain, rest) = trimmed.split_once(':')?;
        if !rest.contains(':') || !rest.contains('.') {
            return None;
        }
        match domain.len() {
            4 => Some(format!("0000{trimmed}").to_ascii_lowercase()),
            8 => Some(trimmed.to_ascii_lowercase()),
            _ => None,
        }
    }

    fn round_up_to_mib(bytes: u64) -> u64 {
        const MIB: u64 = 1024 * 1024;
        bytes.div_ceil(MIB) * MIB
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // The numbers below are the real carrack readings behind #1755: an
        // RTX 5090 at 00000000:01:00.0 and an RTX 3080 at 00000000:06:00.0,
        // captured with CUDA_VISIBLE_DEVICES unset, 0, 1, and 1,0.
        const RTX_5090_BDF: &str = "00000000:01:00.0";
        const RTX_5090_UUID: &str = "GPU-80ded6bd-1a89-2628-3d94-902187dbab1d";
        const RTX_5090_NVML_TOTAL: u64 = 34_190_917_632;
        const RTX_5090_NVML_RESERVED: u64 = 514_850_816;
        const RTX_3080_BDF: &str = "00000000:06:00.0";
        const RTX_3080_UUID: &str = "GPU-6b7fe24c-5f15-4ac5-88d6-c8934135a4ea";
        const RTX_3080_CUDA_TOTAL: u64 = 10_354_032_640;
        const RTX_3080_NVML_TOTAL: u64 = 10_737_418_240;
        const RTX_3080_NVML_RESERVED: u64 = 383_778_816;
        // A slot whose bus digit is hexadecimal, spelled the way each side
        // actually spells it: NVML uses uppercase, ggml lowercases its
        // exported CUDA device id.
        const HEX_BDF_NVML: &str = "00000000:AF:00.0";
        const HEX_BDF_BACKEND: &str = "0000:af:00.0";
        const HEX_BDF_CANONICAL: &str = "00000000:af:00.0";
        const HEX_UUID: &str = "GPU-1c0ffee0-dead-4bee-9f00-0d15ea5eb0a7";
        const HEX_CUDA_TOTAL: u64 = 23_836_852_224;
        const HEX_NVML_TOTAL: u64 = 25_757_220_864;
        const HEX_NVML_RESERVED: u64 = 452_984_832;

        fn gpu(display_name: &str, index: usize, vram_bytes: u64) -> GpuFacts {
            GpuFacts {
                index,
                display_name: display_name.to_string(),
                backend_device: Some(format!("CUDA{index}")),
                vram_bytes,
                stable_id: Some(format!("cuda{index}")),
                ..GpuFacts::default()
            }
        }

        fn visible_3080() -> GpuFacts {
            GpuFacts {
                pci_bdf: Some(RTX_3080_BDF.to_string()),
                stable_id: Some(format!("pci:{RTX_3080_BDF}")),
                ..gpu("NVIDIA GeForce RTX 3080", 0, RTX_3080_CUDA_TOTAL)
            }
        }

        fn info(
            bdf: Option<&str>,
            uuid: Option<&str>,
            total: u64,
            reserved: u64,
        ) -> NvidiaDeviceInfo {
            NvidiaDeviceInfo {
                name: None,
                pci_bdf: bdf.map(str::to_string),
                total_bytes: Some(total),
                reserved_bytes: Some(reserved),
                uuid: uuid.map(str::to_string),
            }
        }

        fn info_5090() -> NvidiaDeviceInfo {
            info(
                Some(RTX_5090_BDF),
                Some(RTX_5090_UUID),
                RTX_5090_NVML_TOTAL,
                RTX_5090_NVML_RESERVED,
            )
        }

        fn info_3080() -> NvidiaDeviceInfo {
            info(
                Some(RTX_3080_BDF),
                Some(RTX_3080_UUID),
                RTX_3080_NVML_TOTAL,
                RTX_3080_NVML_RESERVED,
            )
        }

        #[test]
        fn index_fallback_does_not_borrow_memory_from_a_different_device() {
            let mut gpus = vec![GpuFacts {
                pci_bdf: None,
                ..gpu("NVIDIA GeForce RTX 3080", 0, RTX_3080_CUDA_TOTAL)
            }];

            enrich_nvidia_gpu_facts(&mut gpus, &[info_5090()]);

            assert_eq!(gpus[0].vram_bytes, RTX_3080_CUDA_TOTAL);
        }

        #[test]
        fn enrichment_declines_when_no_identity_matches() {
            let mut gpus = vec![GpuFacts {
                pci_bdf: None,
                ..gpu("NVIDIA GeForce RTX 3080", 0, RTX_3080_CUDA_TOTAL)
            }];

            let unidentified = enrich_nvidia_gpu_facts(&mut gpus, &[info_5090()]);

            assert_eq!(gpus[0].reserved_bytes, None);
            assert_eq!(gpus[0].vendor_uuid, None);
            assert_eq!(gpus[0].stable_id.as_deref(), Some("cuda0"));
            assert_eq!(unidentified, vec!["NVIDIA GeForce RTX 3080".to_string()]);
        }

        #[test]
        fn pci_bdf_match_wins_over_index_position() {
            let mut gpus = vec![visible_3080()];

            let unidentified = enrich_nvidia_gpu_facts(&mut gpus, &[info_5090(), info_3080()]);

            assert_eq!(gpus[0].vram_bytes, RTX_3080_NVML_TOTAL);
            assert_eq!(gpus[0].reserved_bytes, Some(RTX_3080_NVML_RESERVED));
            assert_eq!(gpus[0].vendor_uuid.as_deref(), Some(RTX_3080_UUID));
            assert!(unidentified.is_empty());
        }

        #[test]
        fn uuid_match_used_when_pci_bdf_is_a_placeholder() {
            let mut gpus = vec![GpuFacts {
                pci_bdf: Some("0".to_string()),
                vendor_uuid: Some(RTX_3080_UUID.to_string()),
                ..gpu("NVIDIA GeForce RTX 3080", 0, RTX_3080_CUDA_TOTAL)
            }];

            enrich_nvidia_gpu_facts(&mut gpus, &[info_5090(), info_3080()]);

            assert_eq!(gpus[0].vram_bytes, RTX_3080_NVML_TOTAL);
            assert_eq!(gpus[0].pci_bdf.as_deref(), Some(RTX_3080_BDF));
        }

        #[test]
        fn single_visible_device_still_requires_identity_agreement() {
            let mut gpus = vec![GpuFacts {
                pci_bdf: None,
                ..gpu("NVIDIA GeForce RTX 3080", 0, RTX_3080_CUDA_TOTAL)
            }];

            enrich_nvidia_gpu_facts(&mut gpus, &[info_5090()]);

            assert_eq!(gpus[0].vram_bytes, RTX_3080_CUDA_TOTAL);
            assert_eq!(gpus[0].vendor_uuid, None);
        }

        #[test]
        fn nvml_merge_keys_by_identity_not_position() {
            // CUDA_VISIBLE_DEVICES=1: libcuda sees only the 3080, NVML still
            // reports both with the 5090 first.
            let mut cuda = vec![info(Some(RTX_3080_BDF), None, RTX_3080_CUDA_TOTAL, 0)];
            cuda[0].reserved_bytes = None;

            merge_device_infos(&mut cuda, &[info_5090(), info_3080()]);

            let matched = cuda
                .iter()
                .find(|entry| entry.pci_bdf.as_deref() == Some(RTX_3080_BDF))
                .expect("the visible 3080 survives the merge");
            assert_eq!(matched.total_bytes, Some(RTX_3080_NVML_TOTAL));
            assert_eq!(matched.uuid.as_deref(), Some(RTX_3080_UUID));
            assert!(
                cuda.iter()
                    .all(|entry| entry.pci_bdf.as_deref() != Some(RTX_3080_BDF)
                        || entry.total_bytes != Some(RTX_5090_NVML_TOTAL)),
                "the hidden 5090's memory must not land in the 3080's slot"
            );
        }

        #[test]
        fn nvml_only_hosts_still_enumerate_when_libcuda_is_unavailable() {
            let mut cuda = Vec::new();

            merge_device_infos(&mut cuda, &[info_5090(), info_3080()]);

            assert_eq!(cuda.len(), 2);
            assert_eq!(cuda[0].total_bytes, Some(RTX_5090_NVML_TOTAL));
            assert_eq!(cuda[1].total_bytes, Some(RTX_3080_NVML_TOTAL));
        }

        #[test]
        fn mismatched_list_lengths_never_resolve_positionally() {
            let visible = visible_3080();

            // One visible device, two driver devices, and the driver list is
            // ordered so position 0 is the wrong card. This is the invariant
            // that broke in #1755.
            assert!(match_nvidia_device(&visible, &[info_5090()]).is_none());

            let both = [info_5090(), info_3080()];
            let matched =
                match_nvidia_device(&visible, &both).expect("identity match still resolves");
            assert_eq!(matched.total_bytes, Some(RTX_3080_NVML_TOTAL));
        }

        #[test]
        fn non_nvidia_gpus_are_left_alone_and_not_reported_as_unidentified() {
            let mut gpus = vec![GpuFacts {
                backend_device: Some("ROCm0".to_string()),
                stable_id: Some("pci:00000000:65:00.0".to_string()),
                ..gpu("AMD Instinct MI300X", 0, 206_158_430_208)
            }];

            let unidentified = enrich_nvidia_gpu_facts(&mut gpus, &[info_5090()]);

            assert_eq!(gpus[0].vram_bytes, 206_158_430_208);
            assert!(unidentified.is_empty());
        }

        #[test]
        fn four_digit_and_eight_digit_pci_domains_normalize_to_one_form() {
            assert_eq!(
                normalize_pci_bdf("0000:06:00.0").as_deref(),
                Some(RTX_3080_BDF)
            );
            assert_eq!(
                normalize_pci_bdf("00000000:06:00.0").as_deref(),
                Some(RTX_3080_BDF)
            );
            assert_eq!(normalize_pci_bdf("0"), None);
        }

        #[test]
        fn hexadecimal_pci_addresses_normalize_to_one_case() {
            // NVML's uppercase spelling and ggml's lowercase spelling of the
            // same slot have to land on the same key, or every comparison
            // below them is a miss.
            assert_eq!(
                normalize_pci_bdf(HEX_BDF_NVML).as_deref(),
                Some(HEX_BDF_CANONICAL)
            );
            assert_eq!(
                normalize_pci_bdf(HEX_BDF_BACKEND).as_deref(),
                Some(HEX_BDF_CANONICAL)
            );
            assert_eq!(
                normalize_pci_bdf("00AB:CD:EF.0").as_deref(),
                Some("000000ab:cd:ef.0")
            );
        }

        #[test]
        fn hexadecimal_pci_case_does_not_split_the_nvml_merge() {
            let mut cuda = vec![info(
                normalize_pci_bdf(HEX_BDF_BACKEND).as_deref(),
                None,
                HEX_CUDA_TOTAL,
                0,
            )];
            cuda[0].reserved_bytes = None;

            merge_device_infos(
                &mut cuda,
                &[info(
                    normalize_pci_bdf(HEX_BDF_NVML).as_deref(),
                    Some(HEX_UUID),
                    HEX_NVML_TOTAL,
                    HEX_NVML_RESERVED,
                )],
            );

            assert_eq!(cuda.len(), 1, "the same slot must not merge as two devices");
            assert_eq!(cuda[0].total_bytes, Some(HEX_NVML_TOTAL));
            assert_eq!(cuda[0].reserved_bytes, Some(HEX_NVML_RESERVED));
            assert_eq!(cuda[0].uuid.as_deref(), Some(HEX_UUID));
        }

        #[test]
        fn hexadecimal_pci_case_does_not_split_the_backend_match() {
            let mut gpus = vec![GpuFacts {
                pci_bdf: Some(HEX_BDF_NVML.to_string()),
                stable_id: Some(format!("pci:{HEX_BDF_NVML}")),
                ..gpu("NVIDIA GeForce RTX 4090", 0, HEX_CUDA_TOTAL)
            }];

            let unidentified = enrich_nvidia_gpu_facts(
                &mut gpus,
                &[
                    info_5090(),
                    info(
                        normalize_pci_bdf(HEX_BDF_BACKEND).as_deref(),
                        Some(HEX_UUID),
                        HEX_NVML_TOTAL,
                        HEX_NVML_RESERVED,
                    ),
                ],
            );

            assert!(unidentified.is_empty());
            assert_eq!(gpus[0].vram_bytes, HEX_NVML_TOTAL);
            assert_eq!(gpus[0].reserved_bytes, Some(HEX_NVML_RESERVED));
            assert_eq!(gpus[0].vendor_uuid.as_deref(), Some(HEX_UUID));
            assert_eq!(gpus[0].pci_bdf.as_deref(), Some(HEX_BDF_CANONICAL));
            assert_eq!(
                gpus[0].stable_id.as_deref(),
                Some(format!("pci:{HEX_BDF_CANONICAL}").as_str())
            );
        }
    }
}

#[cfg(target_os = "linux")]
pub(super) use linux::enrich_gpu_facts;

#[cfg(not(target_os = "linux"))]
pub(super) fn enrich_gpu_facts(_gpus: &mut [GpuFacts]) {}
