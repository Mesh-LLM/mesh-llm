//! Exact reviewed SWE-agent patch payloads. Destination SDK admission is deliberately held.
pub(super) struct Mapping<'a> {
    pub(super) source: &'a str,
    pub(super) destination: &'a str,
    pub(super) source_sha256: &'a str,
    pub(super) replacement_sha256: &'a str,
    pub(super) original_sha256: Option<&'a str>,
}

// Official1.4 preimages are admitted; old1.1 source pins bind the reviewed intent only.
pub(super) const SDK_METADATA_SHA256: Option<&str> =
    Some("0292e92f297171d415ab33e519849f68e8817cd719f3ba8cdf22b372fac5f0c8");
pub(super) const MAPPINGS: [Mapping<'static>; 3] = [
    Mapping {
        source: "swerex/deployment/modal.py",
        destination: "deployment/modal.py",
        source_sha256: "c6fb5d6aafe37663dcbbbe5caced0e09466b2df6386ae988d470d865848c805b",
        replacement_sha256: "a827f2b4c90cc56aeffb152b40a24a941af75360735613ee7a8abb457daddc41",
        original_sha256: Some("e53bdcb9da25f13a328a7e1baf58ba39b9d767d8875a7ed67898ea73f8e74651"),
    },
    Mapping {
        source: "swerex/deployment/config.py",
        destination: "deployment/config.py",
        source_sha256: "1ab1aa25f2920c86baa54ccc63ece1869253f471a7fd53740d43e1ee0ae8f721",
        replacement_sha256: "a19643cd219c3fa15de001cb520cf087165c64b47c264688dfa4c26385b02e45",
        original_sha256: Some("a19643cd219c3fa15de001cb520cf087165c64b47c264688dfa4c26385b02e45"),
    },
    Mapping {
        source: "swerex/deployment/runtime/remote.py",
        // This is the actual RemoteRuntime import destination, unlike the unused legacy copy.
        destination: "runtime/remote.py",
        source_sha256: "29c82799d044e53455e780e383bedd33bbe31ecb0d8194256d21a23386341db6",
        replacement_sha256: "7c0dd7a42dfcc99b2dc51f961cb0fe2aee81c09caae08a256694b4e37b4da532",
        original_sha256: Some("21d1639db4872a33f00d1d2cad2d967a3cc97cb7d5f33bdc2fc3cf7f79adcbf9"),
    },
];
