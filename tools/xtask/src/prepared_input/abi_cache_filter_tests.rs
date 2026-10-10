use super::*;

#[test]
fn migration_prepared_inputs_cache_filter_retains_only_legacy_keys_when_mixed() {
    let source = concat!(
        "CMAKE_HOME_DIRECTORY:INTERNAL=/checkout\n",
        "GGML_OPENMP_ENABLED:BOOL=ON\n",
        "OpenMP_C_LIB_NAMES:STRING=gomp;pthread\n",
        "OpenMP_CXX_LIB_NAMES:STRING=omp\n",
        "OpenMP_gomp_LIBRARY:FILEPATH=/usr/lib/gomp.a\n",
        "OpenMP_CXX_12_LIBRARY:UNINITIALIZED=\n",
        "OpenMP___LIBRARY: odd:type =a=b\n",
        "GGML_OPENMP_ENABLED:BOOL=ON\n",
        "CMAKE_CACHEFILE_DIR:INTERNAL=/build\n",
        "OpenMP_Fortran_LIB_NAMES:STRING=gomp\n",
        "OpenMP_LIBRARY:FILEPATH=bad\n",
        "OpenMP__LIBRARY:FILEPATH=bad\n",
        "OpenMP_a-b_LIBRARY:FILEPATH=bad\n",
        "OpenMP_é_LIBRARY:FILEPATH=bad\n",
        "OpenMP_a_LIBRARY_EXTRA:FILEPATH=bad\n",
        "GGML_OPENMP_ENABLED:=bad\n",
        "GGML_OPENMP_ENABLED=BOOL:ON\n",
        " GGML_OPENMP_ENABLED:BOOL=bad\n",
        "//GGML_OPENMP_ENABLED:BOOL=bad\n",
        "ggml_openmp_enabled:BOOL=bad\n",
        "OpenMP_C_LIB_NAMES:STRING\n",
        "GGML_OPENMP_ENABLED_EXTRA:BOOL=bad\n"
    );
    let expected = concat!(
        "# Portable MeshLLM static ABI link metadata\n",
        "GGML_OPENMP_ENABLED:BOOL=ON\n",
        "OpenMP_C_LIB_NAMES:STRING=gomp;pthread\n",
        "OpenMP_CXX_LIB_NAMES:STRING=omp\n",
        "OpenMP_gomp_LIBRARY:FILEPATH=/usr/lib/gomp.a\n",
        "OpenMP_CXX_12_LIBRARY:UNINITIALIZED=\n",
        "OpenMP___LIBRARY: odd:type =a=b\n",
        "GGML_OPENMP_ENABLED:BOOL=ON\n"
    );
    let directory = tempfile::tempdir().expect("scratch");
    let operands = CacheOperands {
        source: directory.path().join("source"),
        destination: directory.path().join("destination"),
    };
    std::fs::write(&operands.source, source).expect("source");

    let result = filter(&operands);

    assert!(result.is_ok());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        expected.as_bytes()
    );
}

#[test]
fn migration_prepared_inputs_cache_filter_translates_only_universal_newlines_when_present() {
    let directory = tempfile::tempdir().expect("scratch");
    let operands = CacheOperands {
        source: directory.path().join("source"),
        destination: directory.path().join("destination"),
    };
    std::fs::write(
        &operands.source,
        b"GGML_OPENMP_ENABLED:BOOL=ON\r\nOpenMP_C_LIB_NAMES:STRING=gomp\rOpenMP_CXX_LIB_NAMES:STRING=omp\nOpenMP_omp_LIBRARY:FILEPATH=x\x0by\x0cz\x1cw",
    ).expect("source");

    let result = filter(&operands);

    assert!(result.is_ok());
    assert_eq!(std::fs::read(&operands.destination).expect("output"), b"# Portable MeshLLM static ABI link metadata\nGGML_OPENMP_ENABLED:BOOL=ON\nOpenMP_C_LIB_NAMES:STRING=gomp\nOpenMP_CXX_LIB_NAMES:STRING=omp\nOpenMP_omp_LIBRARY:FILEPATH=x\x0by\x0cz\x1cw");
}

#[test]
fn migration_prepared_inputs_cache_filter_does_not_strip_bom_when_first_key_is_prefixed() {
    let directory = tempfile::tempdir().expect("scratch");
    let operands = CacheOperands {
        source: directory.path().join("source"),
        destination: directory.path().join("destination"),
    };
    std::fs::write(
        &operands.source,
        "\u{feff}GGML_OPENMP_ENABLED:BOOL=ON\nOpenMP_C_LIB_NAMES:STRING=omp",
    )
    .expect("source");

    let result = filter(&operands);

    assert!(result.is_ok());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        b"# Portable MeshLLM static ABI link metadata\nOpenMP_C_LIB_NAMES:STRING=omp"
    );
}

#[test]
fn migration_prepared_inputs_cache_filter_preserves_destination_when_source_is_invalid_utf8() {
    let directory = tempfile::tempdir().expect("scratch");
    let operands = CacheOperands {
        source: directory.path().join("source"),
        destination: directory.path().join("destination"),
    };
    std::fs::write(&operands.source, b"GGML_OPENMP_ENABLED:BOOL=ON\n\xff").expect("source");
    std::fs::write(&operands.destination, b"untouched").expect("destination");

    let result = filter(&operands);

    assert!(result.is_err());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        b"untouched"
    );
}

#[test]
fn migration_prepared_inputs_cache_filter_reads_before_writing_when_paths_are_identical() {
    let directory = tempfile::tempdir().expect("scratch");
    let path = directory.path().join("cache");
    let operands = CacheOperands {
        source: path.clone(),
        destination: path,
    };
    std::fs::write(&operands.source, b"GGML_OPENMP_ENABLED:BOOL=OFF").expect("source");

    let result = filter(&operands);

    assert!(result.is_ok());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        b"# Portable MeshLLM static ABI link metadata\nGGML_OPENMP_ENABLED:BOOL=OFF"
    );
}

#[test]
fn migration_prepared_inputs_cache_filter_reads_before_writing_when_paths_are_hardlink_aliases() {
    let directory = tempfile::tempdir().expect("scratch");
    let operands = CacheOperands {
        source: directory.path().join("source"),
        destination: directory.path().join("destination"),
    };
    std::fs::write(&operands.source, b"OpenMP_C_LIB_NAMES:STRING=gomp\r").expect("source");
    std::fs::hard_link(&operands.source, &operands.destination).expect("alias");

    let result = filter(&operands);

    assert!(result.is_ok());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        b"# Portable MeshLLM static ABI link metadata\nOpenMP_C_LIB_NAMES:STRING=gomp\n"
    );
}

#[test]
fn migration_prepared_inputs_cache_filter_emits_header_when_no_lines_match() {
    let directory = tempfile::tempdir().expect("scratch");
    let operands = CacheOperands {
        source: directory.path().join("source"),
        destination: directory.path().join("destination"),
    };
    std::fs::write(&operands.source, b"CMAKE_HOME_DIRECTORY:PATH=/source\n").expect("source");

    let result = filter(&operands);

    assert!(result.is_ok());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        b"# Portable MeshLLM static ABI link metadata\n"
    );
}

#[cfg(unix)]
#[test]
fn migration_prepared_inputs_cache_filter_reads_before_writing_when_destination_is_symlink() {
    let directory = tempfile::tempdir().expect("scratch");
    let operands = CacheOperands {
        source: directory.path().join("source"),
        destination: directory.path().join("destination"),
    };
    std::fs::write(&operands.source, b"GGML_OPENMP_ENABLED:BOOL=ON").expect("source");
    std::os::unix::fs::symlink(&operands.source, &operands.destination).expect("alias");

    let result = filter(&operands);

    assert!(result.is_ok());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        b"# Portable MeshLLM static ABI link metadata\nGGML_OPENMP_ENABLED:BOOL=ON"
    );
}

#[test]
fn migration_prepared_inputs_cache_filter_keeps_source_when_invalid_utf8_aliases_destination() {
    let directory = tempfile::tempdir().expect("scratch");
    let source = directory.path().join("source");
    let operands = CacheOperands {
        source: source.clone(),
        destination: source,
    };
    std::fs::write(&operands.source, b"\xfforiginal").expect("source");

    let result = filter(&operands);

    assert!(result.is_err());
    assert_eq!(
        std::fs::read(&operands.destination).expect("output"),
        b"\xfforiginal"
    );
}
