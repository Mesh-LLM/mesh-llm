#!/usr/bin/env zsh
# build-mac.sh — build patched llama.cpp ABI libraries + mesh-llm on macOS.

setopt errexit nounset pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"

LLAMA_DIR="${MESH_LLM_LLAMA_DIR:-$REPO_ROOT/.deps/llama.cpp}"
LLAMA_BUILD_ROOT="${MESH_LLM_LLAMA_BUILD_ROOT:-$REPO_ROOT/.deps/llama-build}"
MESH_DIR="$REPO_ROOT/crates/mesh-llm"
UI_DIR="$REPO_ROOT/crates/mesh-llm-ui"
build_profile="${MESH_LLM_BUILD_PROFILE:-debug}"
MESH_LLM_LOCAL_CODESIGN_IDENTITY="${MESH_LLM_LOCAL_CODESIGN_IDENTITY:-Mesh-LLM Local Codesign}"
MESH_LLM_AUTO_GENERATE_CODESIGN="${MESH_LLM_AUTO_GENERATE_CODESIGN:-1}"
rustc_wrapper=""
build_profile="${build_profile:l}"

append_rustflag() {
    local flag="$1"
    case " ${RUSTFLAGS:-} " in
        *" $flag "*) ;;
        *) export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }$flag" ;;
    esac
}

stamp_build_version() {
    local release_version=""
    local pkgid=""
    local sha=""
    local dirty_suffix=""
    local status_output=""

    if [[ -n "${MESH_LLM_BUILD_VERSION:-}" ]]; then
        echo "Using preset MESH_LLM_BUILD_VERSION: $MESH_LLM_BUILD_VERSION"
        return 0
    fi

    if ! pkgid="$(cd "$REPO_ROOT" && cargo pkgid -p mesh-llm 2>/dev/null)"; then
        echo "Warning: unable to derive build version; cargo pkgid unavailable." >&2
        unset MESH_LLM_BUILD_VERSION || true
        return 0
    fi
    release_version="${pkgid##*#}"
    if [[ -z "$release_version" || "$release_version" == "$pkgid" ]]; then
        echo "Warning: unable to derive build version; cargo pkgid output was unexpected." >&2
        unset MESH_LLM_BUILD_VERSION || true
        return 0
    fi

    if [[ "$build_profile" == "release" ]]; then
        export MESH_LLM_BUILD_VERSION="$release_version"
        echo "Using release MESH_LLM_BUILD_VERSION: $MESH_LLM_BUILD_VERSION"
        return 0
    fi

    if ! sha="$(git -C "$REPO_ROOT" rev-parse --short=6 HEAD 2>/dev/null)"; then
        echo "Warning: unable to derive build version; git SHA unavailable." >&2
        unset MESH_LLM_BUILD_VERSION || true
        return 0
    fi
    sha="$(printf '%s' "$sha" | tr '[:lower:]' '[:upper:]')"

    if ! status_output="$(git -C "$REPO_ROOT" status --porcelain --untracked-files=all 2>/dev/null)"; then
        echo "Warning: unable to derive build version; git status unavailable." >&2
        unset MESH_LLM_BUILD_VERSION || true
        return 0
    fi
    if [[ -n "$status_output" ]]; then
        dirty_suffix=".dirty"
    fi

    export MESH_LLM_BUILD_VERSION="${release_version}+g${sha}${dirty_suffix}"
    echo "Derived MESH_LLM_BUILD_VERSION: $MESH_LLM_BUILD_VERSION"
}

configure_lld_linker() {
    local lld=""
    local lld_prefix=""

    if (( $+commands[ld64.lld] )); then
        lld="$(command -v ld64.lld)"
    elif (( $+commands[brew] )); then
        lld_prefix="$(brew --prefix lld 2>/dev/null || true)"
        if [[ -n "$lld_prefix" && -x "$lld_prefix/bin/ld64.lld" ]]; then
            lld="$lld_prefix/bin/ld64.lld"
        fi
    fi
    if [[ -z "$lld" ]]; then
        for candidate in /opt/homebrew/opt/lld/bin/ld64.lld /usr/local/opt/lld/bin/ld64.lld; do
            if [[ -x "$candidate" ]]; then
                lld="$candidate"
                break
            fi
        done
    fi

    if [[ -z "$lld" ]]; then
        cat >&2 <<'EOF'
Error: LLVM ld64.lld was not found.

lld is required for faster Rust builds (measured up to 26% faster locally).

Install lld, then rerun the just command:
  brew install lld

If Homebrew installed lld but it is not on PATH, Mesh-LLM also checks:
  $(brew --prefix lld)/bin/ld64.lld
  /opt/homebrew/opt/lld/bin/ld64.lld
  /usr/local/opt/lld/bin/ld64.lld
EOF
        exit 1
    fi

    append_rustflag "-C link-arg=-fuse-ld=$lld"
    echo "Using Rust linker: $lld"
}

configure_rust_cache() {
    if (( $+commands[sccache] )); then
        rustc_wrapper="$(command -v sccache)"
        echo "Using Rust compiler wrapper: $rustc_wrapper"
    fi
}

sign_with_keychain_identity_if_available() {
    local binary_path="$1"
    local keychain_name=""
    local identities=""
    local identity=""
    local openssl_cfg=""
    local code_sign_key=""
    local code_sign_cert=""
    local code_sign_p12=""
    local generation_failed=0
    local target_identity="${MESH_LLM_CODESIGN_IDENTITY:-$MESH_LLM_LOCAL_CODESIGN_IDENTITY}"

    print_codesign_identity_help() {
        cat >&2 <<'EOF'
No signing identity found in your macOS keychains, or it is not usable for code signing.

To prepare local signing, install a certificates+private-key pair for codesigning:

1) Use Apple's managed cert (recommended if available):
   a) Open Xcode ▸ Settings/Preferences ▸ Accounts ▸ <Apple Account> ▸ Manage Certificates.
   b) Add an "Apple Development" certificate.

2) Or generate local-only self-signed identity automatically by rerunning this build.

3) Pin a specific identity explicitly:

      export MESH_LLM_CODESIGN_IDENTITY="Mesh-LLM Local Codesign"

4) You can verify discoverability with:

      security find-identity -v -p codesigning login.keychain-db

If you see a certificate but no identity in this list, the private key is missing or not trusted for codesign.
EOF
    }

    refresh_identities() {
        security find-identity -v -p codesigning 2>/dev/null \
            | awk 'match($0, /"[^"]+"/) { print substr($0, RSTART + 1, RLENGTH - 2) }'
    }

    refresh_all_identities() {
        security find-identity -p codesigning 2>/dev/null \
            | awk 'match($0, /"[^"]+"/) { print substr($0, RSTART + 1, RLENGTH - 2) }'
    }

    has_named_identity_in_any_state() {
        local name="$1"
        local all_identities=""

        all_identities="$(refresh_all_identities)"
        if [[ -z "$all_identities" ]]; then
            return 1
        fi

        print -r -- "$all_identities" | awk -v name="$name" '
            index($0, name) {
                found = 1
            }
            END {
                exit(found ? 0 : 1)
        }
    '
    }

    sign_binary_with_identity() {
        local identity="$1"
        local binary="$2"
        local hash
        local signable_hashes=""

        if codesign -f --sign "$identity" "$binary"; then
            echo "Code-signed mesh-llm binary with: $identity"
            return 0
        fi

        signable_hashes="$(security find-identity -v -p codesigning 2>/dev/null \
            | awk -v identity="$identity" 'index($0, "\"" identity "\"") { print $2 }')"

        if [[ -z "$signable_hashes" ]]; then
            return 1
        fi

        while IFS= read -r hash; do
            [[ -z "$hash" ]] && continue
            if codesign -f --sign "$hash" "$binary"; then
                echo "Code-signed mesh-llm binary with: $identity (${hash})"
                return 0
            fi
        done <<< "$signable_hashes"

        return 1
    }

    trust_existing_codesign_certs() {
        local identity_name="$1"
        local keychain="$2"
        local certs_file=""
        local add_trusted_output=""

        if [[ -z "$identity_name" || -z "$keychain" ]]; then
            return 1
        fi

        certs_file="$(mktemp)"
        add_trusted_output="$(mktemp)"

        if ! security find-certificate -a -p -c "$identity_name" "$keychain" >"$certs_file" 2>/dev/null; then
            rm -f "$certs_file" "$add_trusted_output"
            return 1
        fi

        if [[ ! -s "$certs_file" ]]; then
            rm -f "$certs_file" "$add_trusted_output"
            return 1
        fi

        if ! security add-trusted-cert -d -r trustRoot -p codeSign -k "$keychain" "$certs_file" >"$add_trusted_output" 2>&1; then
            echo "Existing identity '$identity_name' was found but could not be marked trusted for code signing." >&2
            cat "$add_trusted_output" >&2
            rm -f "$certs_file" "$add_trusted_output"
            return 1
        fi

        rm -f "$certs_file" "$add_trusted_output"
        return 0
    }

    has_named_identity() {
        local name="$1"
        local all_identities=""

        all_identities="$(refresh_identities)"
        if [[ -z "$all_identities" ]]; then
            return 1
        fi

        print -r -- "$all_identities" | awk -v name="$name" '
            index($0, name) {
                found = 1
            }
            END {
                exit(found ? 0 : 1)
            }
        '
    }

    remove_existing_identity_and_certs() {
        local identity_name="$1"

        if [[ -z "$identity_name" ]]; then
            return 0
        fi

        security delete-identity -c "$identity_name" "$keychain_name" >/dev/null 2>&1 || true
        security delete-certificate -c "$identity_name" "$keychain_name" >/dev/null 2>&1 || true
    }

    generate_dev_codesign_identity() {
        local identity_name="$1"
        local login_keychain="$2"
        local tmpdir
        local pfx_password
        local cmd_output
        local trust_output

        if ! command -v openssl >/dev/null 2>&1; then
            echo "Automatic identity creation requires openssl (not found)." >&2
            return 1
        fi

        if [[ -z "$identity_name" ]]; then
            echo "Automatic identity creation requires a non-empty name." >&2
            return 1
        fi

        tmpdir="$(mktemp -d)"
        remove_existing_identity_and_certs "$identity_name"
        openssl_cfg="$tmpdir/mesh-llm-dev-codesign.cnf"
        code_sign_key="$tmpdir/mesh-llm-dev-codesign.key"
        code_sign_cert="$tmpdir/mesh-llm-dev-codesign.crt"
        code_sign_p12="$tmpdir/mesh-llm-dev-codesign.p12"
        pfx_password="$(openssl rand -hex 24)"
        cmd_output="$(mktemp)"

        cat > "$openssl_cfg" <<EOF
[ req ]
default_bits = 2048
default_md = sha256
prompt = no
distinguished_name = dn
x509_extensions = v3_ext

[ dn ]
CN = $identity_name

[ v3_ext ]
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = 1.3.6.1.5.5.7.3.3
EOF

        if ! openssl req -x509 -newkey rsa:2048 -nodes \
            -days 3650 \
            -keyout "$code_sign_key" \
            -out "$code_sign_cert" \
            -config "$openssl_cfg" \
            >"$cmd_output" 2>&1; then
            echo "Failed to generate local signing certificate." >&2
            cat "$cmd_output" >&2
            rm -rf "$tmpdir"
            rm -f "$cmd_output"
            return 1
        fi

        if ! openssl pkcs12 -export \
            -out "$code_sign_p12" \
            -inkey "$code_sign_key" \
            -in "$code_sign_cert" \
            -name "$identity_name" \
            -passout "pass:$pfx_password" \
            -legacy \
            >"$cmd_output" 2>&1; then
            echo "Failed to package local signing certificate into PKCS#12." >&2
            cat "$cmd_output" >&2
            rm -rf "$tmpdir"
            rm -f "$cmd_output"
            return 1
        fi
        rm -f "$cmd_output"

        security unlock-keychain "$login_keychain" >/dev/null 2>&1 || true
        if ! security import "$code_sign_p12" -k "$login_keychain" -P "$pfx_password" -f pkcs12 -A -T /usr/bin/codesign -T /usr/bin/security >"$cmd_output" 2>&1; then
            echo "Failed to import generated signing identity into keychain: $login_keychain" >&2
            echo "Try unlocking your login keychain and re-running build." >&2
            cat "$cmd_output" >&2
            rm -rf "$tmpdir"
            rm -f "$cmd_output"
            return 1
        fi
        rm -f "$cmd_output"

        trust_output="$(mktemp)"
        if ! security add-trusted-cert -d -r trustRoot -p codeSign -k "$login_keychain" "$code_sign_cert" >"$trust_output" 2>&1; then
            echo "Generated certificate was imported but could not be trusted for code signing." >&2
            echo "Build will continue, but automatic signing may still fail." >&2
            cat "$trust_output" >&2
            rm -f "$trust_output"
            rm -rf "$tmpdir"
            rm -f "$cmd_output"
            return 1
        fi
        rm -f "$trust_output"

        if ! has_named_identity "$identity_name"; then
            echo "Generated identity '${identity_name}' was not detected by security find-identity." >&2
            echo "Try this command to confirm what is currently in keychain:" >&2
            echo "  security find-identity -v -p codesigning \"$login_keychain\"" >&2
            rm -rf "$tmpdir"
            rm -f "$cmd_output"
            return 1
        fi

        rm -rf "$tmpdir"
        rm -f "$cmd_output"
        return 0
    }

    if [[ ! -f "$binary_path" ]]; then
        echo "Skipping auto-sign: binary not found at $binary_path"
        return 0
    fi

    if ! command -v security >/dev/null 2>&1 || ! command -v codesign >/dev/null 2>&1; then
        echo "Skipping auto-sign: codesign/security command missing."
        return 0
    fi

    identities="$(refresh_identities)"
    keychain_name="$(security list-keychains -d user 2>/dev/null | awk -F'\"' '/login\\.keychain/ { print $2; exit }')"
    if [[ -z "$keychain_name" ]]; then
        keychain_name="$(security default-keychain -d user 2>/dev/null | awk -F'\"' 'NF >= 2 { print $2; exit }')"
    fi
    [[ -z "$keychain_name" ]] && keychain_name="${HOME}/Library/Keychains/login.keychain-db"

    if ! has_named_identity "$target_identity"; then
        if has_named_identity_in_any_state "$target_identity"; then
            echo "Found existing local cert for '${target_identity}' but it is not currently trusted for code signing. Attempting repair..."
            if trust_existing_codesign_certs "$target_identity" "$keychain_name"; then
                if has_named_identity "$target_identity"; then
                    echo "Repaired existing codesign cert trust for '${target_identity}'."
                fi
            fi
        fi
        identities="$(refresh_identities)"
    fi

    if ! has_named_identity "$target_identity" && [[ "${MESH_LLM_AUTO_GENERATE_CODESIGN}" == "1" ]]; then
        echo "No signing identity found. Creating temporary local dev identity: ${target_identity}"
        if generate_dev_codesign_identity "$target_identity" "$keychain_name"; then
            identities="$(refresh_identities)"
        else
            generation_failed=1
        fi
    fi

    if (( generation_failed == 0 )) && ! has_named_identity "$target_identity"; then
        echo "Attempting to sign with generated identity directly: ${target_identity}"
        if sign_binary_with_identity "$target_identity" "$binary_path"; then
            return 0
        fi
        echo "Direct sign with generated identity failed." >&2
        if ! has_named_identity "$target_identity"; then
            echo "The generated identity '${target_identity}' is still not discoverable for signing." >&2
        fi
    fi

    if [[ -z "$identities" ]]; then
        if (( generation_failed == 0 )); then
            print_codesign_identity_help
        else
            echo "Automatic identity creation failed while trying to generate a local signing identity." >&2
            print_codesign_identity_help
            echo
            echo "If you want to skip the auto-generation path, set:" >&2
            echo "  export MESH_LLM_AUTO_GENERATE_CODESIGN=0" >&2
            echo "to follow the manual setup instructions above instead." >&2
        fi
        return 0
    fi

    if [[ -n "${MESH_LLM_CODESIGN_IDENTITY:-}" ]]; then
        echo "Using explicit signing identity: ${MESH_LLM_CODESIGN_IDENTITY}"
        if sign_binary_with_identity "${MESH_LLM_CODESIGN_IDENTITY}" "$binary_path"; then
            return 0
        fi
        echo "Explicit signing identity failed: ${MESH_LLM_CODESIGN_IDENTITY}" >&2
        echo "Falling back to the first available identity..."
    fi

    echo "Attempting to sign local mesh-llm binary with an available keychain identity..."
    for identity in ${(f)identities}; do
        [[ -z "$identity" ]] && continue
        if sign_binary_with_identity "$identity" "$binary_path"; then
            return 0
        fi
        echo "Skipping unusable signing identity: $identity" >&2
    done

    echo "Unable to sign mesh-llm with any discovered keychain identity. Check keychain access for Mesh LLM signing." >&2
    echo "Try this to verify identities and retry:" >&2
    echo "  security find-identity -v -p codesigning" >&2
}

export LLAMA_STAGE_BUILD_DIR="${LLAMA_STAGE_BUILD_DIR:-${SKIPPY_LLAMA_BUILD_DIR:-$LLAMA_BUILD_ROOT/build-stage-abi-metal}}"

configure_lld_linker

echo "Preparing patched llama.cpp ABI checkout..."
LLAMA_WORKDIR="$LLAMA_DIR" "$SCRIPT_DIR/prepare-llama.sh" "${MESH_LLM_LLAMA_PIN_SHA:-pinned}"

echo "Building patched llama.cpp ABI (metal)..."
LLAMA_WORKDIR="$LLAMA_DIR" \
    LLAMA_BUILD_DIR="$LLAMA_STAGE_BUILD_DIR" \
    LLAMA_STAGE_BACKEND="${LLAMA_STAGE_BACKEND:-metal}" \
    "$SCRIPT_DIR/build-llama.sh"

if [[ -d "$MESH_DIR" ]]; then
    if [[ -d "$UI_DIR" ]]; then
        MESH_LLM_BUILD_PROFILE="$build_profile" "$SCRIPT_DIR/build-ui.sh" "$UI_DIR"
    fi

    configure_rust_cache
    mesh_binary=""
    case "$build_profile" in
        dev|debug)
            echo "Building mesh-llm (profile: dev, bin only)..."
            stamp_build_version
            if [[ -n "$rustc_wrapper" ]]; then
                (cd "$REPO_ROOT" && RUSTC_WRAPPER="$rustc_wrapper" cargo build -p mesh-llm --bin mesh-llm)
            else
                (cd "$REPO_ROOT" && cargo build -p mesh-llm --bin mesh-llm)
            fi
            mesh_binary="target/debug/mesh-llm"
            echo "Mesh binary: $mesh_binary"
            ;;
        release)
            echo "Building mesh-llm (profile: release)..."
            stamp_build_version
            if [[ -n "$rustc_wrapper" ]]; then
                (cd "$REPO_ROOT" && RUSTC_WRAPPER="$rustc_wrapper" cargo build --release -p mesh-llm)
            else
                (cd "$REPO_ROOT" && cargo build --release -p mesh-llm)
            fi
            mesh_binary="target/release/mesh-llm"
            echo "Mesh binary: $mesh_binary"
            ;;
        *)
            echo "Unsupported MESH_LLM_BUILD_PROFILE '$build_profile'. Expected debug, dev, or release." >&2
            exit 1
            ;;
    esac

    sign_with_keychain_identity_if_available "$REPO_ROOT/$mesh_binary"
fi
