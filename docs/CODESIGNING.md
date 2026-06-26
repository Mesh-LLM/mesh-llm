# macOS Code Signing

> macOS-only. Code signing does not apply to Linux or Windows builds.

The `scripts/build-mac.sh` script automatically signs the compiled `mesh-llm`
binary with a code signing identity from the macOS keychain when one is
available. This prevents macOS Gatekeeper from quarantining or rejecting locally
built development binaries.

## Environment variables

All codesigning behavior is controlled via environment variables. None of these
are CLI flags.

| Variable | Default | Purpose |
|---|---|---|
| `MESH_LLM_CODESIGN_IDENTITY` | *(unset, falls back to `MESH_LLM_LOCAL_CODESIGN_IDENTITY`)* | Explicitly pins the signing identity by common name. Takes priority over auto-detection. |
| `MESH_LLM_LOCAL_CODESIGN_IDENTITY` | `Mesh-LLM Local Codesign` | The identity name to use when `MESH_LLM_CODESIGN_IDENTITY` is not set. Also used as the common name for auto-generated self-signed certificates. |
| `MESH_LLM_AUTO_GENERATE_CODESIGN` | `1` | When `1`, the script auto-generates a self-signed codesigning identity (2048-bit RSA, 10-year validity) if no matching identity is found. Set to `0` to disable and follow manual setup. |

## Priority order

The `sign_with_keychain_identity_if_available` function (called at the end of
`build-mac.sh`) resolves the signing identity in this order:

1. **Explicit identity** — If `MESH_LLM_CODESIGN_IDENTITY` is set, attempt to sign
   with that exact common name. If it fails, fall through.
2. **Auto-generate** — If `MESH_LLM_AUTO_GENERATE_CODESIGN=1` (default) and no
   matching identity exists, generate a self-signed certificate, import it into
   the login keychain, mark it trusted for `codeSign`, and sign with it.
3. **First available** — Iterate over all valid codesigning identities in the
   keychain and try each one until signing succeeds.
4. **Skip** — If no identity is usable and auto-generation is off or failed,
   print setup instructions and leave the binary unsigned.

## Repair flow

If a certificate with the target name exists but is not trusted for code
signing (e.g., from a previous incomplete setup), the script attempts to
repair it via `security add-trusted-cert` before falling through to
auto-generation or the fallback path.

## Auto-generation details

When `MESH_LLM_AUTO_GENERATE_CODESIGN=1` and no suitable identity is found:

1. Creates a temporary directory with OpenSSL config, key, and certificate.
2. Generates a self-signed X.509 certificate with:
   - 2048-bit RSA key, SHA-256 signature
   - `digitalSignature` + `keyEncipherment` key usage
   - `codeSigning` extended key usage (OID `1.3.6.1.5.5.7.3.3`)
   - 10-year validity
3. Packages into a PKCS#12 bundle and imports into the login keychain with
   `/usr/bin/codesign` and `/usr/bin/security` access control.
4. Marks the certificate as trusted for code signing.
5. Verifies the identity is discoverable via `security find-identity`.

## The actual `codesign` invocation

The script calls `codesign` with minimal flags:

```bash
codesign -f --sign "<identity>" <binary>
```

- **`-f`** — force sign (replaces any existing signature).
- **`--sign <identity>`** — the identity to sign with.

Notable flags **not** passed:

| Flag | Status | Implication |
|---|---|---|
| `--options=hardened_runtime` | Not used | Hardened Runtime is not enabled. The binary will pass local Gatekeeper checks but would fail Apple notarization. |
| `--timestamp` | Not used | No signing timestamp is embedded. The signature will not be verifiable after the certificate expires. |
| `--entitlements` | Not used | No entitlements plist is applied. |
| `--deep` | Not used | Deep signing is not used (not applicable for a single Mach-O binary). |

This is a **local development signing** strategy — it satisfies macOS Gatekeeper
on the local machine (prevents "damaged file" / "quarantine" popups for locally
built binaries) but will **not** pass Apple's notarization for distribution
outside the local machine. Release builds distributed via GitHub Releases use
Apple Developer ID signing in the CI pipeline, not this script.

## Manual setup

If auto-generation is disabled or fails, set up a signing identity manually.

### Option 1: Apple Developer certificate (recommended if available)

1. Open Xcode → Settings/Preferences → Accounts → select your Apple Account →
   Manage Certificates.
2. Click **+** → **Apple Development**.
3. The certificate appears in your login keychain automatically.

### Option 2: Manual self-signed identity

If you do not have an Apple Developer account, generate a self-signed
certificate with the same parameters the auto-generation path uses:

```bash
IDENTITY="Mesh-LLM Local Codesign"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

# Generate key and self-signed cert
openssl req -x509 -newkey rsa:2048 -nodes \
  -days 3650 \
  -keyout /tmp/mesh-dev.key \
  -out /tmp/mesh-dev.crt \
  -subj "/CN=$IDENTITY" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=codeSigning"

# Package as PKCS#12
openssl pkcs12 -export \
  -out /tmp/mesh-dev.p12 \
  -inkey /tmp/mesh-dev.key \
  -in /tmp/mesh-dev.crt \
  -name "$IDENTITY" \
  -passout pass:password \
  -legacy

# Import into login keychain
security import /tmp/mesh-dev.p12 \
  -k "$KEYCHAIN" -P password -f pkcs12 \
  -A -T /usr/bin/codesign -T /usr/bin/security

# Mark as trusted for code signing
security add-trusted-cert -d -r trustRoot -p codeSign -k "$KEYCHAIN" /tmp/mesh-dev.crt

# Cleanup
rm -f /tmp/mesh-dev.key /tmp/mesh-dev.crt /tmp/mesh-dev.p12
```

### Option 3: Pin a specific identity

```bash
export MESH_LLM_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
```

## Troubleshooting

### Check which identities are available

```bash
# Only valid codesigning identities
security find-identity -v -p codesigning

# All identities (including untrusted)
security find-identity -p codesigning
```

### "No signing identity found"

If the build prints the help message about missing identities:

1. Run `security find-identity -v -p codesigning` to confirm no identity is
   available.
2. Check whether `MESH_LLM_AUTO_GENERATE_CODESIGN` is set to `0` — set it to
   `1` (or unset it) to let the script generate one automatically.
3. If auto-generation is running but failing, check that `openssl` is
   installed on your system.

### "Certificate not trusted"

If a certificate exists in the keychain but is not trusted for code signing,
the script attempts automatic repair. To repair manually:

```bash
security find-certificate -a -p -c "Mesh-LLM Local Codesign" login.keychain-db \
  | security add-trusted-cert -d -r trustRoot -p codeSign -k login.keychain-db
```

### Binary still quarantined after signing

If macOS still shows a quarantine warning:

```bash
# Remove quarantine attribute
xattr -dr com.apple.quarantine ./target/debug/mesh-llm

# Verify the signature
codesign -dvvv ./target/debug/mesh-llm
```

Expected output includes `Authority=...` and `Signature=...`. A missing
`sealed resources version 2` is normal for a single Mach-O binary without
hardened runtime.
