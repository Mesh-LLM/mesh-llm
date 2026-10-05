# macOS Code Signing

> macOS-only. Code signing does not apply to Linux or Windows builds.

The unified source build produces a backend-neutral dynamic host plus an
adjacent native runtime. `scripts/build-mac.sh` is only a compatibility wrapper
for that product build; it does not select a keychain identity, generate a
certificate, or sign the host.

Release signing and notarization are distribution responsibilities. Keep them
in the release/publishing pipeline so a locally rebuilt host cannot silently
look like a published artifact.

For a locally built product that must be opened on another Mac, sign the host
explicitly after building it, using an identity you selected deliberately:

```bash
codesign --force --sign "Developer ID Application: Example" target/debug/mesh-llm
codesign --verify --verbose=2 target/debug/mesh-llm
```

Signing changes the host bytes. Do not sign or otherwise mutate a host that has
already been release-attested: product composition verifies and copies the
immutable producer host exactly. Instead, obtain the published signed product.
