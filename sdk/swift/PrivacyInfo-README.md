# Privacy Manifest for MeshLLM XCFramework

## What this file is

Apple requires all third-party SDKs to include a `PrivacyInfo.xcprivacy` manifest file since Spring 2024. This file declares the privacy and data handling practices of the SDK to App Store Connect and to host applications that integrate the framework.

The `PrivacyInfo.xcprivacy` file is an Apple property list (plist) in XML format that declares:
- Whether the SDK performs user tracking
- What types of data are collected
- Which privacy-sensitive APIs are accessed
- Which domains are used for tracking

## XCFramework embedding requirement

**CRITICAL**: The `PrivacyInfo.xcprivacy` file MUST be embedded **inside each `.framework` bundle** within the XCFramework, not just placed in the host application.

Apple's preferred path for SDKs that need resources is an Xcode-built static
framework target archived with `xcodebuild archive`, then assembled with
`xcodebuild -create-xcframework`. MeshLLM's FFI binary starts as a Rust static
library, so the current scripts wrap that archive in a static framework bundle
before handing the framework to `xcodebuild -create-xcframework`.

Do not switch this package to `xcodebuild -create-xcframework -library
-headers` unless the privacy manifest is carried by another App Store-visible
SDK target. Plain static-library XCFramework slices do not have a framework
resource bundle for `PrivacyInfo.xcprivacy`.

The directory structure must be:
```
MeshLLM.xcframework/
├── ios-arm64/
│   └── MeshLLM.framework/
│       ├── MeshLLM (binary)
│       ├── Modules/
│       └── PrivacyInfo.xcprivacy  ← MUST be here
├── ios-arm64-simulator/
│   └── MeshLLM.framework/
│       ├── MeshLLM (binary)
│       ├── Modules/
│       └── PrivacyInfo.xcprivacy  ← MUST be here
└── macos-arm64/
    └── MeshLLM.framework/
        ├── MeshLLM (binary)
        ├── Modules/
        └── PrivacyInfo.xcprivacy  ← MUST be here
```

The Swift SDK build scripts are responsible for copying this template into each
`.framework` bundle before XCFramework construction:

- `sdk/swift/scripts/build-xcframework.sh`
- `sdk/swift/scripts/build-host-macos-xcframework.sh`

### Verification

To verify that `PrivacyInfo.xcprivacy` files are correctly embedded in the built XCFramework:

```bash
find sdk/swift/Generated/MeshLLMFFI.xcframework -name PrivacyInfo.xcprivacy | wc -l
```

This command should return a count ≥ 1 (ideally 3 or more, one per platform slice).

## Declarations

This manifest declares the following privacy practices for MeshLLM:

### NSPrivacyTracking
**Value**: `false`

MeshLLM does not perform user tracking. The SDK does not collect identifiers for cross-app or cross-site tracking purposes.

### NSPrivacyCollectedDataTypes
**Value**: Empty array `[]`

MeshLLM does not collect any user data. The SDK operates as a distributed inference client that communicates with mesh peers via POSIX sockets and QUIC protocol. No personal data, device identifiers, or usage analytics are collected.

### NSPrivacyAccessedAPITypes
**Value**: Empty array `[]`

MeshLLM does not access any privacy-sensitive APIs that require declaration. The SDK uses only:
- POSIX socket APIs (standard C library)
- QUIC protocol via the `iroh` crate (transport layer)
- No NSURLSession, NWPathMonitor, or other Apple privacy-sensitive APIs

### NSPrivacyTrackingDomains
**Value**: Empty array `[]`

MeshLLM does not communicate with any tracking domains. All network communication is peer-to-peer via QUIC to mesh nodes, not to centralized tracking or analytics services.

## Implementation notes

- This file is a template and should be copied into each `.framework` bundle during XCFramework construction
- The plist format is XML (not binary) for readability and version control
- No modifications to this file are needed unless MeshLLM's privacy practices change
- If new privacy-sensitive APIs are added to the Rust core, this manifest must be updated accordingly
