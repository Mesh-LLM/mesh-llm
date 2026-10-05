# mesh-llm-release-footer

Dependency-light encoding, parsing, and verification for the attestation footer
embedded in MeshLLM release binaries.

This crate owns only the stable binary footer format and payload-verifier
contract. Attestation claims, key handling, and release orchestration remain
owned by their callers.
