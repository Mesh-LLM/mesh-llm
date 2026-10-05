# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities through GitHub's private vulnerability
reporting: **Security and quality tab → Report a vulnerability**, or
directly at https://github.com/Mesh-LLM/mesh-llm/security/advisories/new.

Please do not open public issues for security vulnerabilities, and do not
disclose details publicly until the maintainers have investigated and released
a fix.

## What to Include

- A description of the issue and its potential impact
- Steps to reproduce, or a proof of concept
- The affected version(s) or commit(s)
- Any suggested mitigation, if you have one

## Scope

This policy covers the mesh-llm node software, its mesh protocol
implementation, and components bundled in its releases. Vulnerabilities in
third-party dependencies should be reported to the upstream project.

mesh-llm nodes can join a live public mesh. Please do not test potential
vulnerabilities against the public mesh or other people's nodes. Use a local
or private test mesh instead.

## What to Expect

Reports are handled through GitHub's private advisory thread, where the
maintainers can investigate, discuss, and coordinate a fix with you directly.
Maintainers may credit reporters in the release notes when a fix ships. Let
them know if you would prefer to remain anonymous.

## Thanks

Security reports directly improve the safety of every operator running a
mesh-llm node. We appreciate the time and care that goes into them.
