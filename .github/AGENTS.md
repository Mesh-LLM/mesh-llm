# CI

The workflows are plain. Keep them that way.

- `ci.yml` runs on every pull request and on main: lint, Rust tests on Linux,
  macOS and Windows, console and public website checks, and one real inference smoke on
  Skippy's CPU runtime.
- `release.yml` builds, packages, signs and publishes a release.
- Each job calls a `just` recipe or a script under `scripts/`. Put logic there,
  where it can be run locally, not in workflow YAML.
- Skippy is built, tested and released in its own repository. Do not add Skippy
  builds, GPU runtimes or Skippy tests here. The smoke downloads the CPU runtime
  from the pinned release; only a named Skippy ref builds one from source.
- Pin every action to a full commit SHA. Run `actionlint` before pushing.
