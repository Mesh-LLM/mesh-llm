# OSS Scanner enrolment

mesh-llm is enrolled in Anthropic's [OSS Scanner](https://red.anthropic.com/oss-scanner),
which builds the project in an isolated VM, cuts its network, and then audits it for
vulnerabilities. Findings are emailed privately to the primary contact with a reproducer
and a proposed patch.

- `Dockerfile` — how the scanner builds mesh-llm. Read the header before editing: the
  build is the only phase with network access, and it has a 45 minute budget.
- `threat_model.md` — what we care about, where untrusted input enters, and how we
  rate severity. Keep this current; it is the main lever on report quality.

The enrolment itself (repo URL, contact address) lives in `anthropics/oss-scanner`
under `projects/mesh-llm/project.yaml`, which points back at these two files. That
means changes here take effect without a pull request against their repository.

To validate changes locally, from a checkout of `anthropics/oss-scanner`:

```sh
tools/validate.py          # checks the enrolment rules
tools/check mesh-llm       # builds exactly as the scanner does, then drops you
                           # into the finished image with no network
```

Note that `tools/check` clones over https from the enrolled branch, so changes to this
directory must be pushed before they can be tested.
