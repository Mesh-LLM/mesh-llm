# Replay export captures

`legacy.json` is a fresh capture of the unchanged `agentic-replay-params.py`
CLI, made before export implementation. It retains raw input, argv, initial
files, exit status, stdout, stderr and final directory/file bytes for 38 cases.
Default tests read it without an interpreter. The opt-in capture test uses
`create_new` and cannot overwrite an existing receipt.

The requested design session `ses_f1a0acbbbffeBA6kh6eMsmsrpj` was unavailable
through `session_read` and the local session database. The case definitions
were reconstructed from the source; only the requested ID ranges and count
are retained. This is not a claim to reproduce the missing design roster.

The unchanged historical 31-case validator receipt remains in
`../optional_replay/legacy-print-shell.json`.

JSON bytes, file effects and stdout are compared exactly except for the two
explicit surrogate blockers G01 and G02. R10-R12, F01-F06 and C01-C02 preserve
status/file effects but use typed Rust diagnostics instead of Python traceback
or argparse bytes. These eleven cases are not exact diagnostic matches.
