## Title

Use an accurate title that describes the user-visible change or fix.

## Original problem

What was broken, missing, confusing, or risky before this PR?

## Diagnostics

How did you verify the problem? Include commands, logs, screenshots, or links when useful.

## Fix

How does this PR fix the problem? Call out compatibility, protocol, or migration concerns if any.

## Validation

- [ ] I ran the relevant local checks, or explained why they do not apply.
- [ ] UI changes include screenshots or video, or explain why visual aids are not needed.
- [ ] For Skippy scheduler, cache, prefix, or restore changes: I reviewed new `clone`, `to_vec`, and `to_owned` calls for payload-sized copies, especially under locks or before a field is cleared or overwritten. I described any necessary deep copy and its measured cost, or marked this check not applicable.
