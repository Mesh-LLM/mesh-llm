# Mesh native serving plugin host

This crate loads and validates one native serving plugin for Mesh's dedicated
local-model OpenAI surface. Plugin proposal calls run on an isolated worker;
Skippy's decode thread waits only until its own absolute deadline and never
joins or drains plugin work. Lifecycle and authoritative verification outcomes
remain ordered through the same bounded queue.
