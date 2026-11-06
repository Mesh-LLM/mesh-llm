# Maintainer decisions

## 2026-11-06: retain the external AgentX client

James directed: “we must make allowances for agentx.” This answered the proposal
to admit PR #2746's existing pinned upstream client and locked environment,
with setup using an installed Python 3.12 interpreter.

Source: [James's decision in Reviews](buzz://message?channel=97e27aa1-1281-4752-91a4-24ea085e1a62&id=31a7946c8680288d723e2831e22218e77d3d62c6732e87b70b24ed8605fd52b8).

The bounded owner is `skippy/evals/agentx/`: its exact upstream AgentX wheel,
dependency manifest, lock and setup invocation are registered in
`ci/python-tooling-policy.json`. Setup does not download an interpreter.
Campaign workers consume the prepared environment offline and verify its bytes.
This decision permits the external client dependency; it does not permit new
repository Python implementations or a general interpreter invocation bypass.
Changing the admitted inputs still requires updating their exact typed policy
and passing the policy's rejection tests and normal repository consistency gate.
