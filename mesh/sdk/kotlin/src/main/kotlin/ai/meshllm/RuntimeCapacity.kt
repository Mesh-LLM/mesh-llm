package ai.meshllm

/** Complete local capacity for one exact serving ID. This does not reserve lanes. */
data class RuntimeCapacity(
    val executionLanes: ULong,
    /** Smallest lane context, safe for any lane. */
    val contextTokensPerLane: ULong,
    /** Sum of lane contexts, not a per-request limit. */
    val contextPoolTokens: ULong,
    /** Oldest observation; the caller chooses its freshness policy. */
    val capturedAtUnixMs: ULong,
)
