/// Complete local capacity for one exact serving ID. This does not reserve lanes.
public struct LaneCapacity: Sendable, Equatable {
    public let executionLanes: UInt64
    /// Smallest lane context, safe for any lane.
    public let contextTokensPerLane: UInt64
    /// Sum of lane contexts, not a per-request limit.
    public let contextPoolTokens: UInt64
    /// Oldest observation; the caller chooses its freshness policy.
    public let capturedAtUnixMs: UInt64

    internal init(_ value: RuntimeCapacityNative) {
        executionLanes = value.executionLanes
        contextTokensPerLane = value.contextTokensPerLane
        contextPoolTokens = value.contextPoolTokens
        capturedAtUnixMs = value.capturedAtUnixMs
    }
}
