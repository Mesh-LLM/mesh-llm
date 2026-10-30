//! MCP 2026-07-28 result cache metadata for the plugin MCP list surface.
//!
//! From protocol revision 2026-07-28 every `resultType: "complete"` result must
//! carry `ttlMs` and `cacheScope`. `rmcp`'s `with_all_items` leaves both unset and
//! serializes them only when present, so a list handler that does not opt in emits
//! a result a conforming client has to reject. Peers on an older revision must keep
//! the legacy wire shape, so the fields are attached only when the request itself
//! declares 2026-07-28 or later, read from the same source `rmcp` uses for its own
//! SEP-2322 gate (`RequestContext::protocol_version`).

use rmcp::{
    RoleServer,
    model::{
        CacheScope, ListPromptsResult, ListResourceTemplatesResult, ListResourcesResult,
        ListToolsResult, ProtocolVersion,
    },
    service::RequestContext,
};

/// Aggregated plugin and external MCP lists change whenever a plugin or endpoint
/// starts or stops, and they are namespaced for the connected session. They are
/// therefore never shareable and never fresh: `ttlMs: 0`, `cacheScope: "private"`.
/// This is the same advertisement the server already makes on `server/discover`.
const LIST_TTL_MS: u64 = 0;
const LIST_CACHE_SCOPE: CacheScope = CacheScope::Private;

/// The list results that carry the 2026-07-28 cache metadata.
pub(super) trait ListResultCacheMetadata: Sized {
    fn with_cache_metadata(self) -> Self;
}

impl ListResultCacheMetadata for ListToolsResult {
    fn with_cache_metadata(self) -> Self {
        self.with_ttl_ms(LIST_TTL_MS)
            .with_cache_scope(LIST_CACHE_SCOPE)
    }
}

impl ListResultCacheMetadata for ListPromptsResult {
    fn with_cache_metadata(self) -> Self {
        self.with_ttl_ms(LIST_TTL_MS)
            .with_cache_scope(LIST_CACHE_SCOPE)
    }
}

impl ListResultCacheMetadata for ListResourcesResult {
    fn with_cache_metadata(self) -> Self {
        self.with_ttl_ms(LIST_TTL_MS)
            .with_cache_scope(LIST_CACHE_SCOPE)
    }
}

impl ListResultCacheMetadata for ListResourceTemplatesResult {
    fn with_cache_metadata(self) -> Self {
        self.with_ttl_ms(LIST_TTL_MS)
            .with_cache_scope(LIST_CACHE_SCOPE)
    }
}

/// Attach the required cache metadata to a list result for a 2026-07-28 peer, and
/// leave it untouched for older peers.
pub(super) fn attach_if_supported<T>(result: T, context: &RequestContext<RoleServer>) -> T
where
    T: ListResultCacheMetadata,
{
    if requires_cache_metadata(context) {
        result.with_cache_metadata()
    } else {
        result
    }
}

fn requires_cache_metadata(context: &RequestContext<RoleServer>) -> bool {
    context
        .protocol_version()
        .is_some_and(|version| version.as_str() >= ProtocolVersion::V_2026_07_28.as_str())
}
