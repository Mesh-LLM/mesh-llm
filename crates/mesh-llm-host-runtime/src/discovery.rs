use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MeshDiscoveryMode {
    #[default]
    Nostr,
    Mdns,
}

impl MeshDiscoveryMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nostr => "nostr",
            Self::Mdns => "mdns",
        }
    }

    pub const fn source(self) -> &'static str {
        match self {
            Self::Nostr => "nostr-relay",
            Self::Mdns => "mdns-sd",
        }
    }

    pub const fn scope(self) -> DiscoveryScope {
        match self {
            Self::Nostr => DiscoveryScope::Public,
            Self::Mdns => DiscoveryScope::Lan,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryScope {
    Public,
    Lan,
}

impl DiscoveryScope {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Lan => "lan",
        }
    }
}
