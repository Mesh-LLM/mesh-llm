export type MeshAuditPresentation = {
  readonly meaning: string
  readonly title: string
  readonly verdict: string
}

const meshAuditPresentations: Readonly<Record<string, MeshAuditPresentation>> = {
  gossip_direct_peer_promoted: {
    title: 'Direct peer path promoted',
    verdict: 'A direct path to this peer was selected.',
    meaning: 'Mesh traffic can now reach this peer without a relay.'
  },
  gossip_peer_removed: {
    title: 'Peer removed from gossip',
    verdict: 'This peer was removed from the active gossip set.',
    meaning: 'The node will no longer exchange gossip with this peer.'
  },
  gossip_policy_rejected: {
    title: 'Peer rejected by policy',
    verdict: 'This node rejected the peer during its policy check.',
    meaning: 'The peer was not admitted to the mesh.'
  },
  gossip_incompatible_version_rejected: {
    title: 'Peer version rejected',
    verdict: 'The peer advertised a protocol version this node does not support.',
    meaning: 'The peers cannot exchange mesh gossip until their protocol versions are compatible.'
  },
  mesh_quic_inbound_accepted: {
    title: 'Inbound QUIC peer accepted',
    verdict: 'This node accepted an inbound QUIC connection from the peer.',
    meaning: 'The peer can use the negotiated mesh transport.'
  },
  mesh_control_connection_accepted: {
    title: 'Control connection accepted',
    verdict: 'This node accepted the peer control connection.',
    meaning: 'The peer can send negotiated mesh control messages.'
  },
  mesh_control_alpn_rejected: {
    title: 'Control protocol rejected',
    verdict: 'This node rejected the control connection because its protocol was unsupported.',
    meaning: 'No control stream was admitted for this connection.'
  },
  mesh_quic_handler_failed: {
    title: 'QUIC handler failed',
    verdict: 'The QUIC handler stopped before it completed the peer request.',
    meaning: 'The affected peer operation did not complete on this connection.'
  },
  mesh_control_handler_failed: {
    title: 'Control handler failed',
    verdict: 'The control handler stopped before it completed the peer request.',
    meaning: 'The affected control operation did not complete on this connection.'
  }
}

export function humanizeAuditCode(value: string): string {
  return value.replaceAll('_', ' ')
}

export function meshAuditPresentation(code: string): MeshAuditPresentation {
  return (
    meshAuditPresentations[code] ?? {
      title: humanizeAuditCode(code),
      verdict: 'This mesh boundary event was recorded without a dedicated explanation.',
      meaning: 'Use the retained signals and metadata below to correlate the event.'
    }
  )
}
