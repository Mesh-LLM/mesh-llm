'use strict'

function status(value) {
  return {
    name: value.name,
    apiBaseUrl: value.api_base_url,
    inviteToken: value.invite_token,
    endpointId: value.endpoint_id,
    peers: value.peers,
    joinError: value.join_error ?? null
  }
}

class Memberships {
  constructor(handle) { this._handle = handle }

  async add(config) {
    const { joinTokens = [], trustedOwners = [], nodeKeyPath = null, enumerateHost = false,
      apiPort, ...rest } = config
    const wire = { ...rest, join_tokens: joinTokens, trusted_owners: trustedOwners,
      api_port: apiPort, node_key_path: nodeKeyPath, enumerate_host: enumerateHost }
    return status(JSON.parse(await this._handle.addMembershipJson(JSON.stringify(wire))))
  }

  remove(name) { return this._handle.removeMembership(name) }

  async list() {
    return JSON.parse(await this._handle.membershipsJson()).map(status)
  }
}

module.exports = { Memberships }
