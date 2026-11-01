'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const { Memberships } = require('../memberships')

const wireStatus = { name: 'team', api_base_url: 'http://localhost:9338/v1',
  invite_token: 'invite', endpoint_id: 'node-a', peers: 2, join_error: null }

test('membership controls preserve allowlists, status, exact names and replacement results', async () => {
  const memberships = new Memberships({
    async addMembershipJson(source) {
      assert.deepEqual(JSON.parse(source), { name: 'team', join_tokens: ['a'],
        trusted_owners: ['owner'], api_port: 9338, node_key_path: '/keys/team', enumerate_host: false })
      return JSON.stringify(wireStatus)
    },
    async removeMembership(name) { assert.equal(name, 'team'); return false },
    async membershipsJson() { return JSON.stringify([wireStatus]) }
  })
  const added = await memberships.add({ name: 'team', joinTokens: ['a'],
    trustedOwners: ['owner'], apiPort: 9338, nodeKeyPath: '/keys/team' })
  assert.equal(added.apiBaseUrl, wireStatus.api_base_url)
  assert.equal(added.joinError, null)
  assert.deepEqual(await memberships.list(), [added])
  assert.equal(await memberships.remove('team'), false)
})

test('membership native failures propagate without synthesizing success', async () => {
  const error = new Error('invalid membership name')
  const memberships = new Memberships({ async addMembershipJson() { throw error },
    async removeMembership() { throw error }, async membershipsJson() { throw error } })
  await assert.rejects(memberships.add({ name: '?', apiPort: 9338 }), value => value === error)
  await assert.rejects(memberships.remove('?'), value => value === error)
  await assert.rejects(memberships.list(), value => value === error)
})
