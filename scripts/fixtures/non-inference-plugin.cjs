#!/usr/bin/env node
'use strict';

// Small protocol-3 conformance provider: current framed protobuf control,
// valid initialization/health/shutdown, no inference endpoint in its manifest.
// Field numbers match crates/mesh-llm-plugin/proto/plugin.proto. This exercises
// the host's real identity and protocol validation, without changing either.
const assert = require('node:assert/strict');
const net = require('node:net');
const name = process.env.MESH_LLM_PLUGIN_NAME;
assert(name && process.env.MESH_LLM_PLUGIN_ENDPOINT);

function varint(value) {
  value = BigInt(value);
  const bytes = [];
  do { bytes.push(Number(value & 127n) | (value > 127n ? 128 : 0)); value >>= 7n; } while (value);
  return Buffer.from(bytes);
}
function uint(field, value) { return Buffer.concat([varint(field << 3), varint(value)]); }
function bytes(field, value) {
  const body = Buffer.isBuffer(value) ? value : Buffer.from(value);
  return Buffer.concat([varint((field << 3) | 2), varint(body.length), body]);
}
function decode(body) {
  let offset = 0;
  const fields = new Map();
  function readVarint() {
    let value = 0n, shift = 0n;
    do {
      assert(offset < body.length && shift < 70n, 'invalid varint');
      const byte = body[offset++];
      value |= BigInt(byte & 127) << shift;
      if (!(byte & 128)) return value;
      shift += 7n;
    } while (true);
  }
  while (offset < body.length) {
    const key = Number(readVarint());
    const field = key >> 3, wire = key & 7;
    if (wire === 0) fields.set(field, readVarint());
    else if (wire === 2) {
      const size = Number(readVarint());
      assert(offset + size <= body.length, 'invalid protobuf length');
      fields.set(field, body.subarray(offset, offset + size)); offset += size;
    } else throw new Error(`unsupported control wire type ${wire}`);
  }
  return fields;
}
const socket = net.connect(process.env.MESH_LLM_PLUGIN_ENDPOINT);
function send(requestId, field, payload) {
  const body = Buffer.concat([uint(1, 3), bytes(2, name), uint(3, requestId), bytes(field, payload)]);
  const size = Buffer.alloc(4); size.writeUInt32LE(body.length);
  socket.write(Buffer.concat([size, body]));
}
let pending = Buffer.alloc(0);
socket.on('data', chunk => {
  pending = Buffer.concat([pending, chunk]);
  while (pending.length >= 4) {
    const size = pending.readUInt32LE();
    assert(size <= 16 * 1024 * 1024, 'oversized plugin frame');
    if (pending.length < size + 4) break;
    const fields = decode(pending.subarray(4, size + 4));
    pending = pending.subarray(size + 4);
    assert.equal(fields.get(1), 3n, 'host must use protocol 3');
    const requestId = fields.get(3) || 0n;
    if (fields.has(10)) {
      assert.equal(decode(fields.get(10)).get(1), 3n, 'initialize must require protocol 3');
      const info = JSON.stringify({ protocolVersion: '2024-11-05', capabilities: {},
        serverInfo: { name, version: '1.0.0' } });
      const manifest = bytes(8, 'fixture:non-inference');
      send(requestId, 11, Buffer.concat([bytes(1, name), uint(2, 3), bytes(3, '1.0.0'),
        bytes(4, info), bytes(5, 'fixture:non-inference'), bytes(6, manifest)]));
      process.stderr.write('fixture protocol-3 non-inference initialize response sent\n');
    } else if (fields.has(12)) {
      send(requestId, 13, Buffer.concat([uint(1, 1), bytes(2, 'non-inference fixture healthy')]));
    } else if (fields.has(14)) {
      send(requestId, 15, Buffer.alloc(0)); socket.end();
    }
  }
});
socket.on('error', error => { console.error(error); process.exitCode = 1; });
