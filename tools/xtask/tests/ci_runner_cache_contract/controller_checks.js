// Finite recorder only; validation is performed solely by the unchanged scalar.
'use strict';
globalThis.fetch = async () => {throw new Error('network disabled in finite controller fixture');};
globalThis.WebSocket = class {constructor() {throw new Error('network disabled in finite controller fixture');}};
const fs = require('fs');
const crypto = require('crypto');
const config = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const source = '0123456789abcdef0123456789abcdef01234567';
const correlation = `ci-91-${source.slice(0, 12)}`;
const plan = config.canonical ? JSON.parse(config.canonical) : {schema_version: 1, source_sha: source, profile: 'manual-full', fixture_capability: {cpu: true}};
if (config.mutation === 'source') plan.source_sha = 'f'.repeat(40);
if (config.mutation === 'schema') plan.schema_version = 2;
const canonical = config.canonical || JSON.stringify(plan);
const digest = config.digest || crypto.createHash('sha256').update(canonical).digest('hex');
const projections = config.lanes;
if (config.mutation === 'lane') projections[2].lane = 'unowned';
for (const projection of projections) process.env[projection.lane === 'unowned' ? 'LINUX_LANE_PLAN' : `${projection.lane.toUpperCase()}_LANE_PLAN`] = JSON.stringify(projection);
const creates = [], dispatches = [], updates = [];
const resolved = {};
const copy = value => JSON.parse(JSON.stringify(value));
const github = {rest: {
  repos: {getCommit: async request => {
    if (request.ref !== source) throw new Error('unexpected resolver source');
    return {data: {parents: [{sha: 'a'.repeat(40)}]}};
  }},
  checks: {
    create: async request => { const id = 701 + creates.length; creates.push({id, request: copy(request)}); return {data: {id}}; },
    update: async request => { updates.push(copy(request)); return {data: {}}; },
  },
  actions: {createWorkflowDispatch: async request => {
    dispatches.push(copy(request));
    if (request.workflow_id === config.fail_lane) throw new Error('finite dispatch failure');
    return {status: 204};
  }},
}};
const context = {repo: {owner: 'fixture-owner', repo: 'fixture-repository'}, serverUrl: 'https://fixture.invalid', runId: 91, sha: source, payload: {inputs: config.manual_input === null ? {} : {use_depot: config.manual_input}}};
const core = {info: () => {}, warning: () => {}, setOutput: (name, value) => {resolved[name] = value;}, setFailed: message => {throw new Error(message)}};
const closedRequire = name => {if (name !== 'crypto') throw new Error(`unowned module ${name}`); return crypto;};
(async () => {
  let error = null;
  try {
    const resolve = fs.readFileSync(process.argv[4], 'utf8');
    const executeResolve = new Function('github', 'context', 'core', `return (async () => {${resolve}\n})()`);
    await executeResolve(github, context, core);
    Object.assign(process.env, {
      SOURCE_SHA: resolved.source_sha, DEFAULT_BRANCH: 'protected-main', ORIGINAL_EVENT_NAME: resolved.original_event_name,
      CORRELATION_ID: resolved.correlation_id, SUPERSESSION_KEY: resolved.supersession_key, USE_DEPOT: resolved.use_depot,
      PLAN_JSON: canonical, PLAN_DIGEST: config.mutation === 'digest' ? '0'.repeat(64) : digest,
    });
    const actual = fs.readFileSync(process.argv[2], 'utf8');
    const execute = new Function('github', 'context', 'core', 'require', `return (async () => {${actual}\n})()`);
    await execute(github, context, core, closedRequire);
  } catch (failure) { error = String(failure); }
  console.log(JSON.stringify({creates, dispatches, updates, error, source, correlation, digest, canonical, projections, resolved}));
})().catch(error => {console.error(error); process.exitCode = 1;});
