#!/usr/bin/env node
'use strict';

// Qualification against a built release host and the real protocol-3 adapter.
// Usage: node scripts/plugin-only-endpoint-smoke.cjs /absolute/mesh-llm [adapter.tar.gz]
// Without an archive, installs the pinned published openai-endpoint 0.2.0.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');
const sourceBinary = path.resolve(process.argv[2] || 'target/release/mesh-llm');
const archive = process.argv[3] && path.resolve(process.argv[3]);
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mesh-plugin-only-'));
const binary = path.join(root, process.platform === 'win32' ? 'mesh-llm.exe' : 'mesh-llm');
const model = 'fixture-endpoint-model';
const reply = 'Endpoint fixture reply';
const children = [];
const requests = [];
let stub;
let unavailableManifestPort;
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

function environment(name) {
  const home = path.join(root, name);
  for (const dir of [home, 'bundle', 'runtime-cache', 'plugins']) {
    fs.mkdirSync(dir === home ? dir : path.join(home, dir), { recursive: true });
  }
  // Only child processes get a temporary home; the operator's environment is unchanged.
  const inherited = Object.fromEntries(Object.entries(process.env)
    .filter(([key]) => !/^MESH(?:_|LLM)|^SKIPPY_/.test(key)));
  return { ...inherited, HOME: home, USERPROFILE: home,
    MESH_LLM_HOME: path.join(home, '.mesh-llm'),
    MESH_LLM_NO_DEFAULT_PLUGINS: '1',
    MESH_LLM_CONFIG: path.join(home, 'config.toml'),
    MESH_LLM_PLUGIN_DIR: path.join(home, 'plugins'),
    MESH_LLM_NATIVE_RUNTIME_CACHE_DIR: path.join(home, 'runtime-cache'),
    MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL: `http://127.0.0.1:${unavailableManifestPort}/unavailable-manifest`,
    DO_NOT_TRACK: '1' };
}

function launch(name, args, env) {
  const log = fs.openSync(path.join(root, `${name}.log`), 'w');
  const child = spawn(binary, ['--log-format', 'json', ...args], {
    env, stdio: ['ignore', log, log], detached: process.platform !== 'win32',
  });
  fs.closeSync(log);
  child.once('error', error => { child.launchError = error; });
  children.push(child);
  return child;
}

async function command(name, args, env) {
  const child = launch(name, args, env);
  const code = await new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', resolve);
    const timer = setTimeout(() => { child.kill(); reject(new Error(`${name} timed out`)); }, 120000);
    timer.unref();
    child.once('exit', () => clearTimeout(timer));
  });
  assert.equal(code, 0, `${name} failed; see ${root}/${name}.log`);
}

async function port() {
  const server = net.createServer();
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const value = server.address().port;
  await new Promise(resolve => server.close(resolve));
  return value;
}

async function json(url, options) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(5000) });
  assert.equal(response.status, 200, `${url}: ${await response.clone().text()}`);
  return response.json();
}

async function waitFor(child, description, probe) {
  const deadline = Date.now() + 90000;
  let last;
  while (Date.now() < deadline) {
    if (child.launchError) throw child.launchError;
    assert.equal(child.exitCode, null, `${description}: process exited; logs: ${root}`);
    try { const value = await probe(); if (value) return value; } catch (error) { last = error; }
    await pause(500);
  }
  throw new Error(`${description} timed out: ${last || 'not ready'}; logs: ${root}`);
}

async function inference(base, stream) {
  const response = await fetch(`${base}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages: [{ role: 'user', content: 'Fixture probe' }],
      max_tokens: 16, stream }), signal: AbortSignal.timeout(15000),
  });
  assert.equal(response.status, 200, await response.clone().text());
  if (!stream) {
    const body = await response.json();
    assert.equal(body.model, model);
    assert.equal(body.choices[0].message.content, reply);
  } else {
    const text = await response.text();
    assert.match(text, /data: \[DONE\]/);
    const chunks = text.split(/\r?\n/).filter(line => line.startsWith('data: ') && !line.includes('[DONE]'))
      .map(line => JSON.parse(line.slice(6)));
    assert(chunks.every(chunk => chunk.model === model), 'stream must preserve exact model ID');
    assert.equal(chunks.map(chunk => chunk.choices[0]?.delta?.content || '').join(''), reply);
    assert(chunks.some(chunk => chunk.choices[0]?.finish_reason === 'stop'));
  }
}


async function failureControl(name, config, hostEnv, args = [], diagnostic = /native runtime|native-runtime/i, setup = () => {}) {
  const env = environment(name);
  fs.cpSync(hostEnv.MESH_LLM_PLUGIN_DIR, env.MESH_LLM_PLUGIN_DIR, { recursive: true });
  fs.writeFileSync(env.MESH_LLM_CONFIG, config);
  setup(env);
  const child = launch(name, ['serve', '--headless', '--no-default-plugins', ...args], env);
  const code = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { child.kill(); reject(new Error(name + ': expected startup failure, process remained alive')); }, 30000);
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('exit', value => { clearTimeout(timer); resolve(value); });
  });
  assert.notEqual(code, 0, name + ': expected startup failure');
  const log = fs.readFileSync(path.join(root, name + '.log'), 'utf8');
  assert.match(log, diagnostic, name + ': expected actionable diagnostic');
  return log;
}

async function startupControls(hostEnv, endpointConfig) {
  const local = path.join(root, 'fixture.gguf');
  fs.writeFileSync(local, 'GGUF fixture is deliberately not a model');
  const corruptLog = await failureControl('malformed-bundled-runtime', endpointConfig, hostEnv, [], /manifest|parse|json/i, env => {
    env.MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR = path.join(env.HOME, 'bundle');
    const broken = path.join(env.MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR, 'broken');
    fs.mkdirSync(broken);
    fs.writeFileSync(path.join(broken, 'manifest.json'), '{ deliberately malformed runtime manifest');
  });
  assert.doesNotMatch(corruptLog, /Serving external plugin inference without a native runtime/);
  await failureControl('no-endpoint', '', hostEnv);
  await failureControl('missing-url', endpointConfig.replace(/url = .*\n/, ''), hostEnv);
  await failureControl('disabled-endpoint', endpointConfig + 'enabled = false\n', hostEnv);
  await failureControl('missing-endpoint', endpointConfig + 'command = "' + path.join(root, 'does-not-exist').replaceAll('\\', '/') + '"\n', hostEnv, [], /Failed to launch plugin|native runtime|native-runtime/i);
  await failureControl('resolved-non-provider', endpointConfig + 'command = ' + JSON.stringify(process.execPath.replaceAll('\\', '/')) + '\nargs = ["-e", "process.exit(0)"]\n[plugin.startup]\nconnect_timeout_secs = 1\ninit_timeout_secs = 1\noptional = true\n', hostEnv);
  const nonInference = '[[plugin]]\nname = "fixture-non-inference"\ncommand = ' + JSON.stringify(process.execPath.replaceAll('\\', '/')) + '\nargs = [' + JSON.stringify(path.join(__dirname, 'fixtures/non-inference-plugin.cjs').replaceAll('\\', '/')) + ']\nurl = "http://127.0.0.1:' + stub.address().port + '/v1"\n';
  const nonInferenceLog = await failureControl('non-inference-endpoint', nonInference, hostEnv, [], /no enabled plugin completed a compatible inference endpoint handshake/);
  assert.match(nonInferenceLog, /fixture protocol-3 non-inference initialize response sent/);
  assert.doesNotMatch(nonInferenceLog, /identified itself as|uses protocol|failed initialize|invalid server_info_json/);
  await failureControl('blank-endpoint', endpointConfig.replace(/url = .*\n/, 'url = " "\n'), hostEnv, [], /url.*http|url.*absolute|invalid.*url/i);
  for (const [name, args] of [
    ['explicit-model', ['--model', local]],
    ['explicit-gguf', ['--gguf', local]],
    ['explicit-layer-package', ['--model', 'meshllm/Qwen3-8B-Q4_K_M-layers']],
    ['explicit-draft', ['--draft', local]],
    ['explicit-mmproj', ['--mmproj', local]],
    ['explicit-split', ['--split']],
  ]) await failureControl(name, endpointConfig, hostEnv, args);
  for (const [name, pin] of [
    ['backend-pin', 'selection = "cpu"'],
    ['version-pin', 'mesh_version = "99.0.0"'],
    ['abi-pin', 'mesh_version = "99.0.0"\nskippy_abi = "99.0.0"'],
  ]) await failureControl(name, endpointConfig + '\n[runtime.native_runtime]\n' + pin + '\n', hostEnv);
  await failureControl('configured-model', endpointConfig + '\n[[models]]\nmodel = "' + local.replaceAll('\\', '/') + '"\n', hostEnv);
}

async function main() {
  fs.accessSync(sourceBinary, fs.constants.X_OK);
  // Discovery also inspects executable-adjacent bundles. Copy the neutral host
  // so even a product with native-runtimes beside it cannot contaminate this test.
  fs.copyFileSync(sourceBinary, binary);
  fs.chmodSync(binary, 0o755);
  stub = http.createServer(async (req, res) => {
    if (req.url === '/v1/models') {
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ object: 'list', data: [{ id: model, object: 'model', owned_by: 'fixture' }] }));
      return;
    }
    if (req.method !== 'POST' || req.url !== '/v1/chat/completions') { res.writeHead(404); res.end(); return; }
    let bytes = '';
    for await (const chunk of req) bytes += chunk;
    const body = JSON.parse(bytes);
    requests.push({ model: body.model, stream: !!body.stream });
    if (body.model !== model) { res.writeHead(400); res.end('unexpected model'); return; }
    const common = { id: 'chatcmpl-fixture', created: 1, model };
    if (body.stream) {
      res.setHeader('content-type', 'text/event-stream');
      for (const content of ['Endpoint ', 'fixture reply']) {
        res.write(`data: ${JSON.stringify({ ...common, object: 'chat.completion.chunk', choices: [{ index: 0, delta: { content }, finish_reason: null }] })}\n\n`);
      }
      res.end(`data: ${JSON.stringify({ ...common, object: 'chat.completion.chunk', choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`);
    } else {
      res.setHeader('content-type', 'application/json');
      res.end(JSON.stringify({ ...common, object: 'chat.completion', choices: [{ index: 0, message: { role: 'assistant', content: reply }, finish_reason: 'stop' }], usage: { prompt_tokens: 1, completion_tokens: 3, total_tokens: 4 } }));
    }
  });
  await new Promise(resolve => stub.listen(0, '127.0.0.1', resolve));
  // A closed loopback listener gives a bounded connection-refused catalog failure.
  unavailableManifestPort = await port();
  const hostEnv = environment('host');
  const endpointConfig = `[[plugin]]\nname = "openai-endpoint"\nurl = "http://127.0.0.1:${stub.address().port}/v1"\n`;
  fs.writeFileSync(hostEnv.MESH_LLM_CONFIG, endpointConfig);
  await command('install', archive
    ? ['plugins', 'install', '--archive', archive, '--name', 'openai-endpoint', '--version', '0.2.0']
    : ['plugins', 'install', 'Mesh-LLM/openai-endpoint@0.2.0'], hostEnv);
  await startupControls(hostEnv, endpointConfig);
  const api = await port(), consolePort = await port(), bind = await port();
  const host = launch('host', ['serve', '--headless', '--no-default-plugins', '--port', String(api), '--console', String(consolePort), '--bind-port', String(bind)], hostEnv);
  const status = await waitFor(host, 'host API and endpoint discovery', async () => {
    const value = await json(`http://127.0.0.1:${consolePort}/api/status`);
    const models = await json(`http://127.0.0.1:${api}/v1/models`);
    return value.token && models.data.some(item => item.id === model) && value;
  });
  assert.equal(status.llama_ready, false, 'plugin-only node must not claim native readiness');
  assert.equal(status.my_memory.usable_bytes, 0, 'plugin-only node must expose no usable native memory');
  assert.equal(status.my_vram_gb, 0, 'plugin-only node must advertise no native capacity');
  assert.deepEqual(status.available_models, [], 'plugin-only node must advertise no local model sources');
  fs.writeFileSync(path.join(root, 'host-status.json'), JSON.stringify(status, null, 2));
  const localLoad = await fetch(`http://127.0.0.1:${consolePort}/api/runtime/models`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: path.join(root, 'fixture.gguf') }), signal: AbortSignal.timeout(10000),
  });
  const localLoadBody = await localLoad.text();
  assert(localLoad.status >= 400, 'later local-model load must reject');
  assert.match(localLoadBody, /native runtime|native-runtime/i);
  fs.writeFileSync(path.join(root, 'local-load-rejection.json'), localLoadBody);
  await inference(`http://127.0.0.1:${api}`, false);
  await inference(`http://127.0.0.1:${api}`, true);
  const clientEnv = environment('client');
  fs.writeFileSync(clientEnv.MESH_LLM_CONFIG, '');
  const clientApi = await port(), clientConsole = await port();
  const client = launch('client', ['client', '--headless', '--no-default-plugins', '--join', status.token, '--port', String(clientApi), '--console', String(clientConsole)], clientEnv);
  await waitFor(client, 'client discovers endpoint model', async () => {
    const value = await json(`http://127.0.0.1:${clientApi}/v1/models`);
    return value.data.some(item => item.id === model);
  });
  await json(`http://127.0.0.1:${clientConsole}/api/status`);
  await inference(`http://127.0.0.1:${clientApi}`, false);
  await inference(`http://127.0.0.1:${clientApi}`, true);
  const hostLog = fs.readFileSync(path.join(root, 'host.log'), 'utf8');
  const warnings = hostLog.split(/\r?\n/).filter(line => {
    try {
      const event = JSON.parse(line);
      return event.event === 'warning' && event.message?.startsWith('Serving external plugin inference without a native runtime');
    } catch { return false; }
  });
  assert.equal(warnings.length, 1, 'runtime-unavailable warning must appear exactly once');
  assert.equal(requests.length, 4);
  assert.deepEqual(requests.map(item => item.stream), [false, true, false, true]);
  fs.writeFileSync(path.join(root, 'upstream-requests.json'), JSON.stringify(requests, null, 2));
  assert.equal(fs.readdirSync(hostEnv.MESH_LLM_NATIVE_RUNTIME_CACHE_DIR).length, 0);
  assert.equal(status.node_state, 'serving', 'available external inference must report serving');
  console.log(`PASS: real protocol-3 adapter, no native runtime, host and peer discovery, non-stream and stream relay. Evidence: ${root}`);
}

main().catch(error => { console.error(error); process.exitCode = 1; }).finally(async () => {
  for (const child of children.reverse()) {
    if (child.exitCode === null && child.pid) {
      try { process.platform === 'win32' ? child.kill('SIGTERM') : process.kill(-child.pid, 'SIGTERM'); } catch {}
    }
  }
  await pause(500);
  for (const child of children) {
    if (child.exitCode === null && child.pid) {
      try { process.platform === 'win32' ? child.kill('SIGKILL') : process.kill(-child.pid, 'SIGKILL'); } catch {}
    }
  }
  if (stub) { stub.closeAllConnections(); stub.close(); }
  console.log(`Fixture logs retained at ${root}`);
});
