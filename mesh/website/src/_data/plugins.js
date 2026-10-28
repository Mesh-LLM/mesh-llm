// First-party Mesh-LLM plugins with a published GitHub release.
// Single source for the homepage plugin strip and the Plugins docs table.
// Only add a plugin here once it has a release with installable archives.
const repo = (name) => `https://github.com/Mesh-LLM/${name}`;

const groups = [
  { id: 'inference', title: 'Inference runtimes' },
  { id: 'integration', title: 'Integrations' },
  { id: 'payments', title: 'Payments' },
  { id: 'utility', title: 'Utilities' },
];

const plugins = [
  {
    name: 'ds4',
    repo: 'ds4-plugin',
    group: 'inference',
    tagline: 'DeepSeek V4 Flash via DwarfStar',
    summary:
      'Run [DwarfStar (ds4)](https://github.com/antirez/ds4) models such as DeepSeek V4 Flash on Apple Silicon. The model downloads on first start. [Setup](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/plugins/dwarfstar.md).',
    install: 'mesh-llm plugins install Mesh-LLM/ds4-plugin',
    platforms: 'macOS arm64',
  },
  {
    name: 'uzu',
    repo: 'uzu-runtime-plugin',
    group: 'inference',
    tagline: 'Uzu Apple Silicon engine',
    summary:
      "Experimental. Serve Uzu-converted models with [Uzu](https://github.com/trymirai/uzu)'s Apple Silicon inference engine. Requires macOS 26.4+.",
    install: 'mesh-llm plugins install Mesh-LLM/uzu-runtime-plugin',
    platforms: 'macOS arm64',
  },
  {
    name: 'eredu',
    repo: 'eredu-plugin',
    group: 'inference',
    tagline: 'Local MLX models',
    summary:
      'Early access. Run a local MLX model with [Eredu](https://github.com/jbg/eredu) through the OpenAI-compatible API. Text chat and streaming.',
    install: 'mesh-llm plugins install Mesh-LLM/eredu-plugin',
    platforms: 'macOS arm64',
  },
  {
    name: 'flash-moe',
    repo: 'flash-moe-plugin',
    group: 'inference',
    tagline: 'Flash-MoE endpoints',
    summary:
      'Attach a Flash-MoE inference endpoint, or let the plugin supervise a local Flash-MoE process.',
    install: 'mesh-llm plugins install flash-moe',
    platforms: 'macOS, Linux, Windows',
  },
  {
    name: 'openai-endpoint',
    repo: 'openai-endpoint-plugin',
    group: 'inference',
    tagline: 'Ollama, vLLM, LM Studio',
    summary:
      'Add an already-running OpenAI-compatible server such as Ollama, vLLM, LM Studio, TGI, or Lemonade Server.',
    install: 'mesh-llm plugins install openai-endpoint',
    platforms: 'macOS, Linux, Windows',
  },
  {
    name: 'blackboard',
    repo: 'blackboard-plugin',
    group: 'integration',
    tagline: 'Agent coordination',
    summary:
      'Share short-lived status, findings, questions, and tips across a mesh. Also ships a Blackboard Agent Skill.',
    install: 'mesh-llm plugins install blackboard',
    platforms: 'macOS, Linux, Windows',
  },
  {
    name: 'agents',
    repo: 'agents-plugin',
    group: 'integration',
    tagline: 'Mesh-native A2A agents',
    summary: 'Run mesh-native A2A agents and expose their tools through the mesh MCP endpoint.',
    install: 'mesh-llm plugins install agents',
    platforms: 'macOS, Linux, Windows',
  },
  {
    name: 'capsules',
    repo: 'capsules-plugin',
    group: 'integration',
    tagline: 'Signed exchange evidence',
    summary:
      'Keep a signed, hash-chained record of every exchange a node serves or requests, with an Evidence page in the console.',
    install: 'mesh-llm plugins install Mesh-LLM/capsules',
    platforms: 'macOS arm64, Linux',
  },
  {
    name: 'lexe-wallet',
    repo: 'lexe-wallet-plugin',
    group: 'payments',
    tagline: 'Lightning wallet (preview)',
    summary:
      'Preview. A `wallet.v1` plugin backed by Lexe for experimental paid inference. It can provision a mainnet wallet; use small amounts.',
    install: 'mesh-llm plugins install Mesh-LLM/lexe-wallet',
    platforms: 'macOS arm64, Windows arm64',
  },
  {
    name: 'model-prices',
    repo: 'model-price-plugin',
    group: 'utility',
    tagline: 'Model prices in the console',
    summary:
      'Preview. Read-only console page showing model prices on a mesh. No wallet needed. Install from the release archive; see the [README](https://github.com/Mesh-LLM/model-price-plugin#readme).',
    install: null,
    platforms: 'macOS arm64, Windows arm64',
  },
  {
    name: 'metrics',
    repo: 'metrics-plugin',
    group: 'utility',
    tagline: 'Telemetry support',
    summary:
      'Advertise metrics support for mesh-llm telemetry. Configure the OTLP destination in mesh-llm, not in the plugin.',
    install: 'mesh-llm plugins install metrics',
    platforms: 'macOS, Linux, Windows',
  },
].map((plugin) => ({ ...plugin, url: repo(plugin.repo) }));

export default {
  groups: groups.map((group) => ({
    ...group,
    plugins: plugins.filter((plugin) => plugin.group === group.id),
  })),
  all: plugins,
  count: plugins.length,
};
