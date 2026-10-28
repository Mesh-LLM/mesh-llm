// First-party Mesh-LLM plugins with a published GitHub release.
// Single source for the homepage plugin strip and the Plugins docs table.
// Only add a plugin here once it has a release with installable archives.
const repo = (name) => `https://github.com/Mesh-LLM/${name}`;

const escapeHtml = (text) =>
  text.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

// Summaries use a tiny Markdown subset (links and inline code) so the same text
// works in the docs table and in HTML components.
const toHtml = (markdown) =>
  escapeHtml(markdown)
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');

const groups = [
  { id: 'inference', title: 'Inference runtimes' },
  { id: 'payments', title: 'Payments' },
  { id: 'tools', title: 'Tools and integrations' },
];

const plugins = [
  {
    name: 'ds4',
    icon: 'brain-circuit',
    repo: 'ds4-plugin',
    group: 'inference',
    tagline: 'DeepSeek V4 Flash on your Mac',
    summary:
      "Run DeepSeek V4 Flash, a large and powerful open model, locally with [DwarfStar (ds4)](https://github.com/antirez/ds4): antirez's small native inference engine built specifically for DeepSeek V4. The model downloads on first start. [Setup](https://github.com/Mesh-LLM/mesh-llm/blob/main/docs/plugins/dwarfstar.md).",
    install: 'mesh-llm plugins install Mesh-LLM/ds4-plugin',
    platforms: 'macOS arm64',
  },
  {
    name: 'uzu',
    icon: 'zap',
    repo: 'uzu-runtime-plugin',
    group: 'inference',
    tagline: 'Super-fast, Metal-native Mac inference',
    summary:
      'Experimental. Very fast inference that runs natively on Apple Silicon GPUs through Metal, powered by the [Uzu](https://github.com/trymirai/uzu) engine. Serves Uzu-converted models; requires macOS 26.4+.',
    install: 'mesh-llm plugins install Mesh-LLM/uzu-runtime-plugin',
    platforms: 'macOS arm64',
  },
  {
    name: 'eredu',
    icon: 'cpu',
    repo: 'eredu-plugin',
    group: 'inference',
    tagline: 'MLX-powered local models',
    summary:
      'Early access. Run MLX models on Apple Silicon with [Eredu](https://github.com/jbg/eredu), served through the normal OpenAI-compatible API. Text chat and streaming.',
    install: 'mesh-llm plugins install Mesh-LLM/eredu-plugin',
    platforms: 'macOS arm64',
  },
  {
    name: 'flash-moe',
    icon: 'gauge',
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
    icon: 'plug',
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
    icon: 'clipboard-list',
    repo: 'blackboard-plugin',
    group: 'tools',
    tagline: 'Agent coordination',
    summary:
      'Share short-lived status, findings, questions, and tips across a mesh. Also ships a Blackboard Agent Skill.',
    install: 'mesh-llm plugins install blackboard',
    platforms: 'macOS, Linux, Windows',
  },
  {
    name: 'agents',
    icon: 'bot',
    repo: 'agents-plugin',
    group: 'tools',
    tagline: 'Mesh-native A2A agents',
    summary: 'Run mesh-native A2A agents and expose their tools through the mesh MCP endpoint.',
    install: 'mesh-llm plugins install agents',
    platforms: 'macOS, Linux, Windows',
  },
  {
    name: 'capsules',
    icon: 'shield-check',
    repo: 'capsules-plugin',
    group: 'tools',
    tagline: 'Signed exchange evidence',
    summary:
      'Keep a signed, hash-chained record of every exchange a node serves or requests, with an Evidence page in the console.',
    install: 'mesh-llm plugins install Mesh-LLM/capsules',
    platforms: 'macOS arm64, Linux',
  },
  {
    name: 'lexe-wallet',
    icon: 'wallet',
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
    icon: 'tag',
    repo: 'model-price-plugin',
    group: 'tools',
    tagline: 'Model prices in the console',
    summary:
      'Preview. Read-only console page showing model prices on a mesh. No wallet needed. Install from the release archive; see the [README](https://github.com/Mesh-LLM/model-price-plugin#readme).',
    install: null,
    platforms: 'macOS arm64, Windows arm64',
  },
  {
    name: 'metrics',
    icon: 'activity',
    repo: 'metrics-plugin',
    group: 'tools',
    tagline: 'Telemetry support',
    summary:
      'Advertise metrics support for mesh-llm telemetry. Configure the OTLP destination in mesh-llm, not in the plugin.',
    install: 'mesh-llm plugins install metrics',
    platforms: 'macOS, Linux, Windows',
  },
].map((plugin) => ({
  ...plugin,
  url: repo(plugin.repo),
  groupTitle: groups.find((group) => group.id === plugin.group).title,
  summaryHtml: toHtml(plugin.summary),
}));

// Keep `all` in group order so carousel cards and panels line up with the grouped docs.
plugins.sort((a, b) => groups.findIndex((g) => g.id === a.group) - groups.findIndex((g) => g.id === b.group));

export default {
  groups: groups.map((group) => ({
    ...group,
    plugins: plugins.filter((plugin) => plugin.group === group.id),
  })),
  all: plugins,
  count: plugins.length,
};
