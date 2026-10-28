// Apps built on mesh-llm. Each entry renders as a card on /showcase/.
// `image` paths are relative to the site root; keep screenshots in /assets/showcase/.
export default {
  apps: [
    {
      name: 'Mesh',
      tagline: 'mesh-llm from your menu bar',
      summary:
        'A desktop launcher that runs mesh-llm from the menu bar or system tray. It picks a model that fits your machine, downloads it on first use, and opens chat in one click. Join the public mesh or run a private one and invite people with a code.',
      image: '/assets/showcase/mesh-app.png',
      imageAlt: 'Mesh desktop app settings panel',
      imageFit: 'contain',
      url: 'https://github.com/Mesh-LLM/mesh-app',
      linkLabel: 'Download for macOS',
      linkUrl: 'https://github.com/Mesh-LLM/mesh-app/releases/download/dev-latest/Mesh-macos-arm64.dmg',
      uses: ['Public and private meshes', 'Automatic model setup', 'OpenAI-compatible API'],
      platforms: 'macOS arm64 · Windows, Linux preview',
    },
    {
      name: 'Buzz',
      tagline: 'Your people, your agents, your project — in one place',
      summary:
        'A collaboration app where people and AI agents work together in channels. Buzz agents can run on models served by mesh-llm.',
      image: '/assets/showcase/buzz.png',
      imageAlt: 'Buzz logo',
      imageFit: 'cover',
      url: 'https://buzz.xyz',
      linkLabel: 'Visit buzz.xyz',
      linkUrl: 'https://buzz.xyz',
      uses: ['Agents on mesh models', 'OpenAI-compatible API'],
      platforms: 'macOS · iOS',
    },
  ],
};
