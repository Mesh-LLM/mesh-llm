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
      imageAlt: 'Buzz channel where people and agents plan and ship a change together',
      imageFit: 'cover',
      url: 'https://buzz.xyz',
      linkLabel: 'Visit buzz.xyz',
      linkUrl: 'https://buzz.xyz',
      uses: ['Agents on mesh models', 'OpenAI-compatible API'],
      platforms: 'macOS · iOS',
    },
    {
      name: 'Earn and burn',
      tagline: 'Earn sats serving models, spend them on models you use',
      summary:
        'Install a wallet plugin and your node gets paid in Lightning micropayments when it serves inference, then pays other nodes for the models it calls. Lexe can create a wallet for you; NWC connects one you already have. Preview: use small amounts.',
      image: '/assets/showcase/earn-burn.svg',
      imageAlt: 'Sats flowing in from serving inference and out to pay for models',
      imageFit: 'contain',
      url: '/plugins/',
      linkLabel: 'See wallet plugins',
      linkUrl: '/plugins/',
      uses: ['lexe-wallet', 'nwc-wallet', 'Lightning micropayments'],
      platforms: 'macOS, Linux, Windows',
    },
  ],
};
