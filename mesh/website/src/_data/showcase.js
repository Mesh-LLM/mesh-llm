// Apps built on mesh-llm. Each entry renders as a card on /showcase/.
// `image` paths are relative to the site root; keep screenshots in /assets/showcase/.
export default {
  apps: [
    {
      name: 'Mesh App',
      tagline: 'A desktop launcher for mesh-llm',
      summary:
        'Mesh App runs mesh-llm for you: on demand, at set times of day, or when the sun is shining and your power is free. It also helps you hook mesh-llm up to the agents and apps you already use.',
      image: '/assets/showcase/mesh-app.png',
      imageAlt: 'Mesh desktop app settings panel',
      imageFit: 'contain',
      url: 'https://github.com/Mesh-LLM/mesh-app',
      linkLabel: 'Download for macOS',
      linkUrl: 'https://github.com/Mesh-LLM/mesh-app/releases/download/dev-latest/Mesh-macos-arm64.dmg',
      uses: ['Scheduled and solar-aware serving', 'Connects your agents and apps', 'Public and private meshes'],
      platforms: 'macOS arm64 · Windows, Linux preview',
    },
    {
      name: 'Buzz',
      tagline: 'Your people, your agents, your project — in one place',
      summary:
        'Buzz has “shared compute”, powered by mesh-llm: your agents can run on compute pooled by your community, for free.',
      image: '/assets/showcase/buzz.png',
      imageAlt: 'Buzz channel where people and agents plan and ship a change together',
      imageFit: 'cover',
      url: 'https://buzz.xyz',
      linkLabel: 'Visit buzz.xyz',
      linkUrl: 'https://buzz.xyz',
      uses: ['Shared compute', 'Agents on mesh models'],
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
