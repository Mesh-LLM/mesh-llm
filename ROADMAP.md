# Roadmap

**High level focus areas and project ideas the mesh-llm community is thinking about. Contributions and more ideas welcome!**

## Inference proof

* Improve the initial proof-of-inference implementation based on commit-llm.
* Explore alternative approaches to proving inference.
* Investigate how to prove inference when using external inference engines.

## Trust & security

* Verify that nodes run untampered mesh-llm binaries.
* Make clients more resistant to prompt injection when using mesh inference.
* Improve the capsules plugin and its support in mesh-llm.
* Explore a node reputation system, potentially based on a web of trust.
* Improve resilience to network denial-of-service attacks.
* Continue security work, including red teaming and addressing Loupe findings.

## Usability

* Maintain a reliable baseline of high-quality inference capacity.
* Make mesh-llm easier for users and developers to use through apps and extensions. Explore desktop apps that run mesh-llm in the background, serve local users, and share spare compute.
* Let GPU owners prioritize their own jobs over incoming requests, manage priorities from a dashboard, and keep local workloads secure.
* Improve mobile support.

## Performance

* Optimize the Skippy inference engine.
* Reduce CPU usage, memory usage, and energy consumption.
* Improve load balancing across the decentralized network, including peer selection.
* Improve opt-in speculative decoding with n-grams and draft models.

## Privacy

* Optimize mesh-llm for trusted execution environments (TEEs).
* Explore privacy protections for inference workloads on hosts outside TEEs.
* Prevent unauthorized access and data exfiltration from serving nodes.

## Training

* Use the mesh for distributed pre-training.
* Use the mesh for distributed fine-tuning.
* Use the mesh for distributed reinforcement learning during post-training.

## Model splitting

* Improve serving for models split across the mesh.

## Mixture of agents

* Improve and benchmark serving split Mixture of Agents (MoA) workloads across the mesh.

## Wide model support

* Improve dynamic model selection.
* Support classification models, including jev-style models.
* Support image-generation models.
* Support image editing and inpainting from an input image, mask, and prompt.
* Support audio generation, including text-to-speech, as a peer capability.
* Support video generation as a peer capability.
* Support real-time models.
* Explore support for long-running asynchronous inference tasks.
* Keep mesh-llm compatible with new open-weight models as they emerge.
* Optimize mesh-llm for specific and widely used models.
* Explore how to optimize mesh-llm for agent swarms.

## Payments

* Refine the payment architecture and mechanics, including streaming payments and possible MPP experiments.
* Suggest default prices.
* Support payments when models are served across multiple nodes.
* Explore Lightning BOLT 12 support.
* Explore support for additional Lightning wallets.
* Add and maintain a stablecoin payment engine and wallet plugins.
* Explore additional payment rails, such as Ark or fiat.

## Buzz

* Maintain and improve the Buzz integration with mesh-llm.

## Dynamic sharing

* Make it easy for GPU providers to switch between their own workloads and earning revenue by sharing spare capacity with the mesh, so the mesh can act as a buyer of last resort for idle GPUs.
* Test and optimize support for additional inference engines, including SGLang and vLLM.

## Dev platform

* Improve the mesh-llm SDK and package it for multiple client languages so apps can easily launch client and serving nodes.
* Improve developer documentation for humans and agents.
* Build example and demo apps that show what developers can build with mesh-llm.
* Expand developer outreach and engagement.

## Testing

* Improve pre-release testing.
* Build simulations to evaluate potential network changes.

## Monitoring

* Improve live network health and performance monitoring. Track metrics such as average time to first token (TTFT), average throughput in tokens per second, total tokens served, total earnings, and the number of active consumers and providers.
* Add alerts for network health and performance.

## Build & release

* Improve the build and release toolchain across supported platforms.
* Improve the Windows build process.

## Distribution

* Explore integration with Vercel AI Gateway.
* Explore integration with OpenRouter.
* Explore how to adapt mesh-llm to specific use cases, such as open science.

## Analyses

* Build a micropayments profitability calculator.
* Estimate market size.
* Conduct competitive and SWOT analyses.

## Marketing

* Build and maintain the mesh website.
* Maintain the mesh social media presence.
* Publish blog posts.
* Reach out to potential partners.
