# Mixture of Models (MoM) Experiment

## Goal

Test whether the NSED protocol (from the Mixture-of-Models paper) actually works for practical tasks — including tool calls — using our real mesh infrastructure. We want to know:

1. **Can 2-3 small/medium models ensemble to beat any single one?**
2. **Does it work for tool calls** (not just math/code benchmarks)?
3. **What's the latency cost** of deliberation rounds?
4. **Is this feasible in production** as a mesh-llm routing strategy?

## Paper Summary (NSED Protocol)

The key idea is simple despite the academic language:

1. **Multiple models get the same prompt** (parallel generation)
2. **Each model proposes an answer**
3. **All models score each other's answers** (but can't score their own — "diagonal mask")
4. **Best-scored answer wins** (quadratic voting — sqrt of scores)
5. **Repeat** with the winning answer as context (recurrent refinement)
6. **Stop** when convergence delta drops below threshold

The paper's consumer ensemble: GPT-OSS-20B + Qwen3-8B + Gemma-12B
Achieved 84% on AIME 2025 (vs DeepSeek-R1's 84.2%), 60.2% on LiveCodeBench Hard.

Key hyperparams: presence_penalty=1.5, max_tokens=20000, up to 6-7 rounds.

## Our Setup

### Models (3 agents)

| Agent | Model | Where | Role |
|-------|-------|-------|------|
| **Alpha** (strong) | MiniMax-M2.5-Q4_K_M (253B MoE) | Studio (206GB) | Lead proposer |
| **Beta** (medium) | Qwen3-8B-Q4_K_M | Local (64GB Mac) | Diverse reasoner |
| **Gamma** (small) | Qwen3.5-0.8B-Q4_K_M | Local (64GB Mac) | Quick evaluator |

Alternative: Swap Gamma for Qwen2.5-72B on James (stronger but higher latency).

### Access

- Studio's MiniMax: via mesh proxy `https://www.mesh-llm.com/v1` or direct
- Local models: 2 separate llama-server instances on different ports

## Implementation Plan

### Phase 1: Orchestrator Script (Python)

A standalone `mom/orchestrator.py` that:
1. Spins up 2 local llama-server processes (Qwen3-8B on :8081, Qwen3.5-0.8B on :8082)
2. Connects to MiniMax via the mesh API
3. Implements the NSED protocol loop

### Phase 2: Core Protocol

```
for round in range(max_rounds):
    # Phase 1: Parallel Generation
    # All 3 models get: system_prompt + user_query + consensus_state
    proposals = parallel_generate(agents, query, consensus_state)
    
    # Phase 2: Trustless Evaluation
    # Each model scores the OTHER proposals (not its own)
    # Proposals are anonymized ("Proposal A", "Proposal B", "Proposal C")
    vote_matrix = parallel_evaluate(agents, proposals)
    
    # Phase 3: Quadratic Aggregation
    # score = sum(sign(v) * sqrt(|v|)) for each proposal
    scores = quadratic_aggregate(vote_matrix)
    winner = proposals[argmax(scores)]
    
    # Phase 4: Convergence Check
    delta = semantic_distance(winner, consensus_state)
    if delta < epsilon:
        break
    consensus_state = winner
```

### Phase 3: Tool Call Support

This is the novel part not in the paper. For tool calls:
- Models propose tool calls as structured JSON
- Evaluators score tool call correctness (right function? right args?)
- Winner's tool call gets executed
- Result fed back into next round

### Phase 4: Benchmarks

Test cases:
1. **Math**: A few AIME-style problems to validate the protocol works
2. **Coding**: Write a function, models evaluate each other's code
3. **Tool use**: "What's the weather in Sydney?" with a mock weather tool
4. **Practical**: "Summarize this text and extract key dates" (real-world)
5. **Comparison**: Same prompts to each model solo vs the ensemble

## What We're NOT Doing

- No fine-tuning or training
- No logit-level access (all via OpenAI API)
- No complex broker/knapsack optimization (just use all 3 models every round)
- No "personas" with presence_penalty tricks (keep it simple first)
- Not integrated into mesh-llm router yet (standalone experiment)

## Success Criteria

- Ensemble measurably outperforms any single model on at least 2/4 task types
- Tool calls work correctly through the deliberation loop
- Total latency per query < 3 minutes (practical for interactive use)
- Clear signal on whether this should become a mesh-llm feature

## Files

```
mom/
  PLAN.md          — this file
  orchestrator.py  — main NSED protocol implementation
  agents.py        — model/endpoint config, generation/scoring
  evaluate.py      — test suite and benchmarks
  results/         — benchmark outputs
```
