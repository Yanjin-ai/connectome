# Application Draft — FlyWire / Princeton Neuroscience Institute Summer Internship

*Target program*: Princeton Neuroscience Institute Summer Internship Program (9 weeks, June 1 – July 31)  
*Lab interest*: Murthy lab / FlyWire Consortium — computational connectomics  
*Status*: Draft. Fill in [brackets] with your personal background before submitting.

---

## Personal Statement

### Motivation

My long-term research direction is embodied AI: building agents that act in physical environments through tight coupling between perception, internal state, and motor output. Most current architectures treat the controller as a blank slate — an MLP or transformer that learns everything from scratch. But biological brains solve embodied control problems reliably and efficiently, and the reason is not just neurons; it is *wiring*. The architecture encodes priors.

The completion of the adult *Drosophila* whole-brain connectome (Dorkenwald et al. 2024, *Nature*) makes it possible, for the first time, to use a real, annotated wiring diagram as a structural prior for a computational controller. Recent work (FlyGM, arXiv 2602.17997; flyGNN, NeurIPS 2025) has already demonstrated that instantiating the fly connectome as a directed message-passing graph and training it with reinforcement learning achieves stable locomotion control — outperforming MLP, random-graph, and rewired baselines without any task-specific architectural modification. This tells us the *structure itself* carries information.

I want to understand what that information is and how to extract it systematically. To do that concretely before applying, I built and ran two independent projects on this question.

### Skills

My technical background is centred on Python, graph-based analysis, and machine learning systems. Specifically:

- **Graph analysis**: NetworkX; degree/centrality/community analysis; Louvain clustering; Global Reaching Centrality and FF/FB decomposition on large directed weighted graphs.
- **GNN and RL**: Pure PyTorch GNN with fixed-topology adjacency; PPO via Stable-Baselines3; experience integrating custom graph controllers into RL training loops using the `BaseFeaturesExtractor` interface.
- **Data engineering**: pandas, NumPy, reproducible notebook pipelines; comfortable processing large biological datasets (processed a 16.8M-row synaptic connection table from the FlyWire Zenodo release).
- **AI-assisted research**: fluent in using LLM coding assistants to accelerate literature digestion, exploratory coding, and experiment scaffolding — relevant to the kind of fast-iteration computational work FlyWire projects require.
- [**Additional background**: add your degree program, relevant courses, prior research, GitHub handle]

### Prior Work

I have completed two projects directly preparatory to a FlyWire internship, described below. Code and results are at [GitHub link].

#### Project A — Structural analysis of the FlyWire v783 whole-brain connectome

I downloaded the full FlyWire v783 connectome from Zenodo (Dorkenwald et al. 2024) — 134,013 neurons, 16.8M raw synaptic connections — and filtered to 2,710,038 connections with syn_count ≥ 5. I classified neurons by neuropil assignment into sensory, inter-, and motor categories (139,255 annotated), then quantified four structural features relevant to controller design:

| Metric | Value |
|--------|-------|
| Feedforward weight fraction | **10.9%** |
| Feedback weight fraction | **6.4%** |
| Lateral / recurrent fraction | **82.8%** |
| Global Reaching Centrality (unweighted) | **0.014** |
| Functional communities (≥ 50 neurons) | **12** |
| Max betweenness centrality | **0.030** |

The near-zero GRC (0.014) means the brain is *not* a feedforward pipeline — information does not flow strictly sensor → inter → motor. The 82.8% lateral fraction means dominant within-layer recurrence, providing intrinsic temporal memory capacity. These numbers translate directly into architectural specifications: a connectome-shaped GNN should retain feedback and lateral edges, not just feedforward; 2–3 message-passing steps match the typical sensor-to-motor path length; and the 12-community modular structure suggests potential for hierarchical or modular sub-policies.

**Deliverable**: reproducible analysis notebook, 4-panel structural summary figure, and a written architecture specification derived from the connectome data.

#### Project B — Connectome-shaped GNN controller experiment

Using the specifications from Project A, I implemented and trained three matched controllers on CartPole-v1 with PPO across 5 random seeds each:

| Architecture | Params | Adjacency | Steps to solve | Final reward | Stability |
|-------------|--------|-----------|---------------|-------------|-----------|
| ConnectomeGNN | 2,306 | Biological topology (30 nodes, FF + FB + lateral) | 62k ± 19k | 388 ± 141 | unstable |
| RandomGraph GNN | 2,306 | Erdős–Rényi (same n, same edge count) | 39k ± 8k | 410 ± 115 | unstable |
| MLP | 1,355 | No graph structure | **22k ± 4k** | **500 ± 0** | **stable** |

The MLP converged fastest and most stably. The connectome GNN was the slowest by a factor of ~3. This is a **meaningful negative result**: the 82.8% lateral recurrence that provides temporal memory capacity is a *liability* on a reactive task like CartPole, which requires no memory — the controller must integrate noise from unnecessary feedback loops on every step. This refines the hypothesis: connectome topology should benefit tasks requiring multi-sensory temporal integration, not pure reflex control, directly motivating follow-on experiments on environments like LunarLander or a minimal locomotion task.

**Deliverable**: training code (resumable multi-seed runner), 4-panel comparison figure, and a written interpretation of the result.

### Why This Program

The FlyWire / PNI environment is uniquely suited to continuing this work because:

1. **Data access and annotation quality**: the FlyWire connectome with Schlegel et al. 2024 cell-type annotations, proofread at single-neuron resolution, is available nowhere else. My Project A used neuropil-based classification as a proxy; cell-type annotations would make the sensorimotor subgraph far more precise.
2. **Expertise**: the Murthy lab produced the network statistics paper (Lin et al. 2024) that Project A builds on. Mentorship from people who know the data deeply is essential for interpreting what the structural metrics actually mean biologically.
3. **Computational mission**: FlyWire is explicitly interested in analysts who can push the computational interpretation of connectome data. The two-stage framework I have already built fits squarely in that mission and would scale naturally to real subgraphs and harder tasks.
4. **Long-term alignment**: understanding how biological wiring encodes controller priors is a foundational question for embodied AI. A summer in this environment would give me the domain knowledge needed to ask sharper questions than I currently can.

---

## Project Idea (short form, for form fields)

**Title**: Connectome structure as inductive bias for embodied GNN controllers

**One-sentence summary**: Quantify the structural priors encoded in the FlyWire whole-brain connectome (hierarchy, feedforward/feedback balance, hub topology, community structure), then test whether a GNN controller instantiated on this topology learns faster and more stably than random or parameter-matched MLP baselines — and identify under what task conditions the biological prior is beneficial.

**Background**: Recent work (FlyGM, arXiv 2602.17997; flyGNN, NeurIPS 2025) showed that a connectome-shaped message-passing network achieves whole-body locomotion control, outperforming MLP and random-graph baselines on sample efficiency. This suggests the *architecture* encodes useful computation, but the specific structural features responsible — and the task conditions under which they help — are not yet identified.

I have already completed a small-scale version of both stages: structural analysis of the full FlyWire v783 connectome (GRC = 0.014, FF = 10.9%, lateral = 82.8%, 12 functional communities) and a controlled experiment comparing a connectome-shaped GNN against random-graph and MLP baselines on CartPole (MLP wins; connectome GNN takes ~3× more steps to converge). The negative result is informative: it identifies CartPole as a reactive task where recurrent topology is a liability, pointing directly toward temporal-integration tasks as the right experimental domain.

**Proposed summer work**:  
*Stage 1* — Replace the toy 30-node connectome with the real FlyWire sensorimotor subgraph (using Schlegel et al. cell-type annotations to extract the relevant neurons). Recompute structural metrics and characterize how they differ from the toy proxy.  
*Stage 2* — Scale the controller experiment to an environment requiring temporal integration (LunarLander or a minimal locomotion task). Test whether the connectome-topology advantage identified in FlyGM replicates at this scale, and which specific structural features — feedback edges, hub neurons, community modularity — drive any observed advantage via ablations.

**Expected outcome**: A reproducible analysis pipeline for connectome → controller specification, a controlled experiment isolating the contribution of biological wiring to RL sample efficiency, and a short report mapping graph properties to learning dynamics.

**Skills**: Python, NetworkX, PyTorch (custom GNN), Stable-Baselines3 PPO, large biological data pipelines. All demonstrated in prior work described above.
