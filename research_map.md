# Research Map: FlyWire Connectome → Embodied Controller

## Core Literature

### Platform & Data

| Paper | Key contribution | Relevance |
|-------|-----------------|-----------|
| Dorkenwald et al. 2022, *Nature Methods* 19:119–128 | FlyWire platform: browser-based 3D EM proofreading, collaborative segmentation, mechanosensory circuit demo | Data schema, annotation pipeline, how the graph is built |
| Dorkenwald et al. 2024, *Nature* (Oct 2, 9-paper package) | 139,255 proofread neurons, >50M synaptic connections, full cell-type annotation | The wiring diagram used for analysis; companion paper by Schlegel et al. adds hierarchical annotation |
| Lin, Yang et al. 2024, *Nature* 634:153–165 | Network statistics of the whole-brain connectome: rich-club organisation, ~30% high-degree hub neurons, hierarchy and community structure on v630 snapshot | Direct methodological template for Project A; scripts on GitHub |

### Connectome → Controller

| Paper | Key contribution | Relevance |
|-------|-----------------|-----------|
| flyGNN — NeurIPS 2025 | Connectome as recurrent message-passing GNN; gait initiation, straight walking, turning on biomechanical fruit fly simulator | First proof that whole-brain connectome topology supports locomotion control |
| FlyGM — arXiv 2602.17997, 2026 | Static connectome as directed message-passing graph; afferent/intrinsic/efferent node partition; outperforms MLP, random, degree-preserving-rewired baselines; no task-specific arch tuning | Core inspiration for Project B; directly maps to sensor/inter/motor scheme |

---

## Field Logic (EM → Embodied Controller)

```
1. Dense EM imaging of the whole adult Drosophila brain
2. Automated ML segmentation + community proofreading (FlyWire platform)
3. Graph construction: neurons = nodes, synaptic contacts = directed weighted edges
4. Structural analysis: degree, betweenness, hierarchy, modules, feedforward/feedback balance
5. Graph priors → controller inductive biases:
     - Hub neurons        → wider hidden states in GNN
     - Feedback edges     → residual skip-connections
     - Lateral recurrence → implicit working memory (no extra LSTM)
     - Community modules  → modular sub-policies
6. Instantiate connectome-shaped GNN controller in embodied environment
7. Compare learning dynamics against random / hand-designed / MLP baselines
```

---

## Implementation Path

### Phase 1 — Literature + codebase digestion (complete)
- Read Dorkenwald 2022, Lin et al. 2024, FlyGM arXiv 2602.17997
- Understand FlyWire data schema via [murthylab/codex](https://github.com/murthylab/codex)
- Explore [flywire-network-analysis scripts](https://github.com/murthylab/flywire-network-analysis)

### Phase 2 — Project A: Structural analysis (scaffold done)
- `notebooks/minimal_structural_analysis.ipynb` runs on toy data
- **Next**: replace toy CSV with FlyWire v630 snapshot ([Zenodo:10676866](https://zenodo.org/records/10676866))
- Annotate with Schlegel et al. 2024 cell types; subset sensorimotor subgraph
- Output: GRC score, FF/FB/lateral fractions, hub list, community partition

### Phase 3 — Project B: Toy connectome-shaped GNN controller (designed, not yet coded)
- Graph: `data/toy_connectome_{nodes,edges}.csv` (30 nodes, sensor/inter/motor)
- GNN: sensor nodes → input features; motor nodes → output heads; message-passing over all edges
- Environment: 2D point-mass navigation or CartPole (minimal RL task)
- Compare: connectome-like topology vs random graph vs MLP of same parameter count
- Metrics: sample efficiency, stability, asymptotic reward

### Phase 4 — Application narrative + portfolio (in progress)
- `application_draft.md` → polished personal statement
- This notebook + toy controller → GitHub portfolio

---

## Key Resources

| Resource | URL | Use for |
|----------|-----|---------|
| FlyWire homepage | https://flywire.ai | Visualise neurons, build intuition |
| Codex data portal | https://codex.flywire.ai/api/download | Programmatic bulk data access |
| murthylab/codex | https://github.com/murthylab/codex | Python API for querying neurons |
| flywire-network-analysis | https://github.com/murthylab/flywire-network-analysis | Analysis scripts (Python graph-tool) |
| Zenodo v630 snapshot | https://zenodo.org/records/10676866 | Real connectivity data for Project A |
| Zenodo analysis data | https://zenodo.org/records/12572930 | Companion data for Lin et al. 2024 |
| FlyGM arXiv | https://arxiv.org/abs/2602.17997 | Project B architecture reference |
| PNI SIP application | https://pni.princeton.edu/apply/undergraduate-summer-research-programs/application-process | Target program |

---

## Future Directions

- **Better connectome → controller abstractions**: learned compression of large-scale graph into small tractable controller topology.
- **Multi-scale subgraph selection**: automatically identify task-relevant sub-circuits (e.g., escape vs. feeding vs. locomotion) for task-conditional controller priors.
- **Comparing biological priors vs learned sparse topologies**: does connectome structure outperform RL-discovered sparse architectures, and under what conditions?
- **Memory and planning grounded in connectomic motifs**: map hippocampal-like motifs in the fly central complex to agent memory/planning modules.
- **Multi-animal connectome comparison**: use the multi-connectome variation data to study robustness of controller priors across individual animals.
