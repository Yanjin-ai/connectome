# Project A — Codex Execution Prompts

6 个独立 prompt，按顺序执行。每个都是自包含的，包括上下文 + 任务 + 预期输出 + 卡点处理。

卡点说明：
- P1 → P2 的关键依赖：P1 必须输出 schema_report.txt（含真实列名）
- P3 是最容易失败的：neuropil 列名必须和关键词匹配
- P5 三个算法都很慢，包含降级方案

---

## Prompt 1 — 数据下载 + Schema 探索

> **目的**：下载 4 个数据文件，探索所有列名和数据结构，输出 schema_report.txt 供后续 prompt 使用。
> **预计时间**：下载 15–30 分钟（视网速），代码运行 2 分钟。
> **卡点**：网络中断（脚本自动跳过已下载文件，重跑即可）。

```
You are helping execute Project A of a computational neuroscience research project.
The project is about analyzing the FlyWire whole-brain Drosophila connectome as a
structural prior for embodied GNN controllers.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Python: 3.14   Packages already installed: pandas, networkx, matplotlib, numpy, pyarrow, scipy, jupyter

TASK 1: Download data files
Run this script to download 4 files (~1.1 GB total) into data/real/:

    python3 scripts/download_data.py

Wait for it to finish. If it errors, check the error message and fix it (usually a
network timeout — just re-run; already-downloaded files are skipped automatically).

TASK 2: Schema exploration
After download completes, run the following Python code and save all output to
data/real/schema_report.txt. This file is REQUIRED by subsequent steps.

---START CODE---
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("data/real")
out_lines = []

def log(s=""):
    print(s)
    out_lines.append(str(s))

# --- connections file ---
log("=== proofread_connections_783.feather ===")
conn = pd.read_feather(DATA / "proofread_connections_783.feather")
log(f"Shape: {conn.shape}")
log(f"Columns: {list(conn.columns)}")
log(f"Dtypes:\n{conn.dtypes}")
log(f"Sample (first 3 rows):\n{conn.head(3).to_string()}")
log(f"Null counts: {conn.isnull().sum().to_dict()}")

# Identify key columns
pre_col  = conn.columns[0]
post_col = conn.columns[1]
syn_col  = conn.columns[2]
log(f"\nASSUMED COLUMNS: pre={pre_col!r}, post={post_col!r}, syn={syn_col!r}")
sc = conn[syn_col]
log(f"Syn count stats: min={sc.min()}, median={sc.median():.0f}, mean={sc.mean():.1f}, max={sc.max()}")
log(f"Pairs with syn>=5: {(sc>=5).sum():,}  ({100*(sc>=5).mean():.1f}%)")
log(f"Pairs with syn>=3: {(sc>=3).sum():,}  ({100*(sc>=3).mean():.1f}%)")
log(f"Unique pre-neurons:  {conn[pre_col].nunique():,}")
log(f"Unique post-neurons: {conn[post_col].nunique():,}")

# --- neuropil pre file ---
log("\n=== per_neuron_neuropil_count_pre_783.feather ===")
np_pre = pd.read_feather(DATA / "per_neuron_neuropil_count_pre_783.feather")
log(f"Shape: {np_pre.shape}")
log(f"Columns (first 5): {list(np_pre.columns[:5])}")
log(f"All columns: {list(np_pre.columns)}")
log(f"Dtypes (first 5): {dict(list(np_pre.dtypes.items())[:5])}")
log(f"Sample:\n{np_pre.head(2).to_string()}")

# --- neuropil post file ---
log("\n=== per_neuron_neuropil_count_post_783.feather ===")
np_post = pd.read_feather(DATA / "per_neuron_neuropil_count_post_783.feather")
log(f"Shape: {np_post.shape}")
log(f"All columns: {list(np_post.columns)}")
log(f"Sample:\n{np_post.head(2).to_string()}")

# --- root IDs ---
log("\n=== proofread_root_ids_783.npy ===")
root_ids = np.load(DATA / "proofread_root_ids_783.npy")
log(f"Shape: {root_ids.shape},  dtype: {root_ids.dtype}")
log(f"Sample IDs: {root_ids[:5]}")

# Save report
report_path = DATA / "schema_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(out_lines))
log(f"\nSchema report saved to: {report_path}")
---END CODE---

Expected output: data/real/schema_report.txt exists and contains column names for all 3 feather files.
Print the full contents of schema_report.txt when done.
```

---

## Prompt 2 — 建图 + 度分布

> **目的**：用真实列名建 NetworkX DiGraph，计算度分布，保存图对象。
> **前置条件**：Prompt 1 完成，data/real/ 有 4 个数据文件，schema_report.txt 存在。
> **预计时间**：建图 3–8 分钟，度分布即时。
> **卡点**：内存不足（降级方案：对 connections 采样 500k 条）。

```
You are executing Project A of a FlyWire connectome analysis project.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Packages: pandas, networkx, matplotlib, numpy, pyarrow (all installed)

CONTEXT:
- data/real/proofread_connections_783.feather  (852 MB, neuron-pair connections)
- data/real/schema_report.txt  (column names from Prompt 1 — read this first)
- Columns are approximately: col[0]=pre_neuron_id, col[1]=post_neuron_id, col[2]=synapse_count
  (verify exact names from schema_report.txt before proceeding)

TASK: Build a directed weighted graph and compute degree statistics.

Step 1 — Read schema_report.txt and extract the exact column names for:
  - pre-synaptic neuron ID column
  - post-synaptic neuron ID column
  - synapse count column
Print the column names you identified.

Step 2 — Load connections and filter:
  conn = pd.read_feather("data/real/proofread_connections_783.feather")
  edges = conn[conn[syn_col] >= 5][[pre_col, post_col, syn_col]].copy()
  edges.columns = ["pre", "post", "weight"]
  Print: number of edges after filtering, number of unique neurons.

Step 3 — Build NetworkX DiGraph:
  G = nx.from_pandas_edgelist(edges, source="pre", target="post",
                               edge_attr="weight", create_using=nx.DiGraph())
  Print: G.number_of_nodes(), G.number_of_edges(), nx.density(G)

  FALLBACK if this takes >15 minutes or raises MemoryError:
    edges_small = edges.sample(500_000, random_state=42)
    Build G from edges_small instead. Print "FALLBACK: using 500k edge sample".

Step 4 — Degree distribution plot:
  Compute in_degree and out_degree sequences (G.in_degree(), G.out_degree()).
  Plot two log-log histograms side by side (in-degree / out-degree).
  Title: "FlyWire connectome degree distribution (n={nodes} neurons)"
  Save to: data/real/fig1_degree_distribution.png
  Print: median/mean/max of in-degree and out-degree.

Step 5 — Degree assortativity:
  r = nx.degree_assortativity_coefficient(G)
  Print the value and a one-line interpretation
  (r > 0 = hubs connect to hubs; r < 0 = hubs connect to low-degree nodes).

Step 6 — Save edge list and node list for reuse:
  edges.to_csv("data/real/edges_filtered.csv", index=False)
  pd.Series(list(G.nodes), name="neuron_id").to_csv("data/real/all_nodes.csv", index=False)
  Print: "Saved edges_filtered.csv and all_nodes.csv"

EXPECTED OUTPUTS:
  data/real/fig1_degree_distribution.png
  data/real/edges_filtered.csv
  data/real/all_nodes.csv
  Console: node count, edge count, density, assortativity value
```

---

## Prompt 3 — 神经元类型分类（最容易卡的步骤）

> **目的**：用 neuropil 投射比例把每个神经元标注为 sensor / inter / motor。
> **前置条件**：Prompt 1 完成（需要看 neuropil 列名），Prompt 2 完成（需要 all_nodes.csv）。
> **预计时间**：30 秒–2 分钟。
> **卡点**：neuropil 列名和关键词不匹配 → 必须先打印所有列名，再选关键词。此步骤要求 Codex 先探索再编码，不要直接硬编码关键词。

```
You are executing Project A of a FlyWire connectome analysis project.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Data files:
  data/real/per_neuron_neuropil_count_pre_783.feather   (pre-synaptic neuropil counts per neuron)
  data/real/per_neuron_neuropil_count_post_783.feather  (post-synaptic neuropil counts per neuron)
  data/real/all_nodes.csv                               (all neuron IDs in the graph)

IMPORTANT: Do NOT hard-code neuropil column names. Follow the 3-step process below.

STEP 1 — Explore column names (REQUIRED before writing any classification code):
  np_pre  = pd.read_feather("data/real/per_neuron_neuropil_count_pre_783.feather")
  np_post = pd.read_feather("data/real/per_neuron_neuropil_count_post_783.feather")
  Print ALL column names of both files.
  Print the first 2 rows of each file.
  Print the ID column name (column 0) for each file.

STEP 2 — Select neuropil columns using what you found in Step 1:
  The neuropil columns (all except column 0) represent brain regions.
  From the post-synaptic file, find columns whose names contain any of these substrings
  (case-insensitive): AL, ME, LO, LOP, AMMC, JO, PRW, AOTU, BU
  These are SENSORY INPUT neuropils.
  From the pre-synaptic file, find columns whose names contain any of these substrings:
  VNC, GNG, PS, SAD, NP, WED, neck, motor
  These are MOTOR OUTPUT neuropils.
  Print the matched column lists. If either list is EMPTY, print a warning and use ALL
  neuropil columns for that file as a fallback (every neuron gets equal score → all inter).

STEP 3 — Classify neurons:
  id_col_pre  = np_pre.columns[0]
  id_col_post = np_post.columns[0]
  
  # Set index by neuron ID
  np_pre_idx  = np_pre.set_index(id_col_pre)
  np_post_idx = np_post.set_index(id_col_post)
  
  # Load all graph neuron IDs
  all_ids = pd.read_csv("data/real/all_nodes.csv")["neuron_id"]
  
  # Sensory score: fraction of post-synaptic inputs from sensory neuropils
  sensory_cols = [columns matched in Step 2 from post file]
  post_total   = np_post_idx[neuropil_cols_post].sum(axis=1).replace(0, np.nan)
  sens_score   = np_post_idx[sensory_cols].sum(axis=1) / post_total
  
  # Motor score: fraction of pre-synaptic outputs into motor neuropils
  motor_cols   = [columns matched in Step 2 from pre file]
  pre_total    = np_pre_idx[neuropil_cols_pre].sum(axis=1).replace(0, np.nan)
  motor_score  = np_pre_idx[motor_cols].sum(axis=1) / pre_total
  
  # Assign types with threshold 0.30
  THRESH = 0.30
  is_sensor = sens_score.reindex(all_ids).fillna(0)  >= THRESH
  is_motor  = motor_score.reindex(all_ids).fillna(0) >= THRESH
  
  node_type = pd.Series("inter", index=all_ids)
  node_type[is_motor]  = "motor"
  node_type[is_sensor] = "sensor"   # sensor takes priority over motor
  
  Print type counts and percentages.
  If sensor count < 100 OR motor count < 100, print a WARNING and try threshold=0.15.
  
STEP 4 — Save:
  node_type.reset_index().rename(columns={"index": "neuron_id", 0: "type"}).to_csv(
      "data/real/node_types.csv", index=False)
  Print: "Saved node_types.csv  sensor=X  inter=Y  motor=Z"

EXPECTED OUTPUTS:
  data/real/node_types.csv  (two columns: neuron_id, type)
  Console: exact column names found, type distribution
```

---

## Prompt 4 — FF/FB/Lateral 方向分析 + 感觉运动子图提取

> **目的**：计算突触权重的方向分布，提取感觉运动子图供后续高计算量算法使用。
> **前置条件**：Prompt 2（edges_filtered.csv），Prompt 3（node_types.csv）。
> **预计时间**：5–10 分钟（遍历所有边一次）。
> **卡点**：子图太小（< 500 节点）或太大（> 50,000 节点）→ 包含尺寸检查和调整策略。

```
You are executing Project A of a FlyWire connectome analysis project.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Input files:
  data/real/edges_filtered.csv   (columns: pre, post, weight)
  data/real/node_types.csv       (columns: neuron_id, type — values: sensor/inter/motor)

TASK A — FF/FB/Lateral direction analysis:

Step 1: Load data and rebuild graph:
  edges     = pd.read_csv("data/real/edges_filtered.csv")
  node_type = pd.read_csv("data/real/node_types.csv").set_index("neuron_id")["type"].to_dict()
  LAYER_RANK = {"sensor": 0, "inter": 1, "motor": 2}
  G = nx.from_pandas_edgelist(edges, source="pre", target="post",
                               edge_attr="weight", create_using=nx.DiGraph())

Step 2: Classify every edge and sum weights:
  Iterate G.edges(data=True). For each edge (u, v, d):
    w   = d["weight"]
    r_u = LAYER_RANK.get(node_type.get(u, "inter"), 1)
    r_v = LAYER_RANK.get(node_type.get(v, "inter"), 1)
    if r_v > r_u: feedforward += w; ff_count += 1
    elif r_v < r_u: feedback += w;  fb_count += 1
    else: lateral += w;              lat_count += 1
  Print percentages by weight AND by edge count.

Step 3: Save FF/FB plot to data/real/fig3_ff_fb_balance.png
  Bar chart: feedforward / feedback / lateral
  Colors: cornflowerblue / tomato / goldenrod
  Show percentage labels on each bar.

TASK B — Extract sensorimotor subgraph:

Step 4: Find sensorimotor inter-neurons:
  sensor_ids = set of all neurons where node_type == "sensor"
  motor_ids  = set of all neurons where node_type == "motor"
  
  inter_with_sensor_input = set of inter-neurons that receive input FROM a sensor neuron
  inter_with_motor_output = set of inter-neurons that send output TO a motor neuron
  sm_inter = inter_with_sensor_input & inter_with_motor_output
  sm_nodes = sensor_ids | motor_ids | sm_inter
  G_sm = G.subgraph(sm_nodes).copy()
  
  Print: "Sensorimotor subgraph: {nodes} nodes, {edges} edges"
  Print: "  sensor: X  inter: Y  motor: Z"
  
  SIZE CHECK:
  - If G_sm.number_of_nodes() < 200: print WARNING "Subgraph too small — check classification"
    → try expanding: include inter-neurons within 2 hops of ANY sensor or motor node
  - If G_sm.number_of_nodes() > 50_000: print WARNING "Subgraph large — betweenness will be slow"
    → sample to top-5000 nodes by degree:
      top_nodes = sorted(G_sm.degree(), key=lambda x: x[1], reverse=True)[:5000]
      G_sm = G_sm.subgraph([n for n, _ in top_nodes]).copy()
      Print "Subgraph truncated to top-5000 by degree"

Step 5: Save subgraph node list:
  sm_node_list = list(G_sm.nodes())
  pd.Series(sm_node_list, name="neuron_id").to_csv("data/real/sm_subgraph_nodes.csv", index=False)
  
  Also save SM subgraph edges:
  nx.to_pandas_edgelist(G_sm).to_csv("data/real/sm_subgraph_edges.csv", index=False)
  Print: "Saved sm_subgraph_nodes.csv and sm_subgraph_edges.csv"

EXPECTED OUTPUTS:
  data/real/fig3_ff_fb_balance.png
  data/real/sm_subgraph_nodes.csv
  data/real/sm_subgraph_edges.csv
  Console: FF/FB/lateral percentages, subgraph node/edge counts
```

---

## Prompt 5 — 图算法三件套（Betweenness + Community + GRC）

> **目的**：在感觉运动子图上运行三个高计算量算法。
> **前置条件**：Prompt 4（sm_subgraph_nodes.csv + sm_subgraph_edges.csv + node_types.csv）。
> **预计时间**：20 分钟–1.5 小时（视子图大小）。
> **卡点**：最耗时的一步。每个算法独立保存结果，任何一个超时可以单独重跑；包含明确的降级参数。

```
You are executing Project A of a FlyWire connectome analysis project.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Input files:
  data/real/sm_subgraph_nodes.csv    (sensorimotor subgraph neuron IDs)
  data/real/sm_subgraph_edges.csv    (sensorimotor subgraph edges: source, target, weight)
  data/real/node_types.csv           (neuron_id → sensor/inter/motor)

SETUP — Rebuild subgraph (do this first):
  sm_nodes = pd.read_csv("data/real/sm_subgraph_nodes.csv")["neuron_id"].tolist()
  sm_edges = pd.read_csv("data/real/sm_subgraph_edges.csv")
  node_type = pd.read_csv("data/real/node_types.csv").set_index("neuron_id")["type"].to_dict()
  
  G_sm = nx.from_pandas_edgelist(sm_edges, source="source", target="target",
                                  edge_attr="weight", create_using=nx.DiGraph())
  Print: f"G_sm loaded: {G_sm.number_of_nodes()} nodes, {G_sm.number_of_edges()} edges"

--- ALGORITHM A: Betweenness Centrality ---
Run in isolation. Save result before starting next algorithm.

  import time
  t0 = time.time()
  print("Starting betweenness centrality (k=300)...")
  
  k_val = 300
  if G_sm.number_of_nodes() > 20_000:
      k_val = 100   # reduce for very large subgraphs
      print(f"Large subgraph: using k={k_val}")
  
  bc = nx.betweenness_centrality(G_sm, normalized=True, weight="weight", k=k_val, seed=42)
  print(f"Done in {time.time()-t0:.0f}s")
  
  bc_df = pd.DataFrame({
      "neuron_id": list(bc.keys()),
      "betweenness": list(bc.values()),
      "type": [node_type.get(n, "inter") for n in bc.keys()]
  })
  bc_df.to_csv("data/real/betweenness_results.csv", index=False)
  Print top-10 bottleneck neurons (sorted by betweenness).
  
  Plot betweenness histogram by type:
    For each type: hist of betweenness values, log y-scale, alpha=0.7
    Save: data/real/fig5_betweenness.png

--- ALGORITHM B: Community Detection (Louvain) ---
Run after Algorithm A. If A is still running, skip and note in output.

  import time
  G_sm_undir = G_sm.to_undirected()
  t0 = time.time()
  print("Starting Louvain community detection...")
  
  communities = nx.community.louvain_communities(G_sm_undir, seed=42, weight="weight")
  print(f"Done in {time.time()-t0:.0f}s — {len(communities)} communities found")
  
  # Assign community IDs
  node_to_comm = {}
  for cid, comm in enumerate(communities):
      for n in comm: node_to_comm[n] = cid
  
  comm_df = pd.DataFrame({
      "neuron_id": list(node_to_comm.keys()),
      "community": list(node_to_comm.values()),
      "type": [node_type.get(n, "inter") for n in node_to_comm.keys()]
  })
  comm_df.to_csv("data/real/community_results.csv", index=False)
  
  comm_sizes = sorted([len(c) for c in communities], reverse=True)
  Print: number of communities, top-5 sizes, size of smallest community.
  Print: for each of the 5 largest communities, the fraction of sensor/inter/motor neurons.
  
  Plot community size histogram, log x-scale.
  Save: data/real/fig6_communities.png

--- ALGORITHM C: Global Reaching Centrality ---
Run after Algorithm B. This may be the slowest.

  import time
  t0 = time.time()
  print("Starting Global Reaching Centrality...")
  
  FALLBACK_TRIGGERED = False
  
  try:
      # Set a soft timeout check: if subgraph > 10k nodes, warn but proceed
      if G_sm.number_of_nodes() > 10_000:
          print(f"WARNING: {G_sm.number_of_nodes()} nodes — GRC may take 20–60 min")
      grc = nx.global_reaching_centrality(G_sm, weight="weight", normalized=True)
      print(f"GRC = {grc:.4f}  (computed in {time.time()-t0:.0f}s)")
  except Exception as e:
      print(f"GRC failed: {e}")
      print("FALLBACK: using feedforward fraction as hierarchy proxy")
      # Read FF fraction from the fig3 bar chart data instead
      # (user must manually enter ff_fraction from Prompt 4 output)
      grc = None
      FALLBACK_TRIGGERED = True
  
  # Save GRC value
  with open("data/real/grc_result.txt", "w") as f:
      f.write(f"GRC: {grc}\n")
      f.write(f"Fallback used: {FALLBACK_TRIGGERED}\n")
      f.write(f"Nodes in subgraph: {G_sm.number_of_nodes()}\n")
  Print: GRC value and interpretation.

EXPECTED OUTPUTS (each saved independently so partial runs are recoverable):
  data/real/betweenness_results.csv
  data/real/community_results.csv
  data/real/grc_result.txt
  data/real/fig5_betweenness.png
  data/real/fig6_communities.png
```

---

## Prompt 6 — 可视化总结 + 控制器先验报告

> **目的**：汇总所有结果，生成 4-panel 总结图，输出控制器先验规格文档。
> **前置条件**：Prompt 2–5 的所有 CSV 和 txt 结果文件存在。
> **预计时间**：5–10 分钟。
> **卡点**：某个 CSV 不存在（某个 Prompt 没跑完）→ 用 try/except 跳过缺失数据，仍然生成其他 panel。

```
You are executing the final step of Project A of a FlyWire connectome analysis project.

Working directory: /Volumes/SANDISK ELE/embodied summer program
Input files (all from previous steps):
  data/real/edges_filtered.csv
  data/real/node_types.csv
  data/real/betweenness_results.csv
  data/real/community_results.csv
  data/real/grc_result.txt
  (fig1–fig6 already saved as individual PNG files)

TASK A — 4-panel summary figure:

Create a 2×2 subplot figure (figsize=(14, 10)):

Panel A (top-left): In-degree distribution by type
  Load node_types and edges_filtered.
  Compute in-degree per neuron: in_deg = edges_filtered.groupby("post")["weight"].sum()
  For each type in [sensor, inter, motor]:
    plot histogram of in_deg values for that type, log y-scale, alpha=0.6
  Title: "A. In-degree distribution by neuron type"
  Colors: sensor=#4c9be8, inter=#f5a623, motor=#4caf50

Panel B (top-right): FF / FB / Lateral bar chart
  Read the three values from what was computed in Prompt 4.
  If you don't have them, re-compute from edges_filtered.csv + node_types.csv (same logic).
  Bar chart with percentage labels. Colors: cornflowerblue, tomato, goldenrod.
  Title: "B. FF / FB / Lateral balance"

Panel C (bottom-left): Betweenness centrality distribution
  Load betweenness_results.csv.
  Histogram of betweenness per type, log y-scale.
  Title: "C. Betweenness centrality (sensorimotor subgraph)"
  If betweenness_results.csv missing: show placeholder text "Pending"

Panel D (bottom-right): Community size distribution
  Load community_results.csv, compute community sizes.
  Histogram, log x-scale.
  Title: "D. Community sizes"
  If community_results.csv missing: show placeholder text "Pending"

Super-title: "FlyWire Whole-Brain Connectome — Structural Analysis (v783)"
Save: data/real/fig0_summary.png   dpi=150

TASK B — Controller-prior specification report:

Read all result files and generate data/real/controller_prior_report.txt with this structure:

=== PROJECT A: CONTROLLER PRIOR REPORT ===
Generated: [timestamp]

GRAPH SCALE
  Full graph nodes      : [from edges_filtered unique neurons]
  Full graph edges      : [from edges_filtered row count]
  Sensorimotor subgraph : [from sm_subgraph_nodes row count] nodes

HIERARCHY
  GRC (sensorimotor)    : [from grc_result.txt, or "see FF fraction"]
  FF weight fraction    : [X%]
  FB weight fraction    : [X%]
  Lateral fraction      : [X%]
  Interpretation        : [1-line based on values]

COMMUNITY
  Number of communities : [from community_results]
  Largest community     : [size]
  Smallest community    : [size]

BOTTLENECK HUBS
  Top-5 betweenness neurons:
    [neuron_id] [type] [betweenness]  ← from betweenness_results.csv

CONTROLLER ARCHITECTURE SPECIFICATION FOR PROJECT B:
  Message-passing depth         : 2 (sensor→inter→motor)
  Hub node embedding dim        : 2× wider than peripheral nodes
  Feedback edges                : implement as residual skip-connections
  Lateral recurrent edges       : keep in adjacency (implicit temporal memory)
  Number of output modules      : [= number of large communities, capped at 6]
  Rationale for GNN over MLP    : [1–2 sentences based on FF/FB/GRC values]

Save this file, then print its full contents.

TASK C — Verify all outputs:
List all files in data/real/ and note which of the 7 expected figures exist.
Print a checklist:
  [x] fig0_summary.png
  [x/missing] fig1_degree_distribution.png
  ... etc
Print "Project A complete." if all 7 figures present.
Print "Project A partial — missing: [list]" otherwise.
```

---

## 执行顺序和依赖关系

```
Prompt 1  ──────────────────────────────────────┐
(下载 + schema)                                  ↓
                                          Prompt 3 (分类)
Prompt 2  ─────────────────────────────────────→ ↓
(建图 + 度分布)                           Prompt 4 (方向 + 子图)
                                                  ↓
                                          Prompt 5 (算法三件套)
                                                  ↓
                                          Prompt 6 (总结 + 报告)
```

P1 和 P2 可以 overlap（P2 的建图在 P1 下载时跑）。  
P3 必须在 P1 完成后才能跑（需要看到真实列名）。  
P4 必须在 P2 + P3 完成后跑。  
P5 必须在 P4 完成后跑。  
P6 必须在 P5 完成后跑（但缺少某些 P5 结果时可以部分运行）。
