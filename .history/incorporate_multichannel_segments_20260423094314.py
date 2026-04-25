import pandas as pd
import networkx as nx
from collections import defaultdict
import math
import numpy as np
import ast 

## ---------------------------------------------------------------------
# Turn dataframe into Graph
## ---------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

# def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
#     d = df[["reach_id", "rch_id_dn"]].copy()
#     d = d.dropna(subset=["rch_id_dn"])
#     d = d.explode("rch_id_dn").rename(columns={"rch_id_dn": "dn_id"})
#     d = d.dropna(subset=["dn_id"])
#     return d[["reach_id", "dn_id"]]

def make_pairs(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.filter(regex=r'^rch_id_dn_(?!m)').columns

    out = (
        df
        .melt(id_vars='reach_id', value_vars=cols, var_name='source', value_name='dn_id')
        .loc[lambda d: d['dn_id'].ne(0)]                 # drop zero inputs first
        .assign(diff=lambda d: d['reach_id'] - d['dn_id'])
        .loc[lambda d: d['diff'].ne(0), ['reach_id', 'dn_id']]
        )
    return out[["reach_id", "dn_id"]]

def reaches_to_junction_graph(df: pd.DataFrame) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    pairs = make_pairs(df)
    uf = UnionFind()

    # 1) Union downstream endpoint of reach with upstream endpoint of its downstream reach
    for r, dn in pairs.itertuples(index=False):
        uf.union(("D", r), ("U", dn))

    # 2) Create graph with UF representative nodes (temporary ids)
    Gtmp = nx.MultiDiGraph()

    # ensure endpoints exist
    for r in df["reach_id"].unique():
        uf.find(("U", r))
        uf.find(("D", r))

    df_grouped = df.groupby("reach_id", sort=False)

    # add edges (each reach is an edge)
    for r in df["reach_id"].unique():
        row = df_grouped.get_group(r).iloc[0]
        u = uf.find(("U", r))
        v = uf.find(("D", r))
        Gtmp.add_edge(
            u, v, key=row["reach_id"], **row.to_dict())

    # 3) Build readable junction labels from incident reach_ids
    incident = {n: set() for n in Gtmp.nodes()}
    for u, v, k, d in Gtmp.edges(keys=True, data=True):
        rid = d.get("reach_id", k)
        incident[u].add(rid)
        incident[v].add(rid)

    # base label = "1-2-7" (sorted)
    base_label = {}
    for n, rset in incident.items():
        parts = sorted(map(str, rset))
        base_label[n] = "-".join(parts) if parts else str(n)

    # disambiguate collisions: same incident set can happen in rare cases
    seen = defaultdict(int)
    mapping = {}
    for old, lab in base_label.items():
        seen[lab] += 1
        mapping[old] = lab if seen[lab] == 1 else f"{lab}_{seen[lab]}"

    # 4) Relabel nodes and return final graph
    G = nx.relabel_nodes(Gtmp, mapping, copy=True)
    return G, mapping

## ---------------------------------------------------------------------
# Filter Graph for main and side channels
## ---------------------------------------------------------------------
def edge_filtered_subgraph(G, pred):
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        # print(d, pred(d))
        if pred(d):
            H.add_edge(u, v, key=k, **d)
    return H

## ---------------------------------------------------------------------
# Idenitfy components in side channels including starting and ending nodes
## ---------------------------------------------------------------------
def component_edges(G: nx.MultiDiGraph, comp_nodes: set):
    """Return list of (u,v,k,d) edges fully inside comp_nodes."""
    return [(u, v, k, d)
            for u, v, k, d in G.edges(keys=True, data=True)
            if u in comp_nodes and v in comp_nodes]

def component_edge_count(G: nx.MultiDiGraph, comp_nodes: set) -> int:
    return len(component_edges(G, comp_nodes))

def side_components(Gside: nx.MultiDiGraph, G:nx.MultiDiGraph) -> list[dict]:
    """
    Returns a list of node-sets, each node-set is one weakly connected component
    considering only side edges.
    """
    # Make a simple directed graph with same nodes and side edges (keys don’t matter for connectivity)
    H = nx.DiGraph()
    H.add_nodes_from(Gside.nodes())
    H.add_edges_from((u, v) for u, v, k in Gside.edges(keys=True))

    first_pass_components = [set(c) for c in nx.weakly_connected_components(H)]
    # print(first_pass_components)
    comps = []
    for comp in first_pass_components:
        m = component_edge_count(G, comp)
        if m >= 2:
            comps.append(comp)
            # print(len(comp), "nodes,", m, "side-edges")
    return comps

## ---------------------------------------------------------------------
# Get nodes and node order in main channel Graph
## ---------------------------------------------------------------------
def get_main_order(Gmain: nx.MultiDiGraph):
    """
    Returns:
        main_nodes : list of nodes in downstream order along the mainstem
        main_idx   : dict mapping node -> order index
    """

    # source = node with no incoming main edges
    sources = [n for n in Gmain.nodes() if Gmain.in_degree(n) == 0 and Gmain.out_degree(n) > 0]

    # sink = node with no outgoing main edges
    sinks = [n for n in Gmain.nodes() if Gmain.out_degree(n) == 0 and Gmain.in_degree(n) > 0]
    print(Gmain.nodes(), source, sinks)
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError(
            f"Gmain not a single path (sources={sources}, sinks={sinks})."
        )

    s, t = sources[0], sinks[0]

    # collapse multiedges for path calculation
    H = nx.DiGraph()
    H.add_nodes_from(Gmain.nodes())
    H.add_edges_from((u, v) for u, v, k in Gmain.edges(keys=True))

    main_nodes = nx.shortest_path(H, s, t)

    main_idx = {n: i for i, n in enumerate(main_nodes)}

    return main_nodes, main_idx



## ---------------------------------------------------------------------
# Compute average channel and width addition for multi channel sections
## ---------------------------------------------------------------------

# Small helpers
def edge_filtered_subgraph(G: nx.MultiDiGraph, pred):
    H = nx.MultiDiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u, v, k, d in G.edges(keys=True, data=True):
        if pred(d):
            H.add_edge(u, v, key=k, **d)
    return H

def corridor_subgraph(G: nx.MultiDiGraph, source, sink) -> nx.MultiDiGraph:
    """Directed corridor: nodes reachable from source AND that can reach sink."""
    fwd = nx.descendants(G, source) | {source}
    bwd = nx.ancestors(G, sink) | {sink}
    nodes = fwd & bwd
    return G.subgraph(nodes).copy()

def build_reach_graph_from_junction_graph(Gj: nx.MultiDiGraph) -> nx.DiGraph:
    """
    Junction graph (nodes=junctions, edges=reaches) -> reach graph (nodes=reaches).
    """
    R = nx.DiGraph()
    in_reaches = defaultdict(list)
    out_reaches = defaultdict(list)

    for u, v, k, d in Gj.edges(keys=True, data=True):
        rid = d.get("reach_id", k)
        if rid not in R:
            R.add_node(rid, **d)
        out_reaches[u].append(rid)  # reach starts at u
        in_reaches[v].append(rid)   # reach ends at v

    for junc in Gj.nodes():
        preds = in_reaches.get(junc, [])
        succs = out_reaches.get(junc, [])
        for rin in preds:
            for rout in succs:
                if rin != rout:
                    R.add_edge(rin, rout)
    return R

def bubble_start_end_reaches(Gj: nx.MultiDiGraph, source, sink):
    starts = [d.get("reach_id", k) for _, _, k, d in Gj.out_edges(source, keys=True, data=True)]
    ends   = [d.get("reach_id", k) for _, _, k, d in Gj.in_edges(sink, keys=True, data=True)]
    starts = list(dict.fromkeys(starts))
    ends   = list(dict.fromkeys(ends))
    return starts, ends

def enumerate_reach_paths(R: nx.DiGraph, starts, ends, cutoff=200, max_paths=5000):
    paths = []
    for a in starts:
        for b in ends:
            if a not in R or b not in R:
                continue
            for p in nx.all_simple_paths(R, a, b, cutoff=cutoff):
                paths.append(p)
                if len(paths) >= max_paths:
                    return paths
    return paths

def path_score(R: nx.DiGraph, reach_path, length_attr="reach_len", width_attr=None):
    s = 0.0
    for rid in reach_path:
        d = R.nodes[rid]
        L = float(d.get(length_attr, 1.0) or 1.0)
        if width_attr is None:
            s += L
        else:
            W = float(d.get(width_attr, 1.0) or 1.0)
            s += L * W
    return s

def path_length(R: nx.DiGraph, reach_path, length_attr="reach_len"):
    return sum(float(R.nodes[rid].get(length_attr, 1.0) or 1.0) for rid in reach_path)

def reach_width_lookup(R: nx.DiGraph, width_attr="width"):
    return {rid: (None if d.get(width_attr) is None else float(d.get(width_attr))) 
            for rid, d in R.nodes(data=True)}

def summarize_width(reach_ids, width_of, len_of=None, stat="median", weighted=False):
    """
    reach_ids: iterable of reach_id
    width_of: dict reach_id -> width (float)
    len_of: dict reach_id -> length (float), required if weighted=True
    stat: "median" or "mean"
    weighted: if True, computes length-weighted mean (stat must be "mean")
    """
    vals = [(rid, width_of.get(rid)) for rid in reach_ids]
    vals = [(rid, float(w)) for rid, w in vals if w is not None and not np.isnan(w)]

    if not vals:
        return float("nan")

    if weighted:
        if stat != "mean":
            raise ValueError("weighted=True only supported for stat='mean'")
        if len_of is None:
            raise ValueError("len_of required for weighted mean")
        num = 0.0
        den = 0.0
        for rid, w in vals:
            L = float(len_of.get(rid, 0.0))
            if L > 0:
                num += w * L
                den += L
        return (num / den) if den > 0 else float("nan")

    ws = np.array([w for _, w in vals], dtype=float)
    if stat == "median":
        return float(np.median(ws))
    if stat == "mean":
        return float(np.mean(ws))
    raise ValueError("stat must be 'median' or 'mean'")

def reach_length_lookup(R: nx.DiGraph, length_attr="reach_len"):
    return {rid: float(d.get(length_attr, 1.0) or 1.0) for rid, d in R.nodes(data=True)}

# k_max via rescaled layer width (your latest method)
def k_max_layer_width_rescaled(
    G: nx.MultiDiGraph,
    source,
    sink,
    main_attr: str = "is_mainstem_edge",
    restrict_to_st_corridor: bool = True,
    main_weight: float | None = None,
    return_debug: bool = False,
    ):
    """
    k_max = max #edges crossing any downstream slice, using weighted dist that
    stretches main edges so main and side have comparable 'depth'.
    """

    # 0) corridor
    if restrict_to_st_corridor:
        fwd = nx.descendants(G, source) | {source}
        bwd = nx.ancestors(G, sink) | {sink}
        nodes = fwd & bwd
        Hm = G.subgraph(nodes).copy()
    else:
        Hm = G

    if source not in Hm or sink not in Hm or Hm.number_of_edges() == 0:
        return (0, {}) if return_debug else 0

    # 1) simple DiGraph for topo/DP with is_main per (u,v)
    Hd = nx.DiGraph()
    Hd.add_nodes_from(Hm.nodes())
    for u, v, k, d in Hm.edges(keys=True, data=True):
        is_main = bool(d.get(main_attr, False))
        if Hd.has_edge(u, v):
            Hd[u][v]["is_main"] = Hd[u][v]["is_main"] or is_main
        else:
            Hd.add_edge(u, v, is_main=is_main)

    if not nx.is_directed_acyclic_graph(Hd):
        raise ValueError("k_max_layer_width_rescaled requires a DAG corridor.")

    topo = list(nx.topological_sort(Hd))

    # helper: longest hop depth in a given edge-filtered DiGraph
    def longest_hop_depth(H: nx.DiGraph) -> float:
        dist = {n: -math.inf for n in H.nodes()}
        dist[source] = 0.0
        for u in topo:
            if dist.get(u, -math.inf) == -math.inf:
                continue
            du = dist[u]
            for v in H.successors(u):
                if du + 1.0 > dist[v]:
                    dist[v] = du + 1.0
        return dist.get(sink, -math.inf)

    if main_weight is None:
        Hd_side = nx.DiGraph()
        Hd_side.add_nodes_from(Hd.nodes())
        Hd_side.add_edges_from((u, v) for u, v, data in Hd.edges(data=True) if not data["is_main"])

        Hd_main = nx.DiGraph()
        Hd_main.add_nodes_from(Hd.nodes())
        Hd_main.add_edges_from((u, v) for u, v, data in Hd.edges(data=True) if data["is_main"])

        depth_side = longest_hop_depth(Hd_side)
        depth_main = longest_hop_depth(Hd_main)

        if depth_side == -math.inf or depth_main == -math.inf or depth_side <= 0 or depth_main <= 0:
            main_weight = 1.0
        else:
            main_weight = float(depth_side) / float(depth_main)

    # 3) weighted dist
    dist = {n: -math.inf for n in Hd.nodes()}
    dist[source] = 0.0

    for u in topo:
        if dist[u] == -math.inf:
            continue
        du = dist[u]
        for v in Hd.successors(u):
            w = main_weight if Hd[u][v]["is_main"] else 1.0
            cand = du + w
            if cand > dist[v]:
                dist[v] = cand

    if dist[sink] == -math.inf:
        return (0, {}) if return_debug else 0

    # 4) count MultiDiGraph edges crossing integer slices
    Lmax = int(math.floor(dist[sink]))
    if Lmax <= 0:
        return (0, {}) if return_debug else 0

    width = [0] * Lmax
    for u, v, k in Hm.edges(keys=True):
        du = dist.get(u, -math.inf)
        dv = dist.get(v, -math.inf)
        if du == -math.inf or dv == -math.inf or dv <= du:
            continue

        a = int(math.floor(du))
        b = int(math.ceil(dv)) - 1
        a = max(a, 0)
        b = min(b, Lmax - 1)
        for i in range(a, b + 1):
            if du <= i < dv:
                width[i] += 1

    kmax = max(width) if width else 0

    if not return_debug:
        return kmax

    debug = {
        "main_weight": main_weight,
        "dist": dist,
        "width_profile": width,
        "kmax_slice": int(max(range(len(width)), key=lambda i: width[i])) if width else None,
        "corridor_nodes": Hm.number_of_nodes(),
        "corridor_edges": Hm.number_of_edges(),
    }
    return kmax, debug

# Main reach ids and lengths between source->sink
def main_reach_ids_between(Gmain: nx.MultiDiGraph, s, t):
    """
    Main assumed single path. Returns (reach_id_set, node_path).
    reach_id pulled from edge attr 'reach_id' else key.
    """
    Hm = Gmain.edge_subgraph(Gmain.edges(keys=True)).copy()
    H = nx.DiGraph()
    H.add_nodes_from(Hm.nodes())
    H.add_edges_from((u, v) for u, v, k in Hm.edges(keys=True))

    node_path = nx.shortest_path(H, s, t)

    rids = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        any_k = next(iter(Gmain[u][v].keys()))
        d = Gmain[u][v][any_k]
        rids.append(d.get("reach_id", any_k))

    return set(rids), node_path

def main_path_length_from_set(len_of: dict, main_set: set):
    return sum(len_of.get(r, 0.0) for r in main_set)

# Choose source/sink per component (uses side edges only for split/merge capability)
def choose_source_sink_for_component(Gside_sub: nx.MultiDiGraph, component_nodes: set, main_nodes: list, main_idx: dict):
    main_set = set(main_nodes)
    touched = [n for n in component_nodes if n in main_set and n in main_idx]
    touched.sort(key=lambda n: main_idx[n])

    splits = [m for m in touched if Gside_sub.out_degree(m) > 0]
    merges = [m for m in touched if Gside_sub.in_degree(m) > 0]

    if not splits or not merges:
        return None, None, {"touched_main": touched, "splits": splits, "merges": merges}

    source = max(splits, key=lambda n: main_idx[n])
    merges_ds = [m for m in merges if main_idx[m] > main_idx[source]]
    if not merges_ds:
        return None, None, {"touched_main": touched, "splits": splits, "merges": merges}

    sink = max(merges_ds, key=lambda n: main_idx[n])
    return source, sink, {"touched_main": touched, "splits": splits, "merges": merges}

# Dominant SIDE-only path (so dom does not include main reaches)
def dominant_side_path_reach_ids(
    Gcorr: nx.MultiDiGraph,
    source, sink,
    length_attr="reach_len",
    width_attr=None,
    cutoff=200,
    max_paths=5000,
    fallback_undirected=True,
    main_attr="is_mainstem_edge",
    ):
    """
    Finds dominant path on SIDE-ONLY subgraph of the corridor.
    Returns (dom_side_set, L_dom, dom_path_list, n_paths).
    """
    Gside_corr = edge_filtered_subgraph(Gcorr, lambda d: not d.get(main_attr, False))
    Rside = build_reach_graph_from_junction_graph(Gside_corr)

    starts, ends = bubble_start_end_reaches(Gside_corr, source, sink)
    paths = enumerate_reach_paths(Rside, starts, ends, cutoff=cutoff, max_paths=max_paths)

    if not paths and fallback_undirected:
        Ru = Rside.to_undirected()
        paths = []
        for a in starts:
            for b in ends:
                if a not in Ru or b not in Ru:
                    continue
                for p in nx.all_simple_paths(Ru, a, b, cutoff=cutoff):
                    paths.append(p)
                    if len(paths) >= max_paths:
                        break

    if not paths:
        return set(), 0.0, None, 0, {"starts": starts, "ends": ends}

    dom = max(paths, key=lambda p: path_score(Rside, p, length_attr=length_attr, width_attr=width_attr))
    dom_set = set(dom)
    L_dom = path_length(Rside, dom, length_attr=length_attr)

    return dom_set, L_dom, dom, len(paths), {"starts": starts, "ends": ends}

# Final: analyze one bubble per side component, with corrected L_extra
def analyze_component_total_channels(
    G: nx.MultiDiGraph,
    Gmain: nx.MultiDiGraph,
    Gside: nx.MultiDiGraph,
    component_nodes: set,
    main_nodes: list,
    main_idx: dict,
    length_attr="reach_len",
    width_attr=None,
    Lref_mode="main",        # "main" or "dom"
    min_extra=10.0,          # meters; if L_extra <= this, treat as no extra
    cutoff=200,
    max_paths=5000,
    fallback_undirected=True,
    main_attr="is_mainstem_edge",
    return_kmax_debug=False,
    df_node = None
    ):
    """
    One result per side component.

    - source/sink chosen from side-edge split/merge capability.
    - corridor subgraph built on FULL G between source and sink.
    - k_max computed with rescaled layer-width method on FULL corridor.
    - dominant path computed on SIDE-ONLY corridor.
    - L_extra computed as reach lengths not covered by (main_set ∪ dom_side_set).
    - k_eff = min(2 + (L_extra/L_ref if L_extra>min_extra), k_max)
    """
    # 1) choose endpoints using side subgraph
    Gside_sub = Gside.subgraph(component_nodes).copy()
    source, sink, meta = choose_source_sink_for_component(Gside_sub, component_nodes, main_nodes, main_idx)
    if source is None or sink is None:
        return None

    # 2) full corridor on G
    Gcorr = corridor_subgraph(G, source, sink)
    if Gcorr.number_of_edges() == 0:
        return None

    # 3) k_max on full corridor (your method)
    if return_kmax_debug:
        k_max, kdbg = k_max_layer_width_rescaled(
            Gcorr, source, sink,
            main_attr=main_attr,
            restrict_to_st_corridor=True,
            return_debug=True
        )
    else:
        k_max = k_max_layer_width_rescaled(
            Gcorr, source, sink,
            main_attr=main_attr,
            restrict_to_st_corridor=True,
            return_debug=False
        )
        kdbg = None

    # reach graph on full corridor (for total lengths by reach_id)
    Rcorr = build_reach_graph_from_junction_graph(Gcorr)
    len_of = reach_length_lookup(Rcorr, length_attr=length_attr)
    L_total = sum(len_of.values())

    # 4) main set + L_main
    main_set, main_node_path = main_reach_ids_between(Gmain, source, sink)
    L_main = main_path_length_from_set(len_of, main_set)

    # 5) dominant SIDE-only set + L_dom
    dom_side_set, L_dom, dom_side_path, n_paths, dom_meta = dominant_side_path_reach_ids(
        Gcorr, source, sink,
        length_attr=length_attr,
        width_attr=width_attr,
        cutoff=cutoff,
        max_paths=max_paths,
        fallback_undirected=fallback_undirected,
        main_attr=main_attr
    )

    # If no side path exists, you can't form a 2-channel braid bubble; bail or set k_eff=1
    if not dom_side_set or L_dom <= 0:
        return {
            "source": source,
            "sink": sink,
            "k_max": k_max,
            "k_eff": 1.0,   # only main effectively
            "reason": "No directed side path found (dominant side missing)",
            "L_main": L_main,
            "L_dom": L_dom,
            "L_total": L_total,
            "dominant_side_reach_ids": [],
            "extra_reach_ids": [],
            "main_node_path": main_node_path,
            **meta,
            **dom_meta,
            **({"kmax_debug": kdbg} if return_kmax_debug else {}),
        }

    # 6) corrected L_extra using covered reach_ids
    covered = set(main_set) | set(dom_side_set)
    extra_rid = [rid for rid, L in len_of.items() if rid not in covered]

    L_extra = sum(L for rid, L in len_of.items() if rid not in covered)

    # 7) L_ref choice
    if Lref_mode == "main":
        L_ref = L_main
    elif Lref_mode == "dom":
        L_ref = L_dom
    else:
        raise ValueError("Lref_mode must be 'main' or 'dom'")

    # 8) k_eff
    dom_side_nchan = df_node.loc[df_node['reach_id'].isin(dom_side_set), 'n_chan_mod'].mean()
    extra_nchan    = df_node.loc[df_node['reach_id'].isin(extra_rid), 'n_chan_mod'].mean()

    L_frac = (L_extra / L_ref) if (L_extra > min_extra and L_ref > 0) else 0.0
    if L_frac <= (float(k_max)-2): 
        frac = L_frac
        extra_nchan = 0
    else:
        frac = k_max / L_frac
    k_eff_raw = dom_side_nchan + (L_frac*extra_nchan)    
    k_eff     = dom_side_nchan + (frac  *extra_nchan)    




    width_of = reach_width_lookup(Rcorr, width_attr="width")  # change to your column name
    W_dom  = summarize_width(dom_side_set, width_of, len_of=len_of, stat="median", weighted=False)
    # print(W_dom)
    extra_set = set(len_of.keys()) - covered
    if len(extra_set) == 0:
        W_extra = 0
    else:
        W_extra = summarize_width(extra_set, width_of, len_of=len_of, stat="median", weighted=False)
        # same fraction used for channel count
        frac_extra_channels = (L_extra / L_ref) if (L_extra > min_extra and L_ref > 0) else 0.0
        frac_extra_channels = min(frac_extra_channels, max(0.0, float(k_max) - 2.0)) # cap to max extra channels

        W_extra = frac_extra_channels * W_extra

    W_add = W_dom + W_extra

    return {
        "source": source,
        "sink": sink,
        "k_max": k_max,
        "k_eff": k_eff,
        "k_eff_raw": k_eff_raw,
        "Lref_mode": Lref_mode,
        "L_ref": L_ref,
        "L_total": L_total,
        "L_main": L_main,
        "L_dom": L_dom,
        "L_extra": L_extra,
        'width_extra':W_add,
        "min_extra": min_extra,
        "dominant_side_reach_path": dom_side_path,
        "dominant_side_reach_ids": sorted(dom_side_set),
        "main_reach_ids": sorted(main_set),
        "extra_reach_ids": sorted(set(len_of.keys()) - covered),
        "n_side_paths_enumerated": n_paths,
        "main_node_path": main_node_path,
        "corridor_nodes": Gcorr.number_of_nodes(),
        "corridor_edges": Gcorr.number_of_edges(),
        **meta,
        **dom_meta,
        **({"kmax_debug": kdbg} if return_kmax_debug else {}),
    }

def run_code(dfG, mpi, df_node):
    # print('Current code expects rch_id_dn column to be made up of: "[]" for missing values an empty list.\
    #       If different dataframe is supplied change first lines of run_code()')

    # filter input dataframe
    D       = dfG.loc[(dfG['main_path_id'] == mpi)].copy()
    df_node = df_node[df_node['reach_id'].isin(D['reach_id'].to_list())]

    # D["rch_id_dn"] = D["rch_id_dn"].apply(ast.literal_eval)

    # Create whole, main and side graph
    G, pairs  = reaches_to_junction_graph(D)
    Gmain     = edge_filtered_subgraph(G, lambda d: d.get("is_mainstem_edge", False))
    Gside     = edge_filtered_subgraph(G, lambda d: not d.get("is_mainstem_edge", False))

    # Compute side channel components
    comps = side_components(Gside, G)

    # Get ordering main channel
    main_nodes, main_idx = get_main_order(Gmain)

    # compute additional number of channels and widths for side channel componentns
    results = []
    for i, comp in enumerate(comps):  # comps = side_components(Gside)
        res = analyze_component_total_channels(
            G=G,
            Gmain=Gmain,
            Gside=Gside,
            component_nodes=set(comp),
            main_nodes=main_nodes,
            main_idx=main_idx,
            length_attr="reach_len",
            width_attr=None,
            Lref_mode="main",     # or "dom"
            min_extra=10.0,
            cutoff=200,
            max_paths=5000,
            fallback_undirected=True,
            main_attr="is_mainstem_edge",
            return_kmax_debug=False,
            df_node=df_node
        )
        if res is not None:
            results.append(res)
        else:
            print(f'Component {i} is not calculated correctly')

    cols = ['k_eff', 'k_max', 'width_extra', 'k_eff_raw']
    for res in results:
        D.loc[D['reach_id'].isin(res['main_reach_ids']), cols] = [
                res[c] for c in cols]
    if len(results) == 0:
        D[cols] = np.nan

    D       = D[D['is_mainstem_edge'] == True]
    df_node = df_node[df_node['reach_id'].isin(D['reach_id'])]

    nchanNode = df_node.groupby('reach_id', as_index = False)['n_chan_mod'].mean()
    nchanNode = nchanNode.rename(columns = {'n_chan_mod':'multi_n_chan'})



    D = D.merge(nchanNode, how = 'left', on = 'reach_id')
    D['multi_n_chan'] = D[["multi_n_chan", "k_eff"]].sum(axis=1) 
    D['multi_width']  = D[["width", "width_extra"]].sum(axis=1)

    DN = df_node[df_node['reach_id'].isin(D['reach_id'].to_list())]
    DN = DN.merge(D[['reach_id', 'k_eff', 'width_extra']], how = 'left', on = 'reach_id')
    DN['multi_n_chan'] = DN[["n_chan_mod", "k_eff"]].sum(axis=1) 
    DN['multi_width']  = DN[["width"     , "width_extra"]].sum(axis=1)
    return D, DN