import re
import heapq
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Set, Union, Iterable

# yEd / GraphML namespaces
NS = {
    "g": "http://graphml.graphdrawing.org/xmlns",
    "y": "http://www.yworks.com/xml/graphml",
}

LABEL_RE = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*$")

# ---------- Helpers to read weights robustly ----------

def _build_edge_keymaps(root: ET.Element) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return (name_to_id, id_to_name) for edge keys, so we can read by attr.name or by key id.
    """
    name_to_id: Dict[str, str] = {}
    id_to_name: Dict[str, str] = {}
    for k in root.findall("g:key", NS):
        if k.attrib.get("for") != "edge":
            continue
        kid = k.attrib.get("id")
        aname = k.attrib.get("attr.name")
        if kid:
            if aname:
                name_to_id[aname] = kid
            id_to_name[kid] = aname or ""
    return name_to_id, id_to_name

def _edge_data_lookup(edge: ET.Element, *, name_to_id: dict,
                      candidates_by_name: Iterable[str],
                      candidates_by_id: Iterable[str]) -> Optional[str]:
    """
    Read <data> by trying attr.name first (mapped to ids), then by explicit key ids.
    Returns raw text or None.
    """
    candidate_ids = [name_to_id[n] for n in candidates_by_name if n in name_to_id]
    candidate_ids += list(candidates_by_id)
    for d in edge.findall("g:data", NS):
        if d.attrib.get("key") in candidate_ids:
            return (d.text or "").strip() if d.text else None
    return None

def _to_float(txt: Optional[str]) -> Optional[float]:
    if txt is None or txt == "":
        return None
    try:
        return float(txt)
    except Exception:
        return None

# ---------- Graph parsing ----------

def _parse_nodes(root: ET.Element) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Return:
      node_id_to_label: node_id -> "row,col" (string) if parseable, else node_id (as fallback)
      label_to_node_id: "row,col" (string) -> node_id
    """
    node_id_to_label: Dict[str, str] = {}
    label_to_node_id: Dict[str, str] = {}

    for n in root.findall(".//g:graph/g:node", NS):
        nid = n.attrib.get("id")
        if not nid:
            continue

        label_txt: Optional[str] = None
        for d in n.findall("./g:data", NS):
            lab = d.find("./y:ShapeNode/y:NodeLabel", NS)
            if lab is None:
                lab = d.find("./y:GenericNode/y:NodeLabel", NS)
            if lab is not None and lab.text:
                label_txt = lab.text.strip()
                break

        # If label matches "row,col", store exactly that string; else fallback to id.
        if label_txt and LABEL_RE.match(label_txt):
            node_id_to_label[nid] = label_txt
            # If multiple nodes somehow share the same label, keep the first mapping.
            label_to_node_id.setdefault(label_txt, nid)
        else:
            node_id_to_label[nid] = nid  # fallback label

    return node_id_to_label, label_to_node_id

def _parse_edges(root: ET.Element,
                 weight_key_names: Tuple[str, ...] = ("w_total", "weight"),
                 weight_key_ids: Tuple[str, ...] = ("d_wtot", "d_weight"),
                 directed: bool = True) -> List[Tuple[str, str, float]]:
    """
    Extract edges as (source_id, target_id, weight). If weight is missing or not numeric, skip the edge.
    If directed=False, we still read what's in the GraphML (directed), but the caller can mirror later if needed.
    """
    name_to_id, _ = _build_edge_keymaps(root)
    edges: List[Tuple[str, str, float]] = []

    for e in root.findall(".//g:graph/g:edge", NS):
        s = e.attrib.get("source")
        t = e.attrib.get("target")
        if not s or not t:
            continue

        # Prefer named keys, then fallback key ids
        w_txt = _edge_data_lookup(
            e, name_to_id=name_to_id,
            candidates_by_name=weight_key_names,
            candidates_by_id=weight_key_ids
        )
        w = _to_float(w_txt)
        if w is None:
            # Edge has no usable weight; skip it
            continue

        edges.append((s, t, w))

    return edges

def _build_adjacency(nodes: Dict[str, str],
                     edges: List[Tuple[str, str, float]],
                     undirected: bool) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build adjacency from (source_id, target_id, weight) list.
    If undirected=True, add reverse neighbor for each edge.
    """
    adj: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in nodes.keys()}
    for s, t, w in edges:
        if s in adj:
            adj[s].append((t, w))
        else:
            adj[s] = [(t, w)]
        if undirected:
            if t in adj:
                adj[t].append((s, w))
            else:
                adj[t] = [(s, w)]
    return adj

# ---------- Dijkstra ----------

def dijkstra_shortest_path_graphml(
    graphml_path: Union[str, Path],
    src_label: str,
    dst_label: str,
    *,
    weight_key_names: Tuple[str, ...] = ("w_total", "weight"),
    weight_key_ids: Tuple[str, ...] = ("d_wtot", "d_weight"),
    directed: bool = True,     # treat the GraphML edges as directed by default
) -> Tuple[List[str], float]:
    """
    Compute the shortest path from src_label to dst_label using Dijkstra.
    Labels are the node labels "row,col" (strings) in the yEd file.

    Returns:
      (path_as_list_of_labels, total_cost)

    Raises:
      KeyError if source/target label not found.
      ValueError if no path exists.
    """
    # Parse GraphML
    tree = ET.parse(str(graphml_path))
    root = tree.getroot()

    node_id_to_label, label_to_node_id = _parse_nodes(root)

    if src_label not in label_to_node_id:
        raise KeyError(f"Source label not found: {src_label}")
    if dst_label not in label_to_node_id:
        raise KeyError(f"Target label not found: {dst_label}")

    src_id = label_to_node_id[src_label]
    dst_id = label_to_node_id[dst_label]

    edges = _parse_edges(root,
                         weight_key_names=weight_key_names,
                         weight_key_ids=weight_key_ids,
                         directed=directed)

    # Build adjacency
    adj = _build_adjacency(node_id_to_label, edges, undirected=not directed)

    # Dijkstra
    dist: Dict[str, float] = {nid: float("inf") for nid in node_id_to_label.keys()}
    prev: Dict[str, Optional[str]] = {nid: None for nid in node_id_to_label.keys()}
    dist[src_id] = 0.0

    heap: List[Tuple[float, str]] = [(0.0, src_id)]
    visited: Set[str] = set()

    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        if u == dst_id:
            break  # found best dist to target

        for v, w in adj.get(u, []):
            if v in visited:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    if dist[dst_id] == float("inf"):
        raise ValueError(f"No path from {src_label} to {dst_label} using current weights.")

    # Reconstruct path (node ids -> labels)
    path_ids: List[str] = []
    cur = dst_id
    while cur is not None:
        path_ids.append(cur)
        cur = prev[cur]
    path_ids.reverse()

    path_labels = [node_id_to_label[nid] for nid in path_ids]
    total_cost = dist[dst_id]
    return path_labels, total_cost