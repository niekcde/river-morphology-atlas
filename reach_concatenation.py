import geopandas as gpd
import networkx as nx
import shapely
import duckdb
import pandas as pd
import numpy as np
import os
import time
from collections import defaultdict

from shapely import wkt
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union
from tqdm import tqdm


def _set_endpoint(ls: LineString, which: str, p: Point) -> LineString:
    coords = list(ls.coords)
    if which == "start":
        coords[0] = (p.x, p.y)
    else:
        coords[-1] = (p.x, p.y)
    return LineString(coords)


def _line_parts(g):
    if isinstance(g, LineString):
        return [g]
    if isinstance(g, MultiLineString):
        return list(g.geoms)
    if hasattr(g, "geoms"):
        return [part for part in g.geoms if isinstance(part, LineString)]
    return []


def _endpoint_record(part_id, line: LineString, which: str,
                     endpoint_tol: float, adaptive: bool = False,
                     margin: float = 0.10, min_tol: float = 10.0,
                     max_tol=None):
    coords = list(line.coords)
    if len(coords) < 2:
        return None

    endpoint = Point(coords[0] if which == "start" else coords[-1])
    next_point = Point(coords[1] if which == "start" else coords[-2])
    step = endpoint.distance(next_point)

    local_tol = float(endpoint_tol)
    if adaptive:
        upper = float(endpoint_tol) if max_tol is None else float(max_tol)
        local_tol = step * (1.0 + margin)
        local_tol = max(float(min_tol), min(upper, local_tol))

    return {
        "part_id": part_id,
        "which": which,
        "endpoint": endpoint,
        "next_point": next_point,
        "step": step,
        "endpoint_tol": local_tol,
    }


def _endpoint_records_from_parts(parts, endpoint_tol: float,
                                 adaptive: bool = False,
                                 margin: float = 0.10,
                                 min_tol: float = 10.0,
                                 max_tol=None):
    records = []
    for part_id, line in enumerate(parts):
        if not isinstance(line, LineString) or len(line.coords) < 2:
            continue
        for which in ("start", "end"):
            record = _endpoint_record(
                part_id,
                line,
                which,
                endpoint_tol=endpoint_tol,
                adaptive=adaptive,
                margin=margin,
                min_tol=min_tol,
                max_tol=max_tol,
            )
            if record is not None:
                records.append(record)
    return records


def _insert_or_snap_point_on_line(line: LineString, point: Point, tol: float = 1e-6):
    coords = list(line.coords)
    if len(coords) < 2:
        return line, point, "unchanged"

    closest_coord = min(coords, key=lambda coord: point.distance(Point(coord)))
    closest_point = Point(closest_coord)
    if point.distance(closest_point) <= tol:
        return line, closest_point, "existing_vertex"

    measure = line.project(point)
    if measure <= tol:
        return line, Point(coords[0]), "line_start"
    if measure >= line.length - tol:
        return line, Point(coords[-1]), "line_end"

    projected = line.interpolate(measure)
    new_coords = [coords[0]]
    distance_so_far = 0.0
    inserted = False
    snap_point = projected
    action = "unchanged"

    for start, end in zip(coords[:-1], coords[1:]):
        segment = LineString([start, end])
        segment_length = segment.length
        segment_end_distance = distance_so_far + segment_length

        if not inserted and measure <= segment_end_distance + tol:
            if measure <= distance_so_far + tol:
                snap_point = Point(start)
                action = "existing_vertex"
            elif measure >= segment_end_distance - tol:
                snap_point = Point(end)
                action = "existing_vertex"
            else:
                snap_coord = (projected.x, projected.y)
                new_coords.append(snap_coord)
                snap_point = Point(snap_coord)
                action = "inserted_vertex"
            inserted = True

        new_coords.append(end)
        distance_so_far = segment_end_distance

    if not inserted:
        return line, Point(coords[-1]), "line_end"

    if action == "inserted_vertex":
        return LineString(new_coords), snap_point, action
    return line, snap_point, action


def _endpoint_diagnostic_record(record, status=None, reasons=None, extra=None):
    out = {
        "part_id": record["part_id"],
        "which": record["which"],
        "x": float(record["endpoint"].x),
        "y": float(record["endpoint"].y),
        "next_x": float(record["next_point"].x),
        "next_y": float(record["next_point"].y),
        "step": float(record["step"]),
    }
    if status is not None:
        out["status"] = status
    if reasons is not None:
        out["reasons"] = sorted(reasons)
    if extra:
        out.update(extra)
    return out


def _nearest_distance_to_other_parts(record, parts):
    distances = []
    for part_id, line in enumerate(parts):
        if part_id == record["part_id"]:
            continue
        if isinstance(line, LineString) and not line.is_empty:
            distances.append(record["endpoint"].distance(line))
    return min(distances) if distances else float("inf")


def _choose_outer_endpoint(candidates, parts):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(
        candidates,
        key=lambda record: _nearest_distance_to_other_parts(record, parts),
    )


def _select_global_terminal_endpoints(parts, loose_records):
    """
    Use MultiLineString part order to identify the two expected outer terminals.

    This assumes the parts produced by line_merge are ordered along the main path.
    The selected endpoints are removed from the large-gap search because they are
    valid loose ends of the full path, not internal gaps.
    """
    if len(parts) < 2:
        return []

    first_part_id = 0
    last_part_id = len(parts) - 1
    first_candidates = [
        record for record in loose_records
        if record["part_id"] == first_part_id
    ]
    last_candidates = [
        record for record in loose_records
        if record["part_id"] == last_part_id
    ]

    terminals = []
    first = _choose_outer_endpoint(first_candidates, parts)
    last = _choose_outer_endpoint(last_candidates, parts)

    if first is not None:
        terminals.append(first)
    if last is not None and (last["part_id"], last["which"]) != (first["part_id"], first["which"]):
        terminals.append(last)

    return terminals


def snap_close_endpoint_gaps(
    g,
    gap_tol: float = 5.0,
    stump_match_tol: float = 1e-6,
    connected_tol: float = 1e-6,
    exclude_global_terminals: bool = True,
    snap_endpoint_to_line: bool = True,
    include_endpoint_diagnostics: bool = False,
    return_diagnostics: bool = False,
):
    """
    Snap tiny endpoint gaps between different line parts before other repairs.

    Pairs that look like removable stumps are skipped; the graph extraction
    stage discards dangling stumps after deliberate noding.
    """
    diagnostics = {
        "attempted": False,
        "input_geom_type": None if g is None else g.geom_type,
        "output_geom_type": None if g is None else g.geom_type,
        "gap_tol": gap_tol,
        "connected_tol": connected_tol,
        "protected_endpoint_count": 0,
        "protected_endpoint_reasons": {},
        "loose_endpoint_count": 0,
        "global_terminal_endpoint_count": 0,
        "global_terminal_method": "part_order",
        "unresolved_endpoint_count": 0,
        "candidate_pair_count": 0,
        "endpoint_line_candidate_count": 0,
        "skipped_stump_like_count": 0,
        "snap_cluster_count": 0,
        "endpoint_line_snap_count": 0,
        "snapped_endpoint_count": 0,
        "nearest_endpoint_pair_distance": np.nan,
        "nearest_endpoint_line_distance": np.nan,
        "nearest_unresolved_endpoint_pair_distance": np.nan,
        "nearest_unresolved_endpoint_line_distance": np.nan,
        "min_gap": np.nan,
        "max_gap": np.nan,
        "pairs": [],
        "endpoint_line_pairs": [],
        "all_endpoints": [],
        "protected_endpoints": [],
        "loose_endpoints": [],
        "global_terminal_endpoints": [],
        "unresolved_endpoints": [],
    }

    if g is None:
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    if isinstance(g, LineString):
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    if not isinstance(g, MultiLineString):
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    parts = list(g.geoms)
    if len(parts) <= 1:
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    records = _endpoint_records_from_parts(parts, endpoint_tol=gap_tol)
    if len(records) < 2:
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    diagnostics["attempted"] = True
    protected = defaultdict(set)

    def endpoint_key(record):
        return (record["part_id"], record["which"])

    def protect(record, reason):
        protected[endpoint_key(record)].add(reason)

    # Stage 1: classify endpoints that already participate in local geometry.
    # These are protected from the larger gap search.
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if left["part_id"] == right["part_id"]:
                continue

            distance = left["endpoint"].distance(right["endpoint"])
            if (
                np.isnan(diagnostics["nearest_endpoint_pair_distance"]) or
                distance < diagnostics["nearest_endpoint_pair_distance"]
            ):
                diagnostics["nearest_endpoint_pair_distance"] = float(distance)

            if distance <= connected_tol:
                protect(left, "connected_endpoint")
                protect(right, "connected_endpoint")

            left_looks_like_stump = (
                left["next_point"].distance(right["endpoint"]) <= stump_match_tol
            )
            right_looks_like_stump = (
                right["next_point"].distance(left["endpoint"]) <= stump_match_tol
            )
            if left_looks_like_stump:
                protect(left, "stump_candidate")
                protect(right, "stump_target")
                diagnostics["skipped_stump_like_count"] += 1
            if right_looks_like_stump:
                protect(right, "stump_candidate")
                protect(left, "stump_target")
                diagnostics["skipped_stump_like_count"] += 1

    for record in records:
        for target_part_id, target_line in enumerate(parts):
            if target_part_id == record["part_id"]:
                continue
            if not isinstance(target_line, LineString) or target_line.length == 0:
                continue

            distance = record["endpoint"].distance(target_line)
            if (
                np.isnan(diagnostics["nearest_endpoint_line_distance"]) or
                distance < diagnostics["nearest_endpoint_line_distance"]
            ):
                diagnostics["nearest_endpoint_line_distance"] = float(distance)

            if distance <= connected_tol:
                protect(record, "connected_to_line")

            if record["next_point"].distance(target_line) <= stump_match_tol:
                protect(record, "stump_candidate_line")

    reason_counts = defaultdict(int)
    for reasons in protected.values():
        for reason in reasons:
            reason_counts[reason] += 1
    diagnostics["protected_endpoint_count"] = len(protected)
    diagnostics["protected_endpoint_reasons"] = dict(reason_counts)

    loose_records = [
        record for record in records
        if endpoint_key(record) not in protected
    ]
    diagnostics["loose_endpoint_count"] = len(loose_records)

    global_terminal_records = []
    if exclude_global_terminals:
        global_terminal_records = _select_global_terminal_endpoints(
            parts,
            loose_records,
        )
    global_terminal_keys = {
        endpoint_key(record)
        for record in global_terminal_records
    }
    diagnostics["global_terminal_endpoint_count"] = len(global_terminal_records)

    unresolved_records = [
        record for record in loose_records
        if endpoint_key(record) not in global_terminal_keys
    ]
    diagnostics["unresolved_endpoint_count"] = len(unresolved_records)

    if include_endpoint_diagnostics:
        diagnostics["all_endpoints"] = [
            _endpoint_diagnostic_record(
                record,
                status=(
                    "protected"
                    if endpoint_key(record) in protected
                    else "global_terminal"
                    if endpoint_key(record) in global_terminal_keys
                    else "unresolved"
                ),
                reasons=protected.get(endpoint_key(record), []),
            )
            for record in records
        ]
        diagnostics["protected_endpoints"] = [
            _endpoint_diagnostic_record(
                record,
                status="protected",
                reasons=protected[endpoint_key(record)],
            )
            for record in records
            if endpoint_key(record) in protected
        ]
        diagnostics["loose_endpoints"] = [
            _endpoint_diagnostic_record(
                record,
                status=(
                    "global_terminal"
                    if endpoint_key(record) in global_terminal_keys
                    else "unresolved"
                ),
            )
            for record in loose_records
        ]
        diagnostics["global_terminal_endpoints"] = [
            _endpoint_diagnostic_record(record, status="global_terminal")
            for record in global_terminal_records
        ]
        diagnostics["unresolved_endpoints"] = [
            _endpoint_diagnostic_record(record, status="unresolved")
            for record in unresolved_records
        ]
    nearest_unresolved_line_distance = {}
    for i, record in enumerate(unresolved_records):
        best_distance = float("inf")
        for target_part_id, target_line in enumerate(parts):
            if target_part_id == record["part_id"]:
                continue
            if not isinstance(target_line, LineString) or target_line.length == 0:
                continue

            measure = target_line.project(record["endpoint"])
            if measure <= connected_tol or measure >= target_line.length - connected_tol:
                continue

            distance = record["endpoint"].distance(target_line)
            if distance < best_distance:
                best_distance = distance

        if best_distance < float("inf"):
            nearest_unresolved_line_distance[i] = best_distance
            if (
                np.isnan(diagnostics["nearest_unresolved_endpoint_line_distance"]) or
                best_distance < diagnostics["nearest_unresolved_endpoint_line_distance"]
            ):
                diagnostics["nearest_unresolved_endpoint_line_distance"] = float(best_distance)

    # Stage 2a: snap remaining unresolved endpoint-to-endpoint gaps.
    accepted_distances = []
    nearest_endpoint = {}
    endpoint_pair_candidates = []
    for i, left in enumerate(unresolved_records):
        best_j = None
        best_distance = float("inf")
        for j, right in enumerate(unresolved_records):
            if i == j:
                continue
            if left["part_id"] == right["part_id"]:
                continue

            distance = left["endpoint"].distance(right["endpoint"])
            if distance < best_distance:
                best_j = j
                best_distance = distance

        if best_j is not None:
            nearest_endpoint[i] = (best_j, best_distance)
            if (
                np.isnan(diagnostics["nearest_unresolved_endpoint_pair_distance"]) or
                best_distance < diagnostics["nearest_unresolved_endpoint_pair_distance"]
            ):
                diagnostics["nearest_unresolved_endpoint_pair_distance"] = float(best_distance)

    for i, left in enumerate(unresolved_records):
        for j in range(i + 1, len(unresolved_records)):
            right = unresolved_records[j]
            if left["part_id"] == right["part_id"]:
                continue

            distance = left["endpoint"].distance(right["endpoint"])
            if distance > gap_tol:
                continue
            if nearest_endpoint.get(i, (None, None))[0] != j:
                continue
            if nearest_endpoint.get(j, (None, None))[0] != i:
                continue
            if nearest_unresolved_line_distance.get(i, float("inf")) < distance:
                continue
            if nearest_unresolved_line_distance.get(j, float("inf")) < distance:
                continue

            left_looks_like_stump = (
                left["next_point"].distance(right["endpoint"]) <= stump_match_tol
            )
            right_looks_like_stump = (
                right["next_point"].distance(left["endpoint"]) <= stump_match_tol
            )
            if left_looks_like_stump or right_looks_like_stump:
                diagnostics["skipped_stump_like_count"] += 1
                continue

            endpoint_pair_candidates.append((distance, i, j, left, right))

    snapped_parts = list(parts)
    snapped_endpoint_keys = set()
    used_endpoint_indices = set()
    for distance, i, j, left, right in sorted(endpoint_pair_candidates, key=lambda item: item[0]):
        if i in used_endpoint_indices or j in used_endpoint_indices:
            continue

        x = float(np.mean([left["endpoint"].x, right["endpoint"].x]))
        y = float(np.mean([left["endpoint"].y, right["endpoint"].y]))
        snap_point = Point(x, y)

        used_endpoint_indices.update([i, j])
        accepted_distances.append(distance)
        diagnostics["candidate_pair_count"] += 1
        diagnostics["snap_cluster_count"] += 1
        diagnostics["snapped_endpoint_count"] += 2
        diagnostics["pairs"].append({
            "part_id_a": left["part_id"],
            "which_a": left["which"],
            "part_id_b": right["part_id"],
            "which_b": right["which"],
            "distance": float(distance),
        })

        for member in (left, right):
            part_id = member["part_id"]
            snapped_endpoint_keys.add((part_id, member["which"]))
            snapped_parts[part_id] = _set_endpoint(
                snapped_parts[part_id],
                member["which"],
                snap_point,
            )

    endpoint_line_distances = []
    if snap_endpoint_to_line:
        # Stage 2b: for still-unresolved endpoints, consider endpoint-to-line gaps.
        records_after_endpoint_snaps = _endpoint_records_from_parts(
            snapped_parts,
            endpoint_tol=gap_tol,
        )

        endpoint_line_candidates = []
        for record in records_after_endpoint_snaps:
            ep_key = (record["part_id"], record["which"])
            if ep_key in protected or ep_key in snapped_endpoint_keys:
                continue

            best = None
            best_distance = float("inf")
            best_projection = None

            for target_part_id, target_line in enumerate(snapped_parts):
                if target_part_id == record["part_id"]:
                    continue
                if not isinstance(target_line, LineString) or target_line.length == 0:
                    continue

                distance = record["endpoint"].distance(target_line)
                if (
                    np.isnan(diagnostics["nearest_unresolved_endpoint_line_distance"]) or
                    distance < diagnostics["nearest_unresolved_endpoint_line_distance"]
                ):
                    diagnostics["nearest_unresolved_endpoint_line_distance"] = float(distance)
                if distance <= connected_tol:
                    continue
                if distance > gap_tol or distance >= best_distance:
                    continue

                measure = target_line.project(record["endpoint"])
                if measure <= stump_match_tol or measure >= target_line.length - stump_match_tol:
                    continue

                projection = target_line.interpolate(measure)
                if record["next_point"].distance(target_line) <= stump_match_tol:
                    diagnostics["skipped_stump_like_count"] += 1
                    continue

                best = {
                    "part_id": record["part_id"],
                    "which": record["which"],
                    "target_part_id": target_part_id,
                    "distance": float(distance),
                    "projected_measure": float(measure),
                }
                best_distance = distance
                best_projection = projection

            if best is None:
                continue

            endpoint_line_candidates.append({
                "record": record,
                "best": best,
                "projection": best_projection,
                "distance": best_distance,
            })

        used_source_target_pairs = set()
        for candidate in sorted(endpoint_line_candidates, key=lambda item: item["distance"]):
            record = candidate["record"]
            best = candidate["best"]
            best_projection = candidate["projection"]
            source_part_id = record["part_id"]
            target_part_id = best["target_part_id"]
            source_target_key = (source_part_id, target_part_id)
            if source_target_key in used_source_target_pairs:
                continue

            used_source_target_pairs.add(source_target_key)
            diagnostics["endpoint_line_candidate_count"] += 1
            diagnostics["endpoint_line_snap_count"] += 1
            diagnostics["snapped_endpoint_count"] += 1
            endpoint_line_distances.append(candidate["distance"])
            best["projected_x"] = float(best_projection.x)
            best["projected_y"] = float(best_projection.y)
            target_line, snap_point, snap_action = _insert_or_snap_point_on_line(
                snapped_parts[target_part_id],
                best_projection,
                tol=stump_match_tol,
            )
            snapped_parts[target_part_id] = target_line
            best["snap_x"] = float(snap_point.x)
            best["snap_y"] = float(snap_point.y)
            best["target_snap_action"] = snap_action
            diagnostics["endpoint_line_pairs"].append(best)

            snapped_parts[source_part_id] = _set_endpoint(
                snapped_parts[source_part_id],
                record["which"],
                snap_point,
            )

    all_distances = accepted_distances + endpoint_line_distances
    if all_distances:
        diagnostics["min_gap"] = float(min(all_distances))
        diagnostics["max_gap"] = float(max(all_distances))

    if diagnostics["snap_cluster_count"] == 0 and diagnostics["endpoint_line_snap_count"] == 0:
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    snapped = MultiLineString(snapped_parts)
    diagnostics["output_geom_type"] = snapped.geom_type
    return _return_with_optional_diagnostics(snapped, diagnostics, return_diagnostics)


def _return_with_optional_diagnostics(geom, diagnostics, return_diagnostics: bool):
    if return_diagnostics:
        return geom, diagnostics
    return geom


def _node_key(coord, node_tol: float = 1e-6):
    x, y = coord[0], coord[1]
    if node_tol and node_tol > 0:
        return (round(x / node_tol) * node_tol, round(y / node_tol) * node_tol)
    return (float(x), float(y))


def _lines_from_geometry(g):
    if isinstance(g, LineString):
        return [g]
    if isinstance(g, MultiLineString):
        return list(g.geoms)
    if hasattr(g, "geoms"):
        lines = []
        for part in g.geoms:
            lines.extend(_lines_from_geometry(part))
        return lines
    return []


def _line_network_graph_from_lines(lines, node_tol: float = 1e-6):
    graph = nx.Graph()
    for line in lines:
        coords = list(line.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            a_key = _node_key(a, node_tol=node_tol)
            b_key = _node_key(b, node_tol=node_tol)
            if a_key == b_key:
                continue

            segment = LineString([a_key, b_key])
            graph.add_edge(a_key, b_key, length=segment.length)

    return graph


def _line_piece_graph_from_lines(lines, node_tol: float = 1e-6):
    graph = nx.MultiGraph()
    for edge_id, line in enumerate(lines):
        coords = list(line.coords)
        if len(coords) < 2:
            continue

        source = _node_key(coords[0], node_tol=node_tol)
        target = _node_key(coords[-1], node_tol=node_tol)
        if source == target:
            continue

        graph.add_edge(
            source,
            target,
            key=edge_id,
            length=float(line.length),
            geometry=line,
            source=source,
            target=target,
        )

    return graph


def _shortest_edge_data(graph, source, target):
    edge_options = graph.get_edge_data(source, target)
    if edge_options is None:
        return None
    return min(
        edge_options.values(),
        key=lambda data: data.get("length", float("inf")),
    )


def _linestring_from_graph_path(graph, path_nodes):
    coords_out = []
    for source, target in zip(path_nodes[:-1], path_nodes[1:]):
        edge_data = _shortest_edge_data(graph, source, target)
        if edge_data is None:
            return None

        coords = list(edge_data["geometry"].coords)
        if source == edge_data["source"] and target == edge_data["target"]:
            oriented = coords
        else:
            oriented = list(reversed(coords))

        oriented[0] = source
        oriented[-1] = target

        if coords_out:
            coords_out.extend(oriented[1:])
        else:
            coords_out.extend(oriented)

    if len(coords_out) < 2:
        return None

    return LineString(coords_out)


def _coord_count(lines):
    return sum(len(line.coords) for line in lines if isinstance(line, LineString))


def _nearest_graph_node(graph, point: Point):
    best_node = None
    best_dist = float("inf")
    for node in graph.nodes:
        d = point.distance(Point(node))
        if d < best_dist:
            best_node = node
            best_dist = d
    return best_node, best_dist


def clean_self_intersecting_line(
    line: LineString,
    node_tol: float = 1e-6,
    max_cycle_rank: int = 5,
    max_vertices=250000,
    return_diagnostics: bool = False,
):
    """
    Repair one self-intersecting LineString by noding only that line and keeping
    the shortest path between its original endpoints.
    """
    diagnostics = {
        "attempted": False,
        "resolved": False,
        "reason": None,
        "input_geom_type": None if line is None else line.geom_type,
        "output_geom_type": None if line is None else line.geom_type,
        "input_vertices": 0,
        "output_vertices": None,
        "n_nodes": 0,
        "n_edges": 0,
        "cycle_rank": None,
        "source_distance": 0.0,
        "target_distance": 0.0,
    }

    if line is None:
        diagnostics["reason"] = "empty geometry"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    if not isinstance(line, LineString):
        diagnostics["reason"] = "not a LineString"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    coords = list(line.coords)
    diagnostics["input_vertices"] = len(coords)
    if len(coords) < 2:
        diagnostics["reason"] = "not enough coordinates"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    if line.is_simple:
        diagnostics["reason"] = "already simple"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    if max_vertices is not None and len(coords) > max_vertices:
        diagnostics["reason"] = "input vertex count exceeds limit"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    diagnostics["attempted"] = True
    noded = unary_union([line])
    noded_lines = _lines_from_geometry(noded)
    graph = _line_network_graph_from_lines(noded_lines, node_tol=node_tol)
    diagnostics["n_nodes"] = graph.number_of_nodes()
    diagnostics["n_edges"] = graph.number_of_edges()

    if graph.number_of_nodes() < 2:
        diagnostics["reason"] = "not enough graph nodes"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    source = _node_key(coords[0], node_tol=node_tol)
    target = _node_key(coords[-1], node_tol=node_tol)

    if source not in graph:
        source, diagnostics["source_distance"] = _nearest_graph_node(graph, Point(coords[0]))
    if target not in graph:
        target, diagnostics["target_distance"] = _nearest_graph_node(graph, Point(coords[-1]))

    if source is None or target is None or source == target:
        diagnostics["reason"] = "could not identify distinct endpoint nodes"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    if not nx.has_path(graph, source, target):
        diagnostics["reason"] = "endpoint nodes are disconnected"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    component = graph.subgraph(nx.node_connected_component(graph, source))
    cycle_rank = component.number_of_edges() - component.number_of_nodes() + 1
    diagnostics["cycle_rank"] = cycle_rank
    if max_cycle_rank is not None and cycle_rank > max_cycle_rank:
        diagnostics["reason"] = "too many cycles"
        return _return_with_optional_diagnostics(line, diagnostics, return_diagnostics)

    path_nodes = nx.shortest_path(graph, source=source, target=target, weight="length")
    cleaned = LineString(path_nodes)
    diagnostics["resolved"] = True
    diagnostics["reason"] = "resolved endpoint path"
    diagnostics["output_geom_type"] = cleaned.geom_type
    diagnostics["output_vertices"] = len(path_nodes)

    return _return_with_optional_diagnostics(cleaned, diagnostics, return_diagnostics)


def clean_self_intersections_in_geometry(
    g,
    node_tol: float = 1e-6,
    max_cycle_rank: int = 5,
    max_vertices=250000,
    return_diagnostics: bool = False,
):
    """
    Clean self-intersections part-by-part before endpoint stump cutting.
    """
    diagnostics = {
        "attempted": False,
        "parts_total": 0,
        "parts_not_simple": 0,
        "parts_resolved": 0,
        "part_diagnostics": [],
        "input_geom_type": None if g is None else g.geom_type,
        "output_geom_type": None if g is None else g.geom_type,
    }

    if g is None:
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    if isinstance(g, LineString):
        diagnostics["parts_total"] = 1
        if g.is_simple:
            return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

        cleaned, line_diag = clean_self_intersecting_line(
            g,
            node_tol=node_tol,
            max_cycle_rank=max_cycle_rank,
            max_vertices=max_vertices,
            return_diagnostics=True,
        )
        diagnostics["attempted"] = True
        diagnostics["parts_not_simple"] = 1
        diagnostics["parts_resolved"] = int(line_diag["resolved"])
        diagnostics["part_diagnostics"].append(line_diag)
        diagnostics["output_geom_type"] = cleaned.geom_type
        return _return_with_optional_diagnostics(cleaned, diagnostics, return_diagnostics)

    if not isinstance(g, MultiLineString):
        return _return_with_optional_diagnostics(g, diagnostics, return_diagnostics)

    cleaned_parts = []
    changed = False
    parts = list(g.geoms)
    diagnostics["parts_total"] = len(parts)

    for part_id, line in enumerate(parts):
        if not isinstance(line, LineString):
            cleaned_parts.append(line)
            continue
        if line.is_simple:
            cleaned_parts.append(line)
            continue

        cleaned, line_diag = clean_self_intersecting_line(
            line,
            node_tol=node_tol,
            max_cycle_rank=max_cycle_rank,
            max_vertices=max_vertices,
            return_diagnostics=True,
        )
        line_diag["part_id"] = part_id
        diagnostics["attempted"] = True
        diagnostics["parts_not_simple"] += 1
        diagnostics["parts_resolved"] += int(line_diag["resolved"])
        diagnostics["part_diagnostics"].append(line_diag)
        cleaned_parts.append(cleaned)
        changed = changed or line_diag["resolved"]

    if changed:
        cleaned_geom = MultiLineString(cleaned_parts)
    else:
        cleaned_geom = g

    diagnostics["output_geom_type"] = cleaned_geom.geom_type
    return _return_with_optional_diagnostics(
        cleaned_geom,
        diagnostics,
        return_diagnostics,
    )


def _endpoint_candidates_from_geometry(geom):
    candidates = []
    for geom_part_id, line in enumerate(_line_parts(geom)):
        coords = list(line.coords)
        if len(coords) < 2:
            continue
        for which, coord in (("start", coords[0]), ("end", coords[-1])):
            candidates.append({
                "geom_part_id": geom_part_id,
                "which": which,
                "point": Point(coord),
            })
    return candidates


def _outer_endpoint_from_row(row, other_geoms, geometry_col: str, reach_id_col: str):
    candidates = _endpoint_candidates_from_geometry(row[geometry_col])
    if not candidates:
        return None

    valid_other_geoms = [
        geom for geom in other_geoms
        if geom is not None and not geom.is_empty
    ]
    best = None
    best_distance = -1.0
    for candidate in candidates:
        point = candidate["point"]
        if valid_other_geoms:
            nearest_distance = min(
                point.distance(other_geom)
                for other_geom in valid_other_geoms
            )
        else:
            nearest_distance = float("inf")

        if nearest_distance > best_distance:
            best = dict(candidate)
            best_distance = nearest_distance

    if best is None:
        best = dict(candidates[0])
        best_distance = float("nan")

    best["outer_distance_to_other_geoms"] = best_distance
    best["reach_id"] = row[reach_id_col] if reach_id_col in row.index else None
    return best


def terminal_points_from_reach_order(
    rows,
    order_col: str = "dist_out",
    geometry_col: str = "geometry",
    reach_id_col: str = "reach_id",
    return_diagnostics: bool = False,
):
    """
    Pick the two expected main-path terminals from the ordered original reaches.

    The first and last rows in `order_col` define the terminal reaches. For each
    terminal reach, choose the endpoint farthest from the rest of the reach
    geometries; this avoids assuming start/end coordinate orientation.
    """
    diagnostics = {
        "attempted": False,
        "resolved": False,
        "reason": None,
        "terminal_count": 0,
        "order_col": order_col,
        "order_col_used": False,
        "ordered_row_count": 0,
        "source": None,
        "target": None,
    }

    if rows is None or rows.empty:
        diagnostics["reason"] = "empty rows"
        return _return_with_optional_diagnostics([], diagnostics, return_diagnostics)

    if geometry_col not in rows.columns:
        diagnostics["reason"] = f"missing geometry column: {geometry_col}"
        return _return_with_optional_diagnostics([], diagnostics, return_diagnostics)

    valid_rows = rows[rows[geometry_col].notna()].copy()
    valid_rows = valid_rows[
        valid_rows[geometry_col].apply(lambda geom: geom is not None and not geom.is_empty)
    ]
    diagnostics["ordered_row_count"] = len(valid_rows)
    if valid_rows.empty:
        diagnostics["reason"] = "no non-empty geometries"
        return _return_with_optional_diagnostics([], diagnostics, return_diagnostics)

    diagnostics["attempted"] = True
    if order_col and order_col in valid_rows.columns:
        ordered = valid_rows.sort_values(order_col, kind="mergesort")
        diagnostics["order_col_used"] = True
    else:
        ordered = valid_rows
        diagnostics["reason"] = "order column missing; used input row order"

    if len(ordered) == 1:
        candidates = _endpoint_candidates_from_geometry(ordered.iloc[0][geometry_col])
        if len(candidates) >= 2:
            terminal_points = [candidates[0]["point"], candidates[-1]["point"]]
            diagnostics["terminal_count"] = 2
            diagnostics["resolved"] = True
            diagnostics["reason"] = "single geometry endpoints"
            return _return_with_optional_diagnostics(
                terminal_points,
                diagnostics,
                return_diagnostics,
            )
        diagnostics["reason"] = "single geometry has fewer than two endpoints"
        return _return_with_optional_diagnostics([], diagnostics, return_diagnostics)

    source_row = ordered.iloc[0]
    target_row = ordered.iloc[-1]
    source_index = ordered.index[0]
    target_index = ordered.index[-1]

    source_other_geoms = ordered.loc[ordered.index != source_index, geometry_col].tolist()
    target_other_geoms = ordered.loc[ordered.index != target_index, geometry_col].tolist()

    source = _outer_endpoint_from_row(
        source_row,
        source_other_geoms,
        geometry_col=geometry_col,
        reach_id_col=reach_id_col,
    )
    target = _outer_endpoint_from_row(
        target_row,
        target_other_geoms,
        geometry_col=geometry_col,
        reach_id_col=reach_id_col,
    )

    if source is None or target is None:
        diagnostics["reason"] = "could not identify source and target endpoints"
        return _return_with_optional_diagnostics([], diagnostics, return_diagnostics)

    source_order_value = source_row[order_col] if order_col in ordered.columns else None
    target_order_value = target_row[order_col] if order_col in ordered.columns else None
    diagnostics["source"] = {
        "row_index": str(source_index),
        "reach_id": source.get("reach_id"),
        "order_value": source_order_value,
        "which": source["which"],
        "geom_part_id": source["geom_part_id"],
        "outer_distance_to_other_geoms": source["outer_distance_to_other_geoms"],
        "x": source["point"].x,
        "y": source["point"].y,
    }
    diagnostics["target"] = {
        "row_index": str(target_index),
        "reach_id": target.get("reach_id"),
        "order_value": target_order_value,
        "which": target["which"],
        "geom_part_id": target["geom_part_id"],
        "outer_distance_to_other_geoms": target["outer_distance_to_other_geoms"],
        "x": target["point"].x,
        "y": target["point"].y,
    }

    terminal_points = [source["point"], target["point"]]
    diagnostics["terminal_count"] = 2
    diagnostics["resolved"] = True
    if diagnostics["reason"] is None:
        diagnostics["reason"] = "ordered terminal endpoints"

    return _return_with_optional_diagnostics(
        terminal_points,
        diagnostics,
        return_diagnostics,
    )


def extract_main_path_from_graph(
    g,
    terminal_points=None,
    node_tol: float = 1e-6,
    union_grid_size: float = 1e-6,
    terminal_snap_tol=None,
    max_input_vertices=250000,
    return_diagnostics: bool = False,
        ):
    """
    Node a line network and keep the shortest weighted path between terminals.

    This intentionally uses unary_union as a graph-building step. Any dangling
    stumps or small cycles that are not on the selected terminal-to-terminal
    path are discarded.
    """
    diagnostics = {
        "attempted": False,
        "resolved": False,
        "reason": None,
        "input_geom_type": None if g is None else g.geom_type,
        "noded_geom_type": None,
        "output_geom_type": None if g is None else g.geom_type,
        "union_grid_size": union_grid_size,
        "input_vertices": 0,
        "noded_part_count": 0,
        "n_nodes": 0,
        "n_edges": 0,
        "n_components": 0,
        "n_terminals": 0,
        "terminal_point_count": 0,
        "terminal_distances": [],
        "selected_terminal_method": None,
        "path_length": None,
        "path_node_count": 0,
        "path_edge_count": 0,
        "path_coord_count": 0,
        "graph": None,
        "timings_sec": {},
    }
    timings = diagnostics["timings_sec"]
    t_start = time.perf_counter()
    t_last = t_start

    def mark_timing(name):
        nonlocal t_last
        now = time.perf_counter()
        timings[name] = now - t_last
        t_last = now

    def finish(geom):
        timings["total"] = time.perf_counter() - t_start
        return _return_with_optional_diagnostics(geom, diagnostics, return_diagnostics)

    if g is None:
        diagnostics["reason"] = "empty geometry"
        return finish(g)

    lines = _lines_from_geometry(g)
    mark_timing("extract_input_lines")
    if not lines:
        diagnostics["reason"] = "no LineString parts"
        return finish(g)

    diagnostics["input_vertices"] = _coord_count(lines)
    mark_timing("count_input_vertices")
    if max_input_vertices is not None and diagnostics["input_vertices"] > max_input_vertices:
        diagnostics["reason"] = "input vertex count exceeds limit"
        return finish(g)

    diagnostics["attempted"] = True
    noded = shapely.unary_union(lines, grid_size=union_grid_size)
    mark_timing("unary_union")
    # print(len(g.geoms))
    noded_lines = _lines_from_geometry(noded)
    diagnostics["noded_geom_type"] = noded.geom_type
    diagnostics["noded_part_count"] = len(noded_lines)
    mark_timing("extract_noded_lines")
    # print(len(noded_lines))
    graph = _line_piece_graph_from_lines(noded_lines, node_tol=node_tol)
    mark_timing("build_graph")
    if return_diagnostics:
        diagnostics["graph"] = graph
    diagnostics["n_nodes"] = graph.number_of_nodes()
    diagnostics["n_edges"] = graph.number_of_edges()
    diagnostics["n_components"] = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
    mark_timing("graph_stats")

    if graph.number_of_nodes() < 2:
        diagnostics["reason"] = "not enough graph nodes"
        return finish(g)

    terminals = [node for node, degree in graph.degree() if degree == 1]
    diagnostics["n_terminals"] = len(terminals)
    mark_timing("find_degree1_terminals")

    selected_nodes = []
    if terminal_points:
        diagnostics["terminal_point_count"] = len(terminal_points)
        for point in terminal_points:
            node, distance = _nearest_graph_node(graph, point)
            diagnostics["terminal_distances"].append(distance)
            if node is None:
                continue
            if terminal_snap_tol is not None and distance > terminal_snap_tol:
                continue
            if node not in selected_nodes:
                selected_nodes.append(node)
    mark_timing("match_terminal_points")

    if len(selected_nodes) >= 2:
        source, target = selected_nodes[0], selected_nodes[1]
        diagnostics["selected_terminal_method"] = "ordered terminal points"
    elif len(terminals) >= 2:
        best_pair = None
        best_length = -1.0
        for component_nodes in nx.connected_components(graph):
            component_terminals = [node for node in terminals if node in component_nodes]
            for i, source_candidate in enumerate(component_terminals):
                lengths = nx.single_source_dijkstra_path_length(
                    graph,
                    source_candidate,
                    weight="length",
                )
                for target_candidate in component_terminals[i + 1:]:
                    length = lengths.get(target_candidate)
                    if length is not None and length > best_length:
                        best_pair = (source_candidate, target_candidate)
                        best_length = length
        if best_pair is None:
            diagnostics["reason"] = "no connected terminal pair"
            mark_timing("select_terminals")
            return finish(g)
        source, target = best_pair
        diagnostics["selected_terminal_method"] = "longest graph terminal pair"
    else:
        diagnostics["reason"] = "not enough terminal nodes"
        mark_timing("select_terminals")
        return finish(g)
    mark_timing("select_terminals")

    diagnostics["selected_source"] = source
    diagnostics["selected_target"] = target

    if source == target:
        diagnostics["reason"] = "selected terminals are the same node"
        return finish(g)

    try:
        path_length, path_nodes = nx.single_source_dijkstra(
            graph,
            source=source,
            target=target,
            weight="length",
        )
    except nx.NetworkXNoPath:
        diagnostics["reason"] = "selected terminals are disconnected"
        mark_timing("shortest_path")
        return finish(g)
    mark_timing("shortest_path")
    resolved = _linestring_from_graph_path(graph, path_nodes)
    mark_timing("build_linestring")

    if resolved is None:
        diagnostics["reason"] = "could not reconstruct path geometry"
        return finish(g)

    diagnostics["resolved"] = True
    diagnostics["reason"] = "resolved shortest terminal path"
    diagnostics["path_length"] = path_length
    diagnostics["path_node_count"] = len(path_nodes)
    diagnostics["path_edge_count"] = max(0, len(path_nodes) - 1)
    diagnostics["path_coord_count"] = len(resolved.coords)
    diagnostics["output_geom_type"] = resolved.geom_type

    return finish(resolved)


def merge_lines(gdf):
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    # -----------------------------
    # 1. Keep ONLY mainstem edges
    # -----------------------------
    # Convert geometry → WKT
    df = gdf.copy(deep = True)
    df['geom_wkt'] = df["geometry"].apply(lambda g: g.wkt)

    # Only keep needed columns
    df = df[["geom_wkt"]]

    # Register in DuckDB
    con.register("segments", df)

    # -----------------------------
    # 2. Merge by main_path_id
    # -----------------------------
    merged = con.execute("""
        SELECT
            ST_AsText(
                ST_LineMerge(
                    ST_Collect(
                        LIST(ST_GeomFromText(geom_wkt))
                    )
                )
            ) AS merged_wkt
        FROM segments
    """).fetchdf()

    # Convert back to shapely
    merged["geometry"] = merged["merged_wkt"].apply(wkt.loads)
    return  merged["merged_wkt"].apply(wkt.loads).iloc[0]


def merge_mainpaths(
    df,
    dfG_or_mainstem_col=None,
    mainstem_col: str = "is_mainstem",
    main_path_col: str = "main_path_id",
    reach_id_col: str = "reach_id",
    terminal_order_col: str = "dist_out",
    snap_endpoint_gaps: bool = True,
    endpoint_gap_tol: float = 5.0,
    endpoint_gap_stump_match_tol: float = 1e-6,
    endpoint_gap_connected_tol: float = 1e-6,
    exclude_global_terminal_gaps: bool = True,
    snap_endpoint_to_line_gaps: bool = True,
    include_endpoint_gap_points: bool = False,
    clean_self_intersections: bool = True,
    self_intersection_node_tol: float = 1e-6,
    self_intersection_max_cycle_rank: int = 5,
    self_intersection_max_vertices=250000,
    graph_node_tol: float = 1e-6,
    graph_union_grid_size: float = 1e-6,
    graph_terminal_snap_tol=None,
    graph_max_input_vertices=250000,
    return_diagnostics: bool = False,
):
    if isinstance(dfG_or_mainstem_col, str):
        mainstem_col = dfG_or_mainstem_col
    elif dfG_or_mainstem_col is not None and not hasattr(dfG_or_mainstem_col, "columns"):
        raise TypeError(
            "The second positional argument must be a column name or a dataframe. "
            "Use merge_mainpaths(df) for the single-dataframe workflow."
        )

    required_cols = [mainstem_col, main_path_col, reach_id_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in df: {missing_cols}")

    mainstem_mask = df[mainstem_col].fillna(False).astype(bool)
    df_mainstem   = df.loc[mainstem_mask].copy()
    total         = df_mainstem[main_path_col].nunique()

    out = []

    for key, rows in tqdm(df_mainstem.groupby(main_path_col), total=total):
        reach_ids = rows[reach_id_col].unique()
        reach_ids_list = reach_ids.tolist()
        input_geom_length = float(rows.geometry.length.sum())

        l = merge_lines(rows)
        endpoint_gap_diagnostics = None
        if snap_endpoint_gaps:
            l, endpoint_gap_diagnostics = snap_close_endpoint_gaps(
                l,
                gap_tol=endpoint_gap_tol,
                stump_match_tol=endpoint_gap_stump_match_tol,
                connected_tol=endpoint_gap_connected_tol,
                exclude_global_terminals=exclude_global_terminal_gaps,
                snap_endpoint_to_line=snap_endpoint_to_line_gaps,
                include_endpoint_diagnostics=include_endpoint_gap_points,
                return_diagnostics=True,
            )

        self_intersection_diagnostics = None
        if clean_self_intersections:
            l, self_intersection_diagnostics = clean_self_intersections_in_geometry(
                l,
                node_tol=self_intersection_node_tol,
                max_cycle_rank=self_intersection_max_cycle_rank,
                max_vertices=self_intersection_max_vertices,
                return_diagnostics=True,
            )

        main_path_terminal_points, terminal_diagnostics = terminal_points_from_reach_order(
            rows,
            order_col=terminal_order_col,
            reach_id_col=reach_id_col,
            return_diagnostics=True,
        )
        l, graph_path_diagnostics = extract_main_path_from_graph(
            l,
            terminal_points=main_path_terminal_points,
            node_tol=graph_node_tol,
            union_grid_size=graph_union_grid_size,
            terminal_snap_tol=graph_terminal_snap_tol,
            max_input_vertices=graph_max_input_vertices,
            return_diagnostics=True,
        )

        is_multilinestring = l.geom_type == "MultiLineString"
        merged_geom_length = float(l.length)
        length_removed = input_geom_length - merged_geom_length
        length_loss_frac = (
            length_removed / input_geom_length
            if input_geom_length > 0
            else np.nan
        )
        record = {
            "main_path_id": key,
            "line": l,
            "is_multilinestring": is_multilinestring,
            "reach_ids": reach_ids_list,
            "input_geom_length": input_geom_length,
            "merged_geom_length": merged_geom_length,
            "length_removed": length_removed,
            "length_loss_frac": length_loss_frac,
        }
        if return_diagnostics:
            record["endpoint_gap_diagnostics"] = endpoint_gap_diagnostics
            record["self_intersection_diagnostics"] = self_intersection_diagnostics
            record["terminal_diagnostics"] = terminal_diagnostics
            record["graph_path_diagnostics"] = graph_path_diagnostics
        out.append(record)

    return pd.DataFrame(out)



if __name__ == "__main__":    

    directory = '/Volumes/PhD/SWORD/v17b/adjusted/'

    continent = 'eu'
    df  = gpd.read_file(directory + f'{continent}_sword_reaches_v17b.gpkg')
    
    df = df.to_crs('EPSG:3857')

    # df = manual_edits(df, continent = continent, remove = True)
    
    df_merged = merge_mainpaths(df)
    
    out_dir = '/Users/6256481/Desktop/PhD_icloud/morphology_atlas/merged_geometries/'
    out_parquet = os.path.join(out_dir, f"{continent}_merged_mainpaths.parquet")
    gdf_merged = gpd.GeoDataFrame(df_merged, geometry="line", crs=df.crs)
    gdf_merged.to_parquet(out_parquet)
