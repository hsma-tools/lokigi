from valhalla import Actor
import pandas as pd

from pathlib import Path
import subprocess

import osmium
import re
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def prepare_valhalla_network(
    osm_path,
    output_dir=None,
    output_name=None,
    force_rebuild=False,
):
    """
    Build Valhalla routing tiles from an .osm.pbf file.

    Parameters
    ----------
    osm_path : str or Path
        Path to .osm.pbf file.

    output_dir : str or Path, optional
        Directory to store Valhalla outputs.
        Defaults to a folder beside the OSM named after the file stem.

    output_name : str, optional
        Base name for generated outputs.
        Used for:
            - config file
            - tile directory
            - traffic extract

        Defaults to OSM filename stem.

    force_rebuild : bool
        If True, rebuild even if outputs already exist.

    Returns
    -------
    dict
        {
            "config_path": ...,
            "tile_dir": ...,
            "traffic_path": ...
        }
    """

    osm_path = Path(osm_path).resolve()

    if not osm_path.exists():
        raise FileNotFoundError(f"OSM file not found: {osm_path}")

    # ------------------------------------------------------------
    # Derive sensible defaults
    # ------------------------------------------------------------

    if output_name is None:
        output_name = osm_path.stem.replace("-latest", "")

    if output_dir is None:
        output_dir = osm_path.parent / f"{output_name}_valhalla"

    output_dir = Path(output_dir).resolve()

    # ------------------------------------------------------------
    # Create output paths
    # ------------------------------------------------------------

    config_path = output_dir / f"{output_name}.json"
    tile_dir = output_dir / f"{output_name}_tiles"
    traffic_path = output_dir / f"{output_name}_traffic.tar"

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Skip rebuild if tiles already exist
    # ------------------------------------------------------------

    if (
        tile_dir.exists()
        and any(tile_dir.iterdir())
        and config_path.exists()
        and not force_rebuild
    ):
        print("Using existing Valhalla build")

        return {
            "config_path": str(config_path),
            "tile_dir": str(tile_dir),
            "traffic_path": str(traffic_path),
        }

    # ------------------------------------------------------------
    # Generate config
    # ------------------------------------------------------------

    print("Generating Valhalla config...")

    with open(config_path, "w") as f:
        subprocess.run(
            [
                "valhalla_build_config",
                "--mjolnir-tile-dir",
                str(tile_dir),
                "--mjolnir-tile-extract",
                str(traffic_path),
            ],
            stdout=f,
            check=True,
        )

    # ------------------------------------------------------------
    # Build routing tiles
    # ------------------------------------------------------------

    print("Building Valhalla tiles (this may take a while)...")

    subprocess.run(
        [
            "valhalla_build_tiles",
            "-c",
            str(config_path),
            str(osm_path),
        ],
        check=True,
    )

    print("Valhalla build complete")

    return {
        "config_path": str(config_path),
        "tile_dir": str(tile_dir),
        "traffic_path": str(traffic_path),
    }


import os


def build_time_matrix_valhalla(
    origins_gdf,
    destinations_gdf,
    valhalla_config_path,
    output_csv_path,
    costing="auto",
    reshape_to_wide=False,
    metric_of_interest="travel_time_minutes",
):
    """
    Build a time/distance matrix between origins and destinations using Valhalla
    and stream results directly to a CSV to maintain a near-zero memory footprint.
    """
    actor = Actor(valhalla_config_path)

    # Fast extraction of coordinates
    sources = [
        {"lat": lat, "lon": lon}
        for lat, lon in zip(origins_gdf.geometry.y, origins_gdf.geometry.x)
    ]
    targets = [
        {"lat": lat, "lon": lon}
        for lat, lon in zip(destinations_gdf.geometry.y, destinations_gdf.geometry.x)
    ]

    origin_ids = origins_gdf["id"].values
    destination_ids = destinations_gdf["id"].values

    # Clean up any pre-existing file at the output path
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    _process_and_stream(
        sources,
        targets,
        origin_ids,
        destination_ids,
        actor,
        costing,
        reshape_to_wide,
        metric_of_interest,
        output_csv_path,
    )

    print(f"Matrix generation complete. Output saved to {output_csv_path}")


def _process_single_batch(
    sources, targets, origin_ids, destination_ids, actor, costing
):
    """Process a single batch without memory-heavy object arrays."""
    request = {
        "sources": sources,
        "targets": targets,
        "costing": costing,
        "verbose": False,
    }

    matrix = actor.matrix(request)

    # Convert None to np.nan inline to preserve ultra-fast native numerical types (float32)
    raw_durations = [
        [float("nan") if x is None else x for x in row]
        for row in matrix["sources_to_targets"]["durations"]
    ]
    raw_distances = [
        [float("nan") if x is None else x for x in row]
        for row in matrix["sources_to_targets"]["distances"]
    ]

    # Fast vectorized calculations using float32 to reduce memory footprint
    travel_times = np.array(raw_durations, dtype=np.float32) / 60.0
    distance_km = np.array(raw_distances, dtype=np.float32)

    # Meshgrid creation for mapping coordinates
    from_ids, to_ids = np.meshgrid(origin_ids, destination_ids, indexing="ij")

    return pd.DataFrame(
        {
            "from_id": from_ids.ravel(),
            "to_id": to_ids.ravel(),
            "travel_time_minutes": travel_times.ravel(),
            "distance_km": distance_km.ravel(),
        }
    )


def _process_and_stream(
    sources,
    targets,
    origin_ids,
    destination_ids,
    actor,
    costing,
    reshape_to_wide,
    metric_of_interest,
    output_csv_path,
):
    """
    Streams chunks directly to disk.
    Respects Valhalla's hard limit of (sources * targets <= 2500).
    """
    n_origins = len(sources)
    n_targets = len(targets)

    # Valhalla counts matrix limits as the product of sources and targets.
    # 50 * 50 = 2500 pairs per API request.
    origin_batch_size = 50
    target_batch_size = 50

    if reshape_to_wide:
        # Write Wide Header
        header_df = pd.DataFrame(columns=["from_id"] + list(destination_ids))
        header_df.to_csv(output_csv_path, index=False)

        for i in tqdm(
            range(0, n_origins, origin_batch_size), desc="Streaming Wide Matrix Rows"
        ):
            b_sources = sources[i : i + origin_batch_size]
            b_origin_ids = origin_ids[i : i + origin_batch_size]

            # Accumulate all destination slices for this specific batch of rows
            row_chunks = []
            for j in range(0, n_targets, target_batch_size):
                b_targets = targets[j : j + target_batch_size]
                b_target_ids = destination_ids[j : j + target_batch_size]

                chunk_df = _process_single_batch(
                    b_sources, b_targets, b_origin_ids, b_target_ids, actor, costing
                )
                row_chunks.append(chunk_df)

            # Combine the chunks for the current row group, pivot it, and append to file
            assembled_rows = pd.concat(row_chunks, ignore_index=True)
            pivoted_rows = (
                assembled_rows.pivot(
                    columns="to_id", index="from_id", values=metric_of_interest
                )
                .reset_index()
                .rename_axis(None, axis=1)
            )
            # Reindex to guarantee correct destination column ordering matches header
            pivoted_rows = pivoted_rows.reindex(
                columns=["from_id"] + list(destination_ids)
            )
            pivoted_rows.to_csv(output_csv_path, mode="a", header=False, index=False)

    else:
        # Long-form streaming
        is_first_chunk = True

        for i in tqdm(
            range(0, n_origins, origin_batch_size), desc="Streaming Long Matrix"
        ):
            b_sources = sources[i : i + origin_batch_size]
            b_origin_ids = origin_ids[i : i + origin_batch_size]

            for j in range(0, n_targets, target_batch_size):
                b_targets = targets[j : j + target_batch_size]
                b_target_ids = destination_ids[j : j + target_batch_size]

                chunk_df = _process_single_batch(
                    b_sources, b_targets, b_origin_ids, b_target_ids, actor, costing
                )

                # Append directly to CSV
                chunk_df.to_csv(
                    output_csv_path, mode="a", header=is_first_chunk, index=False
                )
                is_first_chunk = False


class AdvancedSpeedUpdater(osmium.SimpleHandler):
    def __init__(
        self,
        writer,
        minor_road_cap=None,
        traffic_multiplier=None,
        debug=False,
        debug_limit=50,
    ):
        super().__init__()
        self.writer = writer
        self.minor_road_cap = minor_road_cap
        self.traffic_multiplier = traffic_multiplier
        self.debug = debug
        self.debug_limit = debug_limit
        self.debug_count = 0
        self.changes_made = 0

        # OSM highway tags generally considered "minor"
        self.minor_road_types = {
            "tertiary",
            "unclassified",
            "residential",
            "living_street",
            "service",
            "track",
        }

    def _parse_speed(self, speed_str: str) -> float:
        """Extracts a numeric speed value from common UK OSM maxspeed formats."""
        if not speed_str:
            return None

        speed_str = speed_str.lower()
        # Handle the UK national speed limit for single carriageways
        if speed_str == "national":
            return 60.0

        # Extract the first block of digits found
        match = re.search(r"(\d+)", speed_str)
        if match:
            return float(match.group(1))

        return None

    def _debug_print(self, message):
        """Avoid flooding console with millions of lines."""
        if self.debug and self.debug_count < self.debug_limit:
            print(message)
            self.debug_count += 1

    def node(self, n):
        self.writer.add_node(n)

    def relation(self, r):
        self.writer.add_relation(r)

    def way(self, w):
        tags = dict(w.tags)

        # --- DEBUG: inspect all tags on this way ---
        self._debug_print(f"[WAY {w.id}] ALL TAGS:")
        for k, v in tags.items():
            self._debug_print(f"    {k}: {v}")

        # specifically highlight anything related to maxspeed
        maxspeed_related = {k: v for k, v in tags.items() if "maxspeed" in k.lower()}
        if maxspeed_related:
            self._debug_print(f"[WAY {w.id}] MAXSPEED-RELATED TAGS:")
            for k, v in maxspeed_related.items():
                self._debug_print(f"    {k}: {v}")

        highway = tags.get("highway")

        # We only care about drivable roads
        if not highway:
            self.writer.add_way(w)
            return

        current_speed_str = tags.get("maxspeed")
        speed_val = self._parse_speed(current_speed_str)

        inferred_missing_speed = False

        # If minor road has no speed, assume router default = 60 mph
        if speed_val is None and highway in self.minor_road_types:
            speed_val = 60.0
            inferred_missing_speed = True

        # If we still don't have a speed (e.g., untagged major road), pass it through
        if speed_val is None:
            self.writer.add_way(w)
            return

        original_speed_val = speed_val
        change_reasons = []

        # --- 1. Apply Minor Road Cap ---
        if self.minor_road_cap is not None and highway in self.minor_road_types:
            if speed_val > self.minor_road_cap:
                speed_val = float(self.minor_road_cap)
                change_reasons.append(f"minor road cap -> {self.minor_road_cap}")

        # --- 2. Apply Traffic Modifier ---
        if self.traffic_multiplier is not None:
            old_speed = speed_val
            speed_val = speed_val * self.traffic_multiplier

            if speed_val != old_speed:
                change_reasons.append(f"traffic multiplier x{self.traffic_multiplier}")

        # If the speed changed, update the tags and write the modified way
        if speed_val != original_speed_val:
            new_speed_str = f"{int(round(speed_val))} mph"
            # tags['maxspeed'] = str(int(round(speed_val)))
            tags["maxspeed"] = new_speed_str
            tags["maxspeed:motorcar"] = new_speed_str
            tags.pop("maxspeed:type", None)

            self._debug_print(
                f"[WAY {w.id}] "
                f"{highway} | "
                f"original='{current_speed_str}' "
                f"(parsed={original_speed_val}) -> "
                f"new='{speed_val} ({new_speed_str})' | "
                f"reasons={change_reasons} | "
                f"inferred_missing={inferred_missing_speed}"
            )

            self.changes_made += 1

            self.writer.add_way(w.replace(tags=tags))
        else:
            # Write unmodified
            self.writer.add_way(w)


def process_uk_network(
    input_file: str,
    output_file: str,
    minor_road_cap=30,
    traffic_multiplier=None,
    debug=False,
    debug_limit=50,
):

    abs_input = str(Path(input_file).resolve())
    print(f"Reading from {abs_input}")
    abs_output = str(Path(output_file).resolve())
    print(f"Will write to {abs_output}")

    output_path = Path(abs_output)

    if output_path.exists():
        output_path.unlink()

    writer = osmium.SimpleWriter(abs_output)

    try:
        handler = AdvancedSpeedUpdater(
            writer=writer,
            minor_road_cap=minor_road_cap,
            traffic_multiplier=traffic_multiplier,
            debug=debug,
            debug_limit=debug_limit,
        )

        print(f"Processing {abs_input}...")
        handler.apply_file(abs_input)

        if debug:
            print(
                f"Debug output truncated at {handler.debug_count}/{debug_limit} changes"
            )
        print(f"Done! Saved to {abs_output}")
        print(f"Total Changes made: {handler.changes_made}")

    finally:
        writer.close()


class WayInspector(osmium.SimpleHandler):
    def __init__(self, way_ids):
        super().__init__()
        self.way_ids = set(way_ids)

    def way(self, w):
        if w.id not in self.way_ids:
            return

        tags = dict(w.tags)

        highway = tags.get("highway")
        maxspeed = tags.get("maxspeed")

        print("\n" + "=" * 80)
        print(f"WAY {w.id}")
        print(f"highway: {highway}")
        print(f"maxspeed: {maxspeed}")

        # --- highlight maxspeed-related tags specifically ---
        maxspeed_tags = {k: v for k, v in tags.items() if "maxspeed" in k.lower()}
        if maxspeed_tags:
            print("\n-- maxspeed-related tags --")
            for k, v in maxspeed_tags.items():
                print(f"  {k}: {v}")

        # --- show source-related overrides (very common culprit) ---
        source_tags = {k: v for k, v in tags.items() if "source" in k.lower()}
        if source_tags:
            print("\n-- source-related tags --")
            for k, v in source_tags.items():
                print(f"  {k}: {v}")

        # --- optional: full dump (kept last so it doesn't obscure key info) ---
        print("\n-- all tags --")
        for k, v in sorted(tags.items()):
            print(f"  {k}: {v}")


def valhalla_detailed_route(
    engine,
    origin,
    destination,
    costing="auto",
    costing_options=None,
    units="miles",
    verbose=True,
):
    """
    Run a Valhalla route and compute leg-level speeds.
    """

    request = {
        "locations": [
            {"lat": origin[0], "lon": origin[1], "type": "break"},
            {"lat": destination[0], "lon": destination[1], "type": "break"},
        ],
        "costing": costing,
        "units": units,
        "directions_options": {
            "units": units,
            "narrative": True,
            "language": "en-US",
            "shape_format": "polyline6",
        },
    }

    if costing_options:
        request["costing_options"] = costing_options

    response = engine.route(request)

    trip = response["trip"]

    summary = {
        "total_time_s": trip["summary"]["time"],
        "total_dist": trip["summary"]["length"],
        "legs": [],
    }

    if verbose:
        print("\n=== OVERALL ===")
        print(f"Time: {trip['summary']['time'] / 60:.1f} min")
        print(f"Distance: {trip['summary']['length']:.2f} {units}")

    for i, leg in enumerate(trip["legs"]):
        leg_time_s = leg["summary"]["time"]
        leg_dist = leg["summary"]["length"]

        # compute speed
        if leg_time_s > 0:
            leg_speed = leg_dist / (leg_time_s / 3600)
        else:
            leg_speed = 0

        leg_data = {
            "leg_index": i,
            "time_s": leg_time_s,
            "distance": leg_dist,
            "speed": leg_speed,
        }

        summary["legs"].append(leg_data)

        if verbose:
            print(f"\n--- LEG {i} ---")
            print(f"Time: {leg_time_s / 60:.1f} min")
            print(f"Distance: {leg_dist:.2f} {units}")
            print(f"Speed: {leg_speed:.1f} {units}/h")

            print("\nManeuvers:")
            for m in leg["maneuvers"]:
                print(
                    f"{m['time']:>4}s | {m['length']:>6} | {m.get('instruction', '')}"
                )

    return response, summary


def valhalla_audit_route_speeds(response, speed_bands=None, unit="km"):
    """
    Analyse Valhalla route maneuvers by implied speed distribution.

    Parameters
    ----------
    response : dict
        Valhalla route response
    speed_bands : list of tuples
        [(label, max_speed_kmh), ...]
    unit : str
        "km" or "miles" (Valhalla typically returns km internally unless configured)

    Returns
    -------
    dict summary
    """

    if speed_bands is None:
        speed_bands = [
            ("< 20 km/h (urban/stop-start)", 20),
            ("20–40 km/h (dense urban)", 40),
            ("40–60 km/h (suburban)", 60),
            ("60–90 km/h (A-road)", 90),
            ("90+ km/h (motorway-like)", float("inf")),
        ]

    # accumulators
    band_stats = defaultdict(lambda: {"time": 0.0, "dist": 0.0})
    total_time = 0.0
    total_dist = 0.0

    trip = response["trip"]

    for leg in trip["legs"]:
        for m in leg["maneuvers"]:
            time_s = m["time"]
            dist = m["length"]

            # Valhalla typically returns km in "length"
            if unit == "miles":
                dist_km = dist * 1.60934
            else:
                dist_km = dist

            time_h = time_s / 3600

            total_time += time_s
            total_dist += dist_km

            speed = (dist_km / time_h) if time_h > 0 else 0

            # assign to band
            for label, max_speed in speed_bands:
                if speed <= max_speed:
                    band_stats[label]["time"] += time_s
                    band_stats[label]["dist"] += dist_km
                    break

    # report
    print("\n=== ROUTE SUMMARY ===")
    print(f"Total time: {total_time / 60:.1f} min")
    print(f"Total dist: {total_dist:.2f} km")

    print("\n=== SPEED DISTRIBUTION ===")

    for label, _ in speed_bands:
        t = band_stats[label]["time"]
        d = band_stats[label]["dist"]

        print(
            f"{label:35s} | "
            f"time: {t / 60:6.1f} min ({t / total_time * 100:5.1f}%) | "
            f"dist: {d:6.1f} km ({d / total_dist * 100:5.1f}%)"
        )

    return {
        "total_time_s": total_time,
        "total_dist_km": total_dist,
        "band_stats": dict(band_stats),
    }


def valhalla_maneuvers_to_gdf(response, crs="EPSG:4326", units="km"):
    import polyline
    import geopandas as gpd
    from shapely.geometry import LineString

    rows = []

    for leg_idx, leg in enumerate(response["trip"]["legs"]):
        coords = polyline.decode(leg["shape"], precision=6)

        for m_idx, m in enumerate(leg["maneuvers"]):
            start = m["begin_shape_index"]
            end = m["end_shape_index"]

            segment = coords[start : end + 1]

            if len(segment) < 2:
                continue

            geom = LineString([(lon, lat) for lat, lon in segment])

            time_s = m["time"]
            dist = m["length"]

            speed = dist / (time_s / 3600) if time_s > 0 else 0

            rows.append(
                {
                    "leg": leg_idx,
                    "maneuver": m_idx,
                    "instruction": m.get("instruction"),
                    "time_s": time_s,
                    "distance": dist,
                    "speed_kmh": speed,
                    "geometry": geom,
                }
            )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    return gdf
