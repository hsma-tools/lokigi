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


def build_time_matrix_valhalla(
    origins_gdf,
    destinations_gdf,
    valhalla_config_path,
    costing="auto",
    reshape_for_lokigi=True,
    metric_of_interest="travel_time_minutes",
):
    """
    Build a time/distance matrix between origins and destinations using Valhalla.

    Implements smart batching for datasets exceeding 2500 total locations to balance:
    - API request size limits (Valhalla recommends max ~2500 locations per request)
    - Memory efficiency (avoids loading entire matrix in memory)
    - Network efficiency (fewer, larger requests vs. many small ones)

    Optimizations:
    - Vectorized coordinate extraction using NumPy
    - Vectorized time/distance conversion
    - Smart batching for large datasets
    - Pre-allocate arrays for better memory efficiency
    - Use NumPy operations instead of list comprehension
    - Stream results to avoid loading full matrix in memory
    """
    actor = Actor(valhalla_config_path)

    # Vectorized extraction of coordinates
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

    # If under threshold, process in single batch
    # Using conservative 1000 threshold based on Valhalla's actual limits
    if len(sources) < 50 and len(targets) < 50:
        print("Processing in a single batch")
        if reshape_for_lokigi:
            return (
                _process_single_batch(
                    sources, targets, origin_ids, destination_ids, actor, costing
                )
                .pivot(columns="to_id", index="from_id", values=metric_of_interest)
                .reset_index()
                .rename_axis(None, axis=1)
            )

        else:
            return _process_single_batch(
                sources, targets, origin_ids, destination_ids, actor, costing
            )

    # Smart batching strategy for large datasets
    if reshape_for_lokigi:
        return (
            _process_batched(
                sources, targets, origin_ids, destination_ids, actor, costing
            )
            .pivot(columns="to_id", index="from_id", values=metric_of_interest)
            .reset_index()
            .rename_axis(None, axis=1)
        )
    else:
        return _process_batched(
            sources, targets, origin_ids, destination_ids, actor, costing
        )


def _process_single_batch(
    sources, targets, origin_ids, destination_ids, actor, costing
):
    """Process a single batch of sources and targets."""
    request = {
        "sources": sources,
        "targets": targets,
        "costing": costing,
        "matrix_locations": len(sources) + len(targets),
        "verbose": False,
    }

    matrix = actor.matrix(request)

    # Convert nested lists to NumPy arrays for vectorized operations
    durations = np.array(matrix["sources_to_targets"]["durations"], dtype=object)
    distances = np.array(matrix["sources_to_targets"]["distances"], dtype=object)

    # Vectorized unit conversion - handle None values
    travel_times = np.empty(durations.shape, dtype=object)
    travel_times[durations != None] = durations[durations != None] / 60
    travel_times[durations == None] = None

    distance_km = np.empty(distances.shape, dtype=object)
    distance_km[distances != None] = distances[distances != None]
    distance_km[distances == None] = None

    # Create meshgrid for efficient cartesian product
    from_ids, to_ids = np.meshgrid(origin_ids, destination_ids, indexing="ij")

    # Flatten all arrays for DataFrame construction
    return pd.DataFrame(
        {
            "from_id": from_ids.ravel(),
            "to_id": to_ids.ravel(),
            "travel_time_minutes": travel_times.ravel(),
            "distance_km": distance_km.ravel(),
        }
    )


def _process_batched(sources, targets, origin_ids, destination_ids, actor, costing):
    """
    Process large datasets using smart batching strategy.

    Strategy:
    1. If destinations are small, batch origins
    2. If origins are small, batch destinations
    3. If both are large, batch origins and iterate destinations

    This minimizes redundant API calls and memory usage while respecting
    Valhalla's 2500 location limit per request.
    """
    results = []

    # Determine batching strategy
    n_origins = len(sources)
    n_targets = len(targets)

    # Use conservative limits to ensure we never hit Valhalla's hard limit
    SAFE_LIMIT = 1200  # Conservative safety margin

    print(f"Dataset: {n_origins} origins × {n_targets} targets")

    # # Strategy 1: Batch origins (if destinations are manageable)
    # if n_targets <= 600 and n_origins > 600:
    #     print("Strategy 1: Batching origins")
    #     origin_batch_size = _calculate_batch_size(
    #         n_origins, n_targets, max_batch_size=SAFE_LIMIT
    #     )
    #     n_batches = (n_origins + origin_batch_size - 1) // origin_batch_size

    #     with tqdm(
    #         total=n_batches, desc="Processing origin batches", unit="batch"
    #     ) as pbar:
    #         for i in range(0, n_origins, origin_batch_size):
    #             batch_sources = sources[i : i + origin_batch_size]
    #             batch_origin_ids = origin_ids[i : i + origin_batch_size]

    #             # Verify batch size is safe
    #             batch_locations = len(batch_sources) + n_targets
    #             if batch_locations > SAFE_LIMIT:
    #                 raise ValueError(
    #                     f"Strategy 1 batch exceeds limit: {len(batch_sources)} origins + {n_targets} targets = {batch_locations} > {SAFE_LIMIT}"
    #                 )

    #             batch_df = _process_single_batch(
    #                 batch_sources,
    #                 targets,
    #                 batch_origin_ids,
    #                 destination_ids,
    #                 actor,
    #                 costing,
    #             )
    #             results.append(batch_df)
    #             pbar.update(1)

    # # Strategy 2: Batch destinations (if origins are manageable)
    # elif n_origins <= 600 and n_targets > 600:
    #     print("Strategy 2: Batching destinations")
    #     target_batch_size = _calculate_batch_size(
    #         n_targets, n_origins, max_batch_size=SAFE_LIMIT
    #     )
    #     n_batches = (n_targets + target_batch_size - 1) // target_batch_size

    #     with tqdm(
    #         total=n_batches, desc="Processing destination batches", unit="batch"
    #     ) as pbar:
    #         for j in range(0, n_targets, target_batch_size):
    #             batch_targets = targets[j : j + target_batch_size]
    #             batch_target_ids = destination_ids[j : j + target_batch_size]

    #             # Verify batch size is safe
    #             batch_locations = n_origins + len(batch_targets)
    #             if batch_locations > SAFE_LIMIT:
    #                 raise ValueError(
    #                     f"Strategy 2 batch exceeds limit: {n_origins} origins + {len(batch_targets)} targets = {batch_locations} > {SAFE_LIMIT}"
    #                 )

    #             batch_df = _process_single_batch(
    #                 sources, batch_targets, origin_ids, batch_target_ids, actor, costing
    #             )
    #             results.append(batch_df)
    #             pbar.update(1)

    # Strategy 3: Both large - batch both dimensions aggressively
    # else:
    print("Batching both dimensions")
    # For very large datasets, use smaller fixed batch sizes
    origin_batch_size = 50  # Conservative fixed size
    target_batch_size = 50  # Conservative fixed size

    n_origin_batches = (n_origins + origin_batch_size - 1) // origin_batch_size
    n_target_batches = (n_targets + target_batch_size - 1) // target_batch_size
    total_batches = n_origin_batches * n_target_batches

    with tqdm(total=total_batches, desc="Processing batches", unit="batch") as pbar:
        for i in range(0, n_origins, origin_batch_size):
            batch_sources = sources[i : i + origin_batch_size]
            batch_origin_ids = origin_ids[i : i + origin_batch_size]

            for j in range(0, n_targets, target_batch_size):
                batch_targets = targets[j : j + target_batch_size]
                batch_target_ids = destination_ids[j : j + target_batch_size]

                # Verify this batch won't exceed limit
                batch_locations = len(batch_sources) + len(batch_targets)

                if batch_locations > SAFE_LIMIT:
                    raise ValueError(
                        f"Batch size calculation error: {batch_locations} locations "
                        f"exceeds safe limit of {SAFE_LIMIT}. Try reducing batch sizes."
                    )

                batch_df = _process_single_batch(
                    batch_sources,
                    batch_targets,
                    batch_origin_ids,
                    batch_target_ids,
                    actor,
                    costing,
                )
                results.append(batch_df)
                pbar.update(1)

    # Combine all batches and reset index
    return pd.concat(results, ignore_index=True)


def _calculate_batch_size(primary_size, secondary_size, max_batch_size=2400):
    """
    Calculate optimal batch size for primary dimension.

    Valhalla counts total locations as: primary_batch_size + secondary_size
    Ensures: primary_batch_size + secondary_size <= max_batch_size
    """
    # For Valhalla: total_locations = sources + targets (not their product!)
    max_primary_batch = max_batch_size - secondary_size

    # Ensure at least 1, but cap at primary_size
    return max(1, min(primary_size, max_primary_batch))


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
