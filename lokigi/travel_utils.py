from valhalla import Actor
import pandas as pd

from pathlib import Path
import subprocess

import osmium
import re


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
    origins_gdf, destinations_gdf, valhalla_config_path, costing="auto"
):

    actor = Actor(valhalla_config_path)

    sources = [{"lat": geom.y, "lon": geom.x} for geom in origins_gdf.geometry]

    targets = [{"lat": geom.y, "lon": geom.x} for geom in destinations_gdf.geometry]

    request = {
        "sources": sources,
        "targets": targets,
        "costing": costing,
        "matrix_locations": len(sources) + len(targets),
        "verbose": False,
    }

    matrix = actor.matrix(request)

    rows = []

    durations = matrix["sources_to_targets"]["durations"]
    distances = matrix["sources_to_targets"]["distances"]

    for i, origin in enumerate(origins_gdf.itertuples()):
        for j, destination in enumerate(destinations_gdf.itertuples()):
            # Access durations and distances from the correct structure
            travel_time_seconds = durations[i][j]
            distance_meters = distances[i][j]

            rows.append(
                {
                    "from_id": origin.id,
                    "to_id": destination.id,
                    "travel_time_minutes": travel_time_seconds / 60
                    if travel_time_seconds
                    else None,
                    "distance_km": distance_meters if distance_meters else None,
                }
            )

    return pd.DataFrame(rows)


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
