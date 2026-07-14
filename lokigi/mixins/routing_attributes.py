import pandas as pd
from typing import Literal, Optional
from pathlib import Path
import warnings
from lokigi.travel_utils import prepare_valhalla_network
from shapely.geometry import Point
import geopandas
from lokigi.utils import _guess_crs
import contextily as cx
import matplotlib.pyplot as plt
import textwrap
from adjustText import adjust_text
import folium
import warnings
from requests.exceptions import RequestException


try:
    from valhalla import Actor
except ImportError as e:
    raise ImportError(
        "Routing support requires optional dependencies.\n"
        "Install with:\n"
        "    pip install lokigi[routing]"
    ) from e


class RoutingAttributeMixin:
    """A mixin class providing geographic routing and service configuration
    for community healthcare modeling.
    """

    def add_routing_data(
        self,
        data,
        output_dir=None,
        output_name=None,
        force_rebuild=False,
    ):
        """
        Register routing data with the object.

        Parameters
        ----------
        data : str | Path | dict
            Either:
                - Path to Valhalla config JSON
                - Path to .osm.pbf
                - Parsed Valhalla config dictionary

        output_dir : str | Path, optional
            Output directory for generated Valhalla files when
            building from .osm.pbf.

        output_name : str, optional
            Base name for generated Valhalla outputs.

        force_rebuild : bool
            Passed to prepare_valhalla_network().
        """

        # -------------------------------------------------------------
        # CASE 1: User passed a config dictionary directly
        # -------------------------------------------------------------
        if isinstance(data, dict):
            self._validate_valhalla_config(data)

            self.routing_data = data
            self.routing_config_path = None

            return

        # -------------------------------------------------------------
        # From here on we expect a filesystem path
        # -------------------------------------------------------------
        path = Path(data)

        if not path.exists():
            raise FileNotFoundError(f"Routing data not found: {path}")

        suffixes = [s.lower() for s in path.suffixes]

        # -------------------------------------------------------------
        # CASE 2: Existing Valhalla config JSON
        # -------------------------------------------------------------
        if path.suffix.lower() == ".json":
            self._validate_valhalla_config(path)

            # Store path rather than whole config for memory efficiency
            self.routing_data = path
            self.routing_config_path = path

            return

        # -------------------------------------------------------------
        # CASE 3: OSM PBF -> build Valhalla tiles
        # -------------------------------------------------------------
        if suffixes[-2:] == [".osm", ".pbf"]:
            results = prepare_valhalla_network(
                osm_path=path,
                output_dir=output_dir,
                output_name=output_name,
                force_rebuild=force_rebuild,
            )

            print(f"Saved to {output_dir}{output_name}.json")

            config_path = Path(results["config_path"])

            # Validate generated config
            self._validate_valhalla_config(config_path)

            print(
                "Prepared Valhalla routing network:\n"
                f"  Config: {results['config_path']}\n"
                f"  Tiles: {results['tile_dir']}\n"
                f"  Traffic: {results['traffic_path']}"
            )

            self.routing_data = config_path
            self.routing_config_path = config_path

            return

        # -------------------------------------------------------------
        # Unsupported input
        # -------------------------------------------------------------
        raise ValueError(
            "Routing data must be one of:\n"
            "  - Valhalla config JSON\n"
            "  - .osm.pbf file\n"
            "  - Valhalla config dictionary"
        )

    @staticmethod
    def _validate_valhalla_config(path):
        try:
            Actor(str(path))
        except Exception as e:
            raise ValueError(f"Invalid Valhalla configuration: {e}") from e

    def add_resources_same_starts(
        self,
        count: int,
        label: str = "Resource_",
        starting_location: str | None = None,
        starting_coordinates: tuple | None = None,
        return_to_start: bool = True,
        max_shift_duration_mins: int | None = 480,
        shift_duration_flex_mins: int = 0,
        lunch_duration_mins: int = 30,
    ) -> None:
        """Define a fleet of identical resources starting from the same hub.

        Parameters
        ----------
        count : int
            The number of individual resources (e.g., nurses, cars) to deploy.

        label : str
            A string that will be used to provide a human-readable differentiator for
            created resources.

        starting_location : str, optional
            A location identifier (e.g., postcode sector, depot ID, or matrix ID)
            representing the shared starting location for all resources.

            Ignored if ``starting_coordinates`` is also provided.

        starting_coordinates : tuple, optional
            A ``(longitude, latitude)`` or ``(x, y)`` coordinate pair defining the
            shared starting location for all resources.

            If provided alongside ``starting_location``, the coordinates take
            precedence.

        return_to_start : bool, default True
            If True, the heuristic closes the loop back to the starting hub.
            If False, the shift ends at the final patient or job location
            (open loop).

        max_shift_duration_mins: int or None, default 480
            The maximum combined driving and activity time

        shift_duration_flex_mins: int, default 0
            The maximum permitted overtime in a shift. Will be used if a final visit would take
            them fractionally over the maximum shift time, but it will first try to fairly assign
            to all resources.

        Warns
        -----
        UserWarning
            If both ``starting_location`` and ``starting_coordinates`` are provided.
            In this case, ``starting_coordinates`` will be used.
        """

        if starting_location is not None and starting_coordinates is not None:
            warnings.warn(
                "Both 'starting_location' and 'starting_coordinates' were "
                "provided. 'starting_coordinates' will be used.",
                UserWarning,
                stacklevel=2,
            )

        if starting_location is None and starting_coordinates is None:
            raise (
                ValueError,
                "Please provide at least one of starting_location or starting_coordinates",
            )

        # ------------------------------------------------------------------
        # Base dataframe
        # ------------------------------------------------------------------

        labels = [f"{label}{i}" for i in range(1, count + 1)]

        df = pd.DataFrame(
            {
                "label": labels,
                "return_to_start": return_to_start,
            }
        )

        # ------------------------------------------------------------------
        # Coordinate-based start locations
        # ------------------------------------------------------------------

        if starting_coordinates is not None:
            x, y = starting_coordinates

            df["x"] = x
            df["y"] = y

            geometry = [Point(x, y)] * count

            # Use preferred CRS if available
            crs = getattr(self, "preferred_crs", None)

            # Otherwise attempt CRS inference
            if crs is None:
                crs = _guess_crs(
                    df=df,
                    x_col="x",
                    y_col="y",
                    verbose=False,
                )

            gdf = geopandas.GeoDataFrame(
                df,
                geometry=geometry,
                crs=crs,
            )

            self.resources_data = gdf
            self.resources_data_type = "geopandas"
            self.num_resources = count

            self.resource_max_shift_duration_mins = max_shift_duration_mins
            self.resource_shift_duration_flex_mins = shift_duration_flex_mins
            self.resource_lunch_duration_mins = lunch_duration_mins

        else:
            raise ValueError(
                "Starting location string not yet supported. Please use coordinates and the starting_coordinates argument instead."
            )

    def add_resources_varying_starts(
        self,
        resource_df: pd.DataFrame,
        id_col: str,
        location_col: str,
        max_shift_duration_mins_col: str | None,
        lunch_duration_mins_col: str | None,  # Assume 0 if not provided
        shift_duration_flex_mins_col: str | None,  # Assume 0 if not provided
        return_to_start: bool = True,
    ) -> None:
        """Define a fleet of resources with individual, unique starting locations.

        This supports scenarios where community staff work directly from home
        or drop-in from localized branch clinics.

        Parameters
        ----------
        resource_df : pd.DataFrame
            A DataFrame containing details of the available staff/resources.
        id_col : str
            The column name in `resource_df` containing unique resource identifiers.
        location_col : str
            The column name containing the starting location ID or postcode
            for each resource.
        return_to_start : bool, default True
            If True, each resource returns to their specific individual starting
            location at the end of their route. If False, the route terminates
            at the last patient visit.
        """
        pass

    def add_exact_demand(
        self,
        demand_df: pd.DataFrame,
        location_col: str,
        timescale: Literal["day", "week", "month", "year"],
        datetime_col: Optional[str] = None,
    ) -> None:
        """Pass a deterministic snapshot of patient visits to be routed.

        Parameters
        ----------
        demand_df : pd.DataFrame
            A DataFrame containing explicit patient case/visit data.
        location_col : str
            The column name in `demand_df` identifying the geographic location
            (e.g., postcode) of each patient.
        timescale : {"day", "week", "month", "year"}
            The temporal scope represented by the entire dataset. Used by
            internal heuristics to segment daily workloads.
        datetime_col : str, optional
            The column name containing timestamps or dates. If provided, allows
            the custom heuristic to slice demand chronologically before routing.
        """
        pass

    def add_service_time_constant(self, minutes: float) -> None:
        """Set a uniform, fixed appointment duration across all patient visits.

        Parameters
        ----------
        minutes : float
            The constant time spent at each patient location, excluding travel.
        """
        pass

    def add_service_time_sampled(
        self,
        distribution: Literal["normal", "lognormal", "exponential", "poisson"],
        **kwargs,
    ) -> None:
        """Configure appointment durations to be stochastically sampled for DES.

        Parameters
        ----------
        distribution : {"normal", "lognormal", "exponential", "poisson"}
            The statistical distribution used to model clinical visit variation.
        **kwargs
            Keyword arguments mapped directly to the chosen distribution's
            parameters (e.g., `mean=45.0`, `stdev=10.0` or `lam=45.0`).
        """
        pass

    def add_service_time_defined(self, demand_df_col: str) -> None:
        """Map appointment durations directly to an existing column in the demand data.

        This is used when different patient types require drastically different
        visiting lengths (e.g., complex wound dressing vs. quick checks).

        Parameters
        ----------
        demand_df_col : str
            The column name within the registered demand DataFrame that holds
            the pre-calculated visit lengths in minutes.
        """
        pass

    def add_resource_constraints(
        self,
        max_shift_length: float,
        max_visits_per_shift: Optional[int] = None,
        max_travel_time: Optional[float] = None,
    ) -> None:
        """Set boundaries that force custom heuristics to break routes and trigger new shifts.

        Parameters
        ----------
        max_shift_length : float
            The maximum allowable duration of a shift in minutes, representing
            the hard ceiling for (`Total Travel Time` + `Total Service Time`).
        max_visits_per_shift : int, optional
            The upper limit on the number of patients an individual resource
            can see in a single day.
        max_travel_time : float, optional
            The maximum total minutes a resource can spend purely driving or traveling
            per shift. Used to protect staff from rural burnout.
        """
        pass

        def estimate_routes():
            pass

    def show_resources(self):
        return self.resources_data

    def plot_resources(self, add_basemap=True, show_labels=True, interactive=False):
        """
        Generate a visualization of the resource locations.

        If multiple resources share identical coordinates, they are grouped
        together and displayed with a count.

        Parameters
        ----------
        add_basemap : bool, default True
            If True, adds a contextily basemap to static plots.
        show_labels : bool, default True
            If True, displays labels/counts on static plots.
        interactive : bool, default False
            If True, returns an interactive Folium map.

        Returns
        -------
        matplotlib.axes.Axes or folium.Map
        """

        # ------------------------------------------------------------------
        # Aggregate coincident points
        # ------------------------------------------------------------------
        grouped = (
            self.resources_data.groupby(
                self.resources_data.geometry.to_wkt(), dropna=False
            )
            .agg(
                geometry=("geometry", "first"),
                count=("label", "size"),
                labels=("label", lambda x: list(x)),
            )
            .reset_index(drop=True)
        )

        # Preserve CRS
        grouped = geopandas.GeoDataFrame(
            grouped, geometry="geometry", crs=self.resources_data.crs
        )

        # Create display label
        grouped["display_label"] = grouped.apply(
            lambda row: (
                f"{row['count']} resources" if row["count"] > 1 else row["labels"][0]
            ),
            axis=1,
        )

        # ------------------------------------------------------------------
        # Static plot
        # ------------------------------------------------------------------
        if not interactive:
            # Scale marker size by count
            grouped["marker_size"] = 50 + grouped["count"] * 25

            ax = grouped.plot(
                figsize=(10, 10),
                markersize=grouped["marker_size"],
            )

            if show_labels:
                texts = []

                for x, y, label in zip(
                    grouped.geometry.x,
                    grouped.geometry.y,
                    grouped["display_label"],
                ):
                    wrapped_label = textwrap.fill(str(label), 15).title()

                    texts.append(
                        plt.text(
                            x,
                            y,
                            wrapped_label,
                            ha="center",
                            va="bottom",
                        )
                    )

                adjust_text(texts, force_explode=(0.05, 0.05))

            if add_basemap:
                try:
                    cx.add_basemap(ax, crs=grouped.crs.to_string(), timeout=30)
                except RequestException as e:
                    warnings.warn(
                        f"Unable to download background map tiles ({type(e).__name__}). "
                        "Continuing without a basemap.",
                        stacklevel=2,
                    )

            return ax

        # ------------------------------------------------------------------
        # Interactive plot
        # ------------------------------------------------------------------
        else:
            grouped["popup"] = grouped.apply(
                lambda row: (
                    "<br>".join(row["labels"])
                    if row["count"] <= 5
                    else f"{row['count']} resources"
                ),
                axis=1,
            )

            m = folium.Map(
                location=[
                    grouped.geometry.y.mean(),
                    grouped.geometry.x.mean(),
                ],
                zoom_start=8,
            )

            for _, row in grouped.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5 + row["count"],
                    tooltip=row["display_label"],
                    popup=row["popup"],
                    fill=True,
                ).add_to(m)

            return m
