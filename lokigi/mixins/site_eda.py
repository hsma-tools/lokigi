import esda
from pysal.lib import weights
import warnings
from typing import Literal

NeighbourhoodMethod = Literal[
    "rook",
    "queen",
    "k-nearest",
]

ClusterDisplay = Literal[
    "all",
    "hotspots",
    "coldspots",
    "outliers",
]

SupportedInputs = Literal["demand", "equity"]


class SiteEDAMixin:
    def _get_weights(
        self,
        df,
        neighbourhood_method: NeighbourhoodMethod = "rook",
        k: int | None = None,
        verbose: bool = True,
        return_weights: bool = True,
        force_recalculation: bool = True,
    ):
        """
        Create and cache a spatial weights matrix.

        Generates a PySAL spatial weights object based on the region
        geometry layer attached to the problem. The resulting weights
        matrix is row-standardised and cached for reuse by exploratory
        spatial data analysis methods.

        Parameters
        ----------
        neighbourhood_method : {"rook", "queen", "k-nearest"}, default="rook"
            Method used to define neighbouring regions.

            * ``"rook"``: Regions sharing an edge are considered neighbours.
            * ``"queen"``: Regions sharing an edge or vertex are considered
            neighbours.
            * ``"k-nearest"``: Each region is connected to its ``k`` nearest
            neighbours.

        k : int, optional
            Number of neighbours to use when ``neighbourhood_method="k-nearest"``.
            Ignored for rook and queen contiguity.

        Raises
        ------
        ValueError
            If ``neighbourhood_method="k-nearest"`` and ``k`` is not provided.

        Notes
        -----
        The resulting weights matrix is row-standardised using
        ``w.transform = "R"``.

        The weights matrix and metadata describing how it was generated
        are stored on the instance as:

        * ``self.spatial_weights``
        * ``self.spatial_weights_method``
        * ``self.spatial_weights_k``
        """
        valid_neighbourhood_method = ["rook", "queen", "k-nearest"]

        if neighbourhood_method not in valid_neighbourhood_method:
            raise ValueError(
                f"neighbourhood_method must be one of {valid_neighbourhood_method}, got {neighbourhood_method!r}"
            )

        # We look at the hash for the entire region geometry as we're just trying to capture
        # if the region geometry has changed, not whether it's the same region requested as
        # before
        current_hash = self._get_geometry_hash(self.region_geometry_layer)

        if (
            self.spatial_weights is None
            or neighbourhood_method != self.spatial_weights_method
            or k != self.spatial_weights_k
            or current_hash != self._region_geometry_hash
            or force_recalculation
        ):
            if neighbourhood_method == "queen":
                w = weights.Queen.from_dataframe(df)
            elif neighbourhood_method == "rook":
                w = weights.Rook.from_dataframe(df)
            elif neighbourhood_method == "k-nearest":
                if k is None:
                    raise ValueError(
                        "Please specify a value of k in get_hotspots if you want to use k-nearest neighbours"
                    )
                w = weights.KNN.from_dataframe(df, k=k)

            w.transform = "R"

            self.spatial_weights = w
            self.spatial_weights_method = neighbourhood_method
            self.spatial_weights_k = k
            self._region_geometry_hash = current_hash
        else:
            if verbose:
                warnings.warn(
                    "Using cached spatial weights.",
                    stacklevel=2,
                )

        if return_weights:
            return self.spatial_weights

    def get_hotspots(
        self,
        what: SupportedInputs = "demand",
        neighbourhood_method: NeighbourhoodMethod = "rook",
        k: int | None = None,
        verbose: bool = True,
        significance_threshold: float = 0.05,
        force_weight_recalculation: bool = False,
    ):
        """
        Identify statistically significant hotspots and coldspots.

        Returns hotspot, coldspot, outlier and non-significant classifications
        derived from Local Moran's I.

        Performs local spatial autocorrelation analysis on a selected
        variable using the specified spatial weights matrix. Areas with
        significantly high values surrounded by high values are classified
        as hotspots, while areas with significantly low values surrounded
        by low values are classified as coldspots.

        Parameters
        ----------
        what : str, default="demand"
            Variable to analyse. This may be the name of a stored dataset
            or metric associated with the problem, such as demand,
            accessibility, or equity measures.

        neighbourhood_method : {"rook", "queen", "k-nearest"}, default="rook"
            Method used to define neighbouring regions.

        k : int, optional
            Number of neighbours to use when ``neighbourhood_method="k-nearest"``.
            Required if ``neighbourhood_method="k-nearest"``.

        Returns
        -------
        geopandas.GeoDataFrame
            Region geometry layer with additional columns describing
            hotspot/coldspot classification and associated statistics.

        Notes
        -----
        Spatial weights are generated when required and cached for
        reuse. Cached weights are automatically invalidated when the
        region geometry or weighting parameters change.
        """
        # Build the weights if they've not been generated before or if anything about
        # the weight parameters or region geometry layer have changed

        # Demand is assumed to only be passed for the region of interest.
        # In effect, it's expected to define the region of interest!
        # And this holds if you are generating a problem simply from a travel matrix without
        # passing in demand (although then you're not going to be looking for demand hotspots,
        # though you could be interested in equity hotspots)
        # However, there's a good chance you'll have passed an unfiltered whole-country
        # equity dataset, so this needs to filter down too.

        if what == "demand":
            df = self.demand_data
            df_col = self._demand_data_demand_col
            df_merge_col = self._demand_data_id_col
        elif what == "equity":
            df = self.equity_data.copy()
            df_col = self._equity_data_equity_col
            df_merge_col = self._equity_data_common_col
            df = df[
                df[df_merge_col].isin(
                    self.demand_data[self._demand_data_id_col].unique()
                )
            ]

        print(f"DF length: {len(df)}")

        # Filter down to only geometry regions that are actually present
        result = df.copy()

        filtered_region_geometry = self.region_geometry_layer.copy()
        filtered_region_geometry = filtered_region_geometry[
            filtered_region_geometry[self._region_geometry_layer_common_col].isin(
                self.demand_data[self._demand_data_id_col].unique()
            )
        ]

        print(f"Length of filtered region geometry: {len(filtered_region_geometry)}")

        # The weights need to be the same dimension as the dataset of interest, so we can't calculate
        # them once upfront; they need to be done on the filtered dataset
        w = self._get_weights(
            df=filtered_region_geometry,
            neighbourhood_method=neighbourhood_method,
            k=k,
            verbose=verbose,
            return_weights=True,
            force_recalculation=force_weight_recalculation,
        )

        # Calculate Local Moran’s I
        lisa = esda.moran.Moran_Local(result[df_col], w)

        result["local_moran_i"] = lisa.Is
        result["p_value"] = lisa.p_sim
        result["quadrant"] = lisa.q

        result["cluster_type"] = "Not Significant"

        significant = lisa.p_sim < significance_threshold

        result.loc[
            significant & (lisa.q == 1),
            "cluster_type",
        ] = "Hotspot"

        result.loc[
            significant & (lisa.q == 3),
            "cluster_type",
        ] = "Coldspot"

        result.loc[
            significant & (lisa.q == 2),
            "cluster_type",
        ] = "Low-High Outlier"

        result.loc[
            significant & (lisa.q == 4),
            "cluster_type",
        ] = "High-Low Outlier"

        return self.region_geometry_layer.merge(
            result,
            left_on=self._region_geometry_layer_common_col,
            right_on=df_merge_col,
        )

    def plot_hotspots():
        pass
