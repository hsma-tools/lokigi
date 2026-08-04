import warnings

import contextily as cx
import geopandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from requests.exceptions import RequestException

from lokigi.plot_utils import _add_plot_caption, _attach_deferred_fit_bounds
from lokigi.utils import _min_max_normalize, _resolve_site_numeric_column


class SFCAMixin:
    """
    Two-step floating catchment area (2SFCA) accessibility.

    Mixed into `SiteProblem`, giving it `two_step_floating_catchment()`.
    `SiteSolutionSet.two_step_floating_catchment()` resolves a solution's
    site list and delegates back to the `SiteProblem` method rather than
    duplicating this engine, since it has no travel frame of its own beyond
    what the problem already builds.
    """

    @staticmethod
    def _resolve_supply(candidate_sites, candidate_id_col, supply_col, site_names):
        return _resolve_site_numeric_column(
            candidate_sites,
            candidate_id_col,
            supply_col,
            site_names,
            param_name="supply_col",
            quantity_label="supply quantity",
            allow_missing=False,
        )

    @staticmethod
    def _resolve_catchment_weights(cost_frame, catchment_size, distance_decay):
        """
        Returns a `pandas.DataFrame` aligned to `cost_frame`, giving each
        (demand location, site) pair's catchment weight in `[0, 1]`.

        Exactly one of `catchment_size` (classic 2SFCA's hard cutoff) or
        `distance_decay` (E2SFCA step-decay bands or a continuous kernel)
        must be given.
        """
        if (catchment_size is None) == (distance_decay is None):
            raise ValueError(
                "Please provide exactly one of 'catchment_size' (a single "
                "hard cutoff) or 'distance_decay' (step-decay bands or a "
                "continuous decay spec) -- not both, not neither."
            )

        if catchment_size is not None:
            return (cost_frame <= catchment_size).astype(float)

        if isinstance(distance_decay, dict):
            return SFCAMixin._resolve_continuous_decay_weights(
                cost_frame, distance_decay
            )
        return SFCAMixin._resolve_step_decay_weights(cost_frame, distance_decay)

    @staticmethod
    def _resolve_step_decay_weights(cost_frame, bands):
        """
        Enhanced 2SFCA (E2SFCA, Luo & Qi 2009) step-decay weighting.

        `bands` is a list of `(upper_bound, weight)` pairs, need not be
        pre-sorted. A cost is assigned the weight of the smallest band whose
        `upper_bound` it falls within (inclusive `<=`, matching
        `catchment_size`'s convention); a cost beyond every band's
        `upper_bound` gets weight `0.0` -- excluded from the catchment
        entirely, the same as "beyond catchment_size" today.
        """
        bands = list(bands)
        if not bands:
            raise ValueError(
                "distance_decay must contain at least one (upper_bound, "
                "weight) band."
            )

        try:
            bands = sorted((float(upper), float(weight)) for upper, weight in bands)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "distance_decay must be a list of (upper_bound, weight) "
                f"pairs. Got: {bands!r}."
            ) from e

        upper_bounds = [upper for upper, _ in bands]
        if len(set(upper_bounds)) != len(upper_bounds):
            raise ValueError(
                f"distance_decay band upper bounds must be unique: {upper_bounds}."
            )

        non_positive = [upper for upper in upper_bounds if upper <= 0]
        if non_positive:
            raise ValueError(
                f"distance_decay upper bounds must be positive: {non_positive}."
            )

        negative_weights = [weight for _, weight in bands if weight < 0]
        if negative_weights:
            raise ValueError(
                f"distance_decay weights must be non-negative: {negative_weights}."
            )

        weight_matrix = pd.DataFrame(
            0.0, index=cost_frame.index, columns=cost_frame.columns
        )
        assigned = pd.DataFrame(
            False, index=cost_frame.index, columns=cost_frame.columns
        )
        for upper, weight in bands:
            band_mask = (cost_frame <= upper) & ~assigned
            weight_matrix[band_mask] = weight
            assigned |= band_mask

        return weight_matrix

    @staticmethod
    def _resolve_continuous_decay_weights(cost_frame, spec):
        """
        Continuous distance-decay weighting. `spec` is a dict naming a
        `method` and that method's parameters; `"gaussian"` (Dai 2010) and
        `"power"` (the classic gravity-model decay) are supported.
        """
        method = spec.get("method")
        if method == "gaussian":
            return SFCAMixin._resolve_gaussian_decay_weights(cost_frame, spec)
        if method == "power":
            return SFCAMixin._resolve_power_decay_weights(cost_frame, spec)

        raise ValueError(
            f"Unknown distance_decay method {method!r}. Supported: "
            "'gaussian', 'power'."
        )

    @staticmethod
    def _resolve_gaussian_decay_weights(cost_frame, spec):
        """
        Dai (2010)'s truncated/adjusted Gaussian decay: weight is exactly
        `1.0` at distance 0, exactly `0.0` at `catchment_size` (the
        truncation radius d0), and decays continuously in between --
        unlike a raw untruncated Gaussian, which never reaches 0.

            d0_term = exp(-0.5 * (d0/bandwidth)^2)
            weight(d) = (exp(-0.5 * (d/bandwidth)^2) - d0_term) / (1 - d0_term)   for d <= d0
            weight(d) = 0                                                          for d > d0
        """
        missing = [k for k in ("catchment_size", "bandwidth") if k not in spec]
        if missing:
            raise ValueError(
                "distance_decay with method='gaussian' requires "
                f"{missing}: e.g. {{'method': 'gaussian', 'catchment_size': "
                "30, 'bandwidth': 10}}."
            )

        catchment_size = spec["catchment_size"]
        bandwidth = spec["bandwidth"]

        if catchment_size <= 0:
            raise ValueError(
                f"distance_decay['catchment_size'] must be positive: {catchment_size}."
            )
        if bandwidth <= 0:
            raise ValueError(
                f"distance_decay['bandwidth'] must be positive: {bandwidth}."
            )

        d0_term = np.exp(-0.5 * (catchment_size / bandwidth) ** 2)
        raw = np.exp(-0.5 * (cost_frame / bandwidth) ** 2)
        weight_matrix = (raw - d0_term) / (1 - d0_term)
        weight_matrix = weight_matrix.where(cost_frame <= catchment_size, 0.0)

        return weight_matrix

    @staticmethod
    def _resolve_power_decay_weights(cost_frame, spec):
        """
        The classic gravity-model decay (Huff, 1963; the `d_ij^-beta` term
        in the gravity-based accessibility index -- e.g. Luo & Qi (2009)'s
        Eq. (1)), truncated to 0 beyond `catchment_size`:

            weight(d) = (max(d, min_dist) / scale) ** alpha    for d <= catchment_size
            weight(d) = 0                                       for d > catchment_size

        `alpha` is not implicitly negative -- pass a negative `alpha` for
        decay (weight falling with distance). `min_dist` guards the
        singularity at `d=0` for negative `alpha` (default 0, matching a
        raw inverse-distance weight, but a cost of exactly 0 combined with
        `min_dist=0` and negative `alpha` raises rather than silently
        producing an infinite weight). Parameterisation matches pysal/
        access's `weights.gravity(scale, alpha, min_dist)`.
        """
        missing = [k for k in ("catchment_size", "scale", "alpha") if k not in spec]
        if missing:
            raise ValueError(
                "distance_decay with method='power' requires "
                f"{missing}: e.g. {{'method': 'power', 'catchment_size': "
                "30, 'scale': 1, 'alpha': -1}}."
            )

        catchment_size = spec["catchment_size"]
        scale = spec["scale"]
        alpha = spec["alpha"]
        min_dist = spec.get("min_dist", 0)

        if catchment_size <= 0:
            raise ValueError(
                f"distance_decay['catchment_size'] must be positive: {catchment_size}."
            )
        if scale <= 0:
            raise ValueError(f"distance_decay['scale'] must be positive: {scale}.")
        if min_dist < 0:
            raise ValueError(
                f"distance_decay['min_dist'] must be non-negative: {min_dist}."
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            weight_matrix = np.power(cost_frame.clip(lower=min_dist) / scale, alpha)
        weight_matrix = weight_matrix.where(cost_frame <= catchment_size, 0.0)

        if np.isinf(weight_matrix.to_numpy()).any():
            raise ValueError(
                "distance_decay with method='power' produced an infinite "
                "weight -- a cost of 0 combined with min_dist=0 and a "
                "negative alpha. Set 'min_dist' to a small positive value."
            )

        return weight_matrix

    @staticmethod
    def _two_step_floating_catchment(
        cost_frame, demand, supply, weight_matrix, per_capita, return_site_ratios
    ):
        """
        Parameters
        ----------
        cost_frame : pandas.DataFrame
            Wide travel-cost frame: index = demand location IDs, columns =
            site names, matching `supply.index`.
        demand : pandas.Series
            Demand weight per demand location, aligned to `cost_frame.index`.
        supply : pandas.Series
            Supply quantity per site, aligned to `cost_frame.columns`.
        weight_matrix : pandas.DataFrame
            Catchment weight per (demand location, site) pair, aligned to
            `cost_frame`, values in `[0, 1]`. A cell of `1.0` means "fully in
            catchment" (classic 2SFCA's hard cutoff); `0.0` means "excluded
            entirely"; anything in between is a distance-decay weight
            (E2SFCA step bands or a continuous kernel). See
            `_resolve_catchment_weights`.
        per_capita : float
            Multiplier applied to `accessibility`, e.g. 1_000 to express
            supply per 1,000 head instead of raw supply units per head.
        return_site_ratios : bool
            If True, also return the step-1 per-site table.

        Returns
        -------
        pandas.DataFrame or tuple of (pandas.DataFrame, pandas.DataFrame)
        """
        # Step 1: supply-to-demand ratio per site. A matrix-vector dot
        # product (BLAS gemv under the hood) rather than an elementwise
        # multiply-then-sum: no N x M intermediate array, and materially
        # faster once scoring hundreds of sites against thousands of
        # regions. Safe against NaN propagating silently different from
        # .mul().sum()'s skipna=True, because weight_matrix can never
        # itself contain NaN -- every _resolve_*_decay_weights path builds
        # it from a `cost_frame <= threshold`-style boolean comparison,
        # and NaN comparisons are always False, so a NaN travel cost
        # always resolves to weight 0.0 (see
        # test_nan_travel_cost_treated_as_unreachable_not_propagated).
        catchment_demand = weight_matrix.T.dot(demand)
        n_regions_in_catchment = (weight_matrix > 0).sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = supply / catchment_demand
        # A site with nobody in its catchment divides by zero either way
        # (0/0 or S/0): both mean "this site's ratio is undefined", so both
        # collapse to NaN rather than one becoming +/-inf.
        ratio = ratio.replace([np.inf, -np.inf], np.nan)

        empty_sites = catchment_demand[catchment_demand == 0].index.tolist()
        if empty_sites:
            warnings.warn(
                "two_step_floating_catchment: the following sites have no "
                "demand within their catchment, so their supply-to-demand "
                "ratio is undefined (NaN) and they are excluded from every "
                f"region's accessibility score: {empty_sites}.",
                stacklevel=3,
            )

        site_frame = pd.DataFrame(
            {
                "supply": supply,
                "catchment_demand": catchment_demand,
                "n_regions_in_catchment": n_regions_in_catchment,
                "ratio": ratio,
            }
        )
        # catchment_demand/n_regions_in_catchment/ratio all derive their
        # index from cost_frame.columns, which carries no name -- without
        # this, pd.DataFrame(dict-of-Series) drops supply.index.name too,
        # silently turning site_frame.reset_index()'s site-name column
        # into a generic "index" rather than the candidate ID column name.
        site_frame.index.name = supply.index.name

        # Step 2: sum reachable sites' ratios per demand region. A NaN
        # ratio (empty-catchment site) must drop out of this sum rather
        # than propagate -- an unreachable site's undefined ratio
        # shouldn't poison every other region that happens to be near it
        # too. fillna(0.0) here is also what keeps this dot product safe;
        # see the comment on catchment_demand above.
        usable_ratio = ratio.fillna(0.0)
        accessibility = weight_matrix.dot(usable_ratio) * per_capita
        n_sites_in_catchment = (weight_matrix > 0).sum(axis=1)

        region_frame = pd.DataFrame(
            {
                "accessibility": accessibility,
                "n_sites_in_catchment": n_sites_in_catchment,
                "demand": demand,
            }
        )

        if return_site_ratios:
            return region_frame, site_frame
        return region_frame

    def two_step_floating_catchment(
        self,
        supply_col,
        catchment_size=None,
        distance_decay=None,
        site_names=None,
        site_indices=None,
        matrix=None,
        demand=None,
        per_capita=1,
        return_site_ratios=False,
    ):
        """
        Two-step floating catchment area (2SFCA) accessibility score.

        Unlike binary threshold coverage, 2SFCA accounts for competition:
        a demand region near a site that is also the closest option for
        many other regions gets less credit for that site than a region
        near an uncontested one. Scores an arbitrary set of sites directly
        -- no `solve()` is required, so it can describe any subset of
        `candidate_sites` as a baseline, e.g. only the sites that are
        already open, via `site_names`/`site_indices`. With neither
        argument, every registered candidate site is scored -- the right
        default when the candidate pool *is* the current network, but not
        when it also includes not-yet-built proposals, in which case pass
        the currently-open subset explicitly.

        Parameters
        ----------
        supply_col : str
            Column in `candidate_sites` holding each site's supply
            quantity (e.g. number of GPs, beds, weekly appointment slots).
            Named at call time rather than registered via `add_sites()`,
            so the same problem can be scored under different supply
            definitions without re-adding sites.
        catchment_size : float, optional
            Classic 2SFCA's hard catchment threshold d0, in the travel
            matrix's registered units. A site is "in catchment" for a
            demand region if the travel cost between them is
            `<= catchment_size`, with weight 1; beyond it, weight 0.
            Mutually exclusive with `distance_decay` -- exactly one of the
            two must be given.
        distance_decay : list of (float, float) or dict, optional
            A softer catchment than a single hard cutoff. Two forms:

            - A list of `(upper_bound, weight)` pairs -- Enhanced 2SFCA
              (E2SFCA, Luo & Qi 2009) step-decay bands, e.g.
              ``[(10, 1.0), (20, 0.68), (30, 0.22)]`` -- this is not just an
              illustrative example: it is Luo & Qi's own published "weight
              set 1" for 0-10/10-20/20-30 minute zones (their sharper
              "weight set 2" is ``[(10, 1.0), (20, 0.42), (30, 0.09)]``).
              Need not be pre-sorted; a cost beyond the largest
              `upper_bound` gets weight 0. `catchment_size=<upper_bound>`
              with a single band of weight 1 is exactly equivalent to
              classic 2SFCA.
            - A dict describing a continuous decay kernel. Two forms:
              ``{"method": "gaussian", "catchment_size": d0, "bandwidth":
              sigma}`` -- Dai (2010)'s truncated Gaussian: weight 1 at
              distance 0, weight 0 at `catchment_size` (the truncation
              radius), decaying continuously in between; or
              ``{"method": "power", "catchment_size": d0, "scale": s,
              "alpha": a, "min_dist": m=0}`` -- the classic gravity-model
              decay, ``weight(d) = (max(d, min_dist)/scale)**alpha``,
              truncated to 0 beyond `catchment_size` (`alpha` is not
              implicitly negative; pass a negative value for decay).
              Parameterisation matches pysal/access's
              ``weights.gravity(scale, alpha, min_dist)``.
        site_names : list of str, optional
        site_indices : list of int, optional
            The site set to score. At most one may be given. If neither is
            given, every candidate site is scored.
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Scores accessibility under
            that matrix's travel costs instead of the primary matrix's.
        demand : str, optional
            Label of a secondary demand scenario registered via
            `add_secondary_demand()`. Scores accessibility under that
            scenario's demand instead of the primary demand data (both
            `catchment_demand` and `demand` in the returned frames reflect
            the chosen scenario). Combines freely with `matrix=`.
        per_capita : float, default 1
            Multiplier applied to the `accessibility` column, e.g. 1_000
            to express supply per 1,000 head instead of raw supply units
            per head. Does not affect `site_frame`. Must be a positive
            number.
        return_site_ratios : bool, default False
            If True, also return the step-1 per-site table -- useful for
            finding which site is driving an implausible regional score.

        Returns
        -------
        pandas.DataFrame
            Per demand region, indexed by demand location ID: `accessibility`
            (supply units per head, x `per_capita`), `n_sites_in_catchment`,
            `demand`.
        pandas.DataFrame
            Only if `return_site_ratios=True`. Per site, indexed by site
            name: `supply`, `catchment_demand`, `n_regions_in_catchment`,
            `ratio`.

        Raises
        ------
        ValueError
            If both `site_names` and `site_indices` are given, if
            `per_capita` is not a positive number, if neither or both of
            `catchment_size`/`distance_decay` are given, if no demand data
            is registered, if `supply_col` is missing or contains a
            null/negative value for a scored site, if `matrix` is not a
            registered secondary travel matrix label, if `demand` is not a
            registered secondary demand scenario label, or if
            `distance_decay` fails its own validation (see above).
        TypeError
            If `supply_col` is not numeric.
        IndexError
            If `site_indices` are out of range.
        KeyError
            If `site_names` are not found in the travel matrix or
            `candidate_sites`.

        Notes
        -----
        A site with no demand within its catchment has an undefined
        (`NaN`) supply-to-demand ratio and is excluded from every region's
        accessibility score, with a warning naming it. A demand region with
        no site in its catchment correctly scores `accessibility == 0`
        (a real "no supply available" result, not a missing value) --
        these two zero-like cases are deliberately kept distinguishable.
        `n_regions_in_catchment`/`n_sites_in_catchment` count non-zero-weight
        membership, so this holds under `distance_decay` too.

        `accessibility` obeys `sum(demand * accessibility) == sum(supply)`
        (before `per_capita` scaling) whenever every scored site has at
        least one region in its catchment, since the sum of demand-weighted
        ratios is exactly the supply that produced them -- true regardless
        of whether `catchment_size` or `distance_decay` was used.
        """
        if site_names is not None and site_indices is not None:
            raise ValueError(
                "Please provide at most one of 'site_names' or "
                "'site_indices', not both."
            )

        if not isinstance(per_capita, (int, float)) or isinstance(per_capita, bool):
            raise ValueError(
                f"per_capita must be a positive number, got {per_capita!r}."
            )
        if per_capita <= 0:
            raise ValueError(
                f"per_capita must be a positive number, got {per_capita}. A "
                "value of 0 would zero out every region's accessibility "
                "score regardless of supply or demand."
            )

        if self._demand_data_demand_col is None:
            raise ValueError(
                "two_step_floating_catchment requires demand data. Call "
                "add_demand() before using this method."
            )

        if self.travel_and_demand_df is None:
            self._create_joined_demand_travel_df(index_col=self._demand_data_id_col)
            self._build_secondary_travel_frames()
            self._build_secondary_demand_frames()

        if site_indices is not None:
            if len(site_indices) != len(set(site_indices)):
                raise ValueError(
                    f"site_indices contains duplicate entries: {site_indices}."
                )
            valid_indices = set(self.candidate_sites["canonical_site_index"])
            invalid_indices = sorted(set(site_indices) - valid_indices)
            if invalid_indices:
                raise IndexError(
                    f"Site indices {invalid_indices} not found in candidate "
                    f"sites (valid range: 0 to {self.total_n_sites - 1})."
                )
            site_names_resolved = (
                self.candidate_sites[
                    self.candidate_sites["canonical_site_index"].isin(site_indices)
                ]
                .sort_values("canonical_site_index")[self._candidate_sites_candidate_id_col]
                .tolist()
            )
        elif site_names is not None:
            site_names_resolved = list(site_names)
        else:
            site_names_resolved = (
                self.candidate_sites.sort_values("canonical_site_index")[
                    self._candidate_sites_candidate_id_col
                ]
                .tolist()
            )

        if matrix is None:
            missing = [
                s for s in site_names_resolved if s not in self.travel_and_demand_df.columns
            ]
            if missing:
                raise KeyError(f"Sites not found in travel matrix: {missing}")
            cost_frame = self.travel_and_demand_df[site_names_resolved]
        else:
            if matrix not in self.secondary_travel_matrices:
                raise ValueError(
                    f"Unknown secondary travel matrix '{matrix}'. Registered "
                    f"labels: {sorted(self.secondary_travel_matrices)}."
                )
            frame = self._secondary_travel_frames[matrix]
            missing = [s for s in site_names_resolved if s not in frame.columns]
            if missing:
                raise KeyError(
                    f"Sites not found in secondary travel matrix '{matrix}': {missing}"
                )
            cost_frame = frame[site_names_resolved]

        if demand is None:
            demand_series = self.travel_and_demand_df[
                self._demand_data_demand_col
            ].reindex(cost_frame.index)
        else:
            if demand not in self.secondary_demand_matrices:
                raise ValueError(
                    f"Unknown secondary demand scenario '{demand}'. Registered "
                    f"labels: {sorted(self.secondary_demand_matrices)}."
                )
            demand_series = self._secondary_demand_frames[demand].reindex(
                cost_frame.index
            )

        supply = self._resolve_supply(
            self.candidate_sites,
            self._candidate_sites_candidate_id_col,
            supply_col,
            site_names_resolved,
        )

        weight_matrix = self._resolve_catchment_weights(
            cost_frame, catchment_size, distance_decay
        )

        return self._two_step_floating_catchment(
            cost_frame=cost_frame,
            demand=demand_series,
            supply=supply,
            weight_matrix=weight_matrix,
            per_capita=per_capita,
            return_site_ratios=return_site_ratios,
        )


class AccessibilityPlotMixin:
    """
    Shared 2SFCA accessibility map for `SiteProblem` and `SiteSolutionSet`.

    Mirrors the `get_hotspots()`/`plot_hotspots()` split in `site_eda.py`:
    the plotting method optionally takes a precomputed result, computing it
    automatically via `two_step_floating_catchment()` if not supplied.
    """

    @property
    def _prob_ctx(self):
        """
        Dynamically routes requests to the object holding the problem data.
        Returns `self.site_problem` if attached to a `SiteSolutionSet`,
        otherwise `self` (already a `SiteProblem`).
        """
        return getattr(self, "site_problem", self)

    def plot_accessibility(
        self,
        region_frame=None,
        site_frame=None,
        supply_col=None,
        catchment_size=None,
        distance_decay=None,
        site_names=None,
        site_indices=None,
        matrix=None,
        per_capita=1,
        interactive=False,
        cmap="Blues",
        show_site_ratio=False,
        site_colour="black",
        site_marker_size=60,
        site_cmap="RdYlGn",
        missing_site_colour="lightgrey",
        marker_size_range=(40, 220),
        edgecolor="black",
        linewidth=0.5,
        tiles="CartoDB positron",
        add_basemap=True,
        show_axis=False,
        title=None,
        caption=None,
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """
        Map 2SFCA accessibility: a region choropleth of `accessibility`,
        overlaid with site markers. By default the markers are plain,
        uniformly coloured/sized location dots -- pass
        `show_site_ratio=True` to instead colour and size them by their
        step-1 supply-to-demand `ratio`, so a map reader can see both where
        access is poor and which site is driving it (an overloaded,
        low-ratio site drawn small and red) in one view. The ratio view is
        opt-in because it's easy to misread as a raw capacity/
        overutilisation metric, when it's actually a relative figure
        specific to the chosen `catchment_size`/`distance_decay`.

        Parameters
        ----------
        region_frame, site_frame : pandas.DataFrame, optional
            The two tables returned by
            `two_step_floating_catchment(return_site_ratios=True)`. If
            either is None (the default), both are computed automatically
            from `supply_col`/`catchment_size` and the selection arguments
            below.
        supply_col, catchment_size, distance_decay, site_names, site_indices,
        matrix, per_capita
            Forwarded to `two_step_floating_catchment()` when
            `region_frame`/`site_frame` are not supplied; see that
            method's docstring (`catchment_size`/`distance_decay` are
            mutually exclusive there too). On a `SiteSolutionSet`, this
            always scores `solution_rank=1` -- to plot a specific
            `sort_by`/`solution_rank` solution, call
            `two_step_floating_catchment(..., return_site_ratios=True)`
            yourself and pass the results in as `region_frame`/`site_frame`.
        interactive : bool, default False
            If True, returns an interactive Folium map via `.explore()`.
            Otherwise returns a static matplotlib Axes.
        cmap : str, default "Blues"
            Colormap for the region choropleth (`accessibility`).
        show_site_ratio : bool, default False
            If False (the default), site markers are plain, uniformly
            coloured/sized location dots (`site_colour`/`site_marker_size`
            control their appearance). If True, markers are instead
            coloured and sized by their step-1 supply-to-demand `ratio`
            -- see `site_cmap`/`missing_site_colour`/`marker_size_range`,
            which only apply in this mode.
        site_colour : str, default "black"
            Marker colour for site markers when `show_site_ratio=False`.
        site_marker_size : float, default 60
            Static marker size for site markers when `show_site_ratio=False`.
            Ignored on interactive maps, where Folium markers are a fixed
            size.
        site_cmap : str, default "RdYlGn"
            Colormap for site markers (`ratio`) when `show_site_ratio=True`
            -- red for an overloaded, low-ratio site, green for a
            relatively uncontested one.
        missing_site_colour : str, default "lightgrey"
            When `show_site_ratio=True`: colour (and static marker size, at
            the smallest of `marker_size_range`) for a site with an
            undefined `ratio` -- no demand fell within its catchment, so
            there is nothing to colour or size it by.
        marker_size_range : tuple of (float, float), default (40, 220)
            When `show_site_ratio=True`: smallest and largest static marker
            size, linearly scaled by `ratio`. Ignored on interactive maps,
            where Folium markers are a fixed size.
        add_basemap : bool, default True
            If True, adds a background web map. Set False to skip the tile
            download entirely.
        title : str, optional
            Axes title. Ignored on interactive maps.
        caption : str, optional
            Explanatory text shown below the chart, wrapped to fit. If
            `None` (the default), a stakeholder-facing caption explaining
            how to read the region shading -- and, depending on
            `show_site_ratio`, either that the site markers are plain
            location dots or how to read their colour/size -- is
            generated; pass `""` to suppress it, or a custom string to
            replace it. Ignored on interactive maps.
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot onto. Ignored if `interactive=True`.
        figsize : tuple, optional
            Passed to `plt.subplots()` if `ax` is not supplied. Ignored if
            `interactive=True`.
        **kwargs : dict
            Additional keyword arguments passed to the region choropleth's
            plotting call (`GeoDataFrame.plot`/`.explore`).

        Returns
        -------
        matplotlib.axes.Axes or folium.Map

        Raises
        ------
        ValueError
            If neither `region_frame`/`site_frame` nor `supply_col` plus
            one of `catchment_size`/`distance_decay` are supplied, or if no
            region geometry layer has been registered via
            `add_region_geometry_layer()`.

        Notes
        -----
        Site markers are only drawn if `candidate_sites` was registered
        with real geometry (i.e. `add_sites()` was given a GeoDataFrame or
        lat/long columns, not a bare site list derived from the travel
        matrix's column names) -- otherwise only the region choropleth is
        shown.
        """
        ctx = self._prob_ctx

        if ctx.region_geometry_layer is None:
            raise ValueError(
                "No region geometry layer has been initialised. Please run "
                "`.add_region_geometry_layer()` first."
            )

        if region_frame is None or site_frame is None:
            if supply_col is None or (catchment_size is None and distance_decay is None):
                raise ValueError(
                    "Either pass precomputed `region_frame`/`site_frame` "
                    "(from `two_step_floating_catchment(return_site_ratios="
                    "True)`), or `supply_col` plus one of `catchment_size`/"
                    "`distance_decay` so they can be computed automatically."
                )
            region_frame, site_frame = self.two_step_floating_catchment(
                supply_col=supply_col,
                catchment_size=catchment_size,
                distance_decay=distance_decay,
                site_names=site_names,
                site_indices=site_indices,
                matrix=matrix,
                per_capita=per_capita,
                return_site_ratios=True,
            )

        region_gdf = ctx.region_geometry_layer.merge(
            region_frame.reset_index(),
            left_on=ctx._region_geometry_layer_common_col,
            right_on=ctx._demand_data_id_col,
            suffixes=("", "_y"),
        )
        region_gdf = region_gdf.drop(region_gdf.filter(regex="_y$").columns, axis=1)

        # NOTE: deliberately not `ctx._candidate_sites_type == "geopandas"`.
        # That attribute records add_sites()'s *input* format, not whether
        # self.candidate_sites ended up with real geometry -- tabular
        # lat/long input is converted to a GeoDataFrame internally but
        # still leaves _candidate_sites_type == "pandas". Checking the
        # actual type of candidate_sites is correct for both input paths;
        # it is only a plain DataFrame with no geometry when sites were
        # never registered via add_sites() at all (site names inferred
        # from the travel matrix's columns instead).
        has_site_geometry = isinstance(ctx.candidate_sites, geopandas.GeoDataFrame)
        site_gdf = None
        if has_site_geometry:
            site_gdf = ctx.candidate_sites.merge(
                site_frame.reset_index(),
                left_on=ctx._candidate_sites_candidate_id_col,
                right_on=ctx._candidate_sites_candidate_id_col,
                suffixes=("", "_y"),
            )
            site_gdf = site_gdf.drop(site_gdf.filter(regex="_y$").columns, axis=1)

            if show_site_ratio:
                min_size, max_size = marker_size_range
                site_gdf["_marker_size"] = (
                    min_size
                    + _min_max_normalize(site_gdf["ratio"], constant_fill=1.0)
                    * (max_size - min_size)
                ).fillna(min_size)

        if interactive:
            m = region_gdf.explore(
                column="accessibility",
                tooltip=[
                    ctx._region_geometry_layer_common_col,
                    "accessibility",
                    "n_sites_in_catchment",
                ],
                popup=True,
                cmap=cmap,
                style_kwds=dict(color="black"),
                tiles=tiles if add_basemap else None,
                **kwargs,
            )

            if has_site_geometry:
                tooltip = [
                    ctx._candidate_sites_candidate_id_col,
                    "supply",
                    "catchment_demand",
                    "ratio",
                ]
                if show_site_ratio:
                    site_gdf.explore(
                        m=m,
                        column="ratio",
                        cmap=site_cmap,
                        tooltip=tooltip,
                        popup=True,
                        marker_kwds=dict(radius=8),
                        missing_kwds=dict(color=missing_site_colour),
                    )
                else:
                    site_gdf.explore(
                        m=m,
                        color=site_colour,
                        tooltip=tooltip,
                        popup=True,
                        marker_kwds=dict(radius=8),
                    )

            return _attach_deferred_fit_bounds(m)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        region_gdf.plot(
            column="accessibility",
            legend=True,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            ax=ax,
            legend_kwds={"label": "Accessibility"},
            **kwargs,
        )

        if has_site_geometry:
            if show_site_ratio:
                site_gdf.plot(
                    column="ratio",
                    cmap=site_cmap,
                    markersize=site_gdf["_marker_size"],
                    edgecolor="black",
                    linewidth=0.5,
                    ax=ax,
                    legend=True,
                    legend_kwds={"label": "Site supply:demand ratio", "shrink": 0.6},
                    missing_kwds=dict(
                        color=missing_site_colour, label="No catchment demand"
                    ),
                )

                # geopandas' missing_kwds `label` only surfaces in a discrete
                # `scheme=` legend, not the continuous colorbar used here, so
                # the grey "no catchment demand" markers would otherwise have
                # no legend entry explaining them.
                if site_gdf["ratio"].isna().any():
                    ax.legend(
                        handles=[
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="none",
                                markerfacecolor=missing_site_colour,
                                markeredgecolor="black",
                                markersize=8,
                                label="No catchment demand",
                            )
                        ],
                        loc="lower left",
                    )
            else:
                site_gdf.plot(
                    color=site_colour,
                    markersize=site_marker_size,
                    edgecolor="black",
                    linewidth=0.5,
                    ax=ax,
                )

        if add_basemap:
            try:
                cx.add_basemap(ax, crs=region_gdf.crs.to_string(), timeout=30)
            except RequestException as e:
                warnings.warn(
                    f"Unable to download background map tiles ({type(e).__name__}). "
                    "Continuing without a basemap.",
                    stacklevel=2,
                )

        if title:
            ax.set_title(title)

        if not show_axis:
            ax.axis("off")

        default_caption = (
            "How to read this: each region is shaded by its accessibility "
            "score -- how much supply is available per head within reach, "
            "relative to the rest of this map (see the legend for the "
            "scale)."
        )
        if has_site_geometry:
            if show_site_ratio:
                default_caption += (
                    " Each site is coloured and sized by its own supply:demand "
                    "ratio: a small marker means a low ratio (an overloaded "
                    "site with little spare capacity), a large marker means a "
                    "high ratio (relatively more to go around); a grey marker "
                    "means no demand fell within that site's catchment, so no "
                    "ratio could be calculated."
                )
            else:
                default_caption += (
                    " Sites are shown as plain markers at their location "
                    "only -- their colour and size carry no meaning here. "
                    "Pass show_site_ratio=True to instead colour and size "
                    "them by their own supply:demand ratio."
                )
        _add_plot_caption(fig, caption, default_caption)

        return ax
