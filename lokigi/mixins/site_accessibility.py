import warnings

import numpy as np
import pandas as pd


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
        if supply_col not in candidate_sites.columns:
            raise ValueError(
                f"supply_col={supply_col!r} not found in candidate_sites. "
                f"Available columns: {list(candidate_sites.columns)}."
            )

        indexed = candidate_sites.set_index(candidate_id_col)[supply_col]

        missing = [s for s in site_names if s not in indexed.index]
        if missing:
            raise KeyError(f"Sites not found in candidate_sites: {missing}.")

        supply = indexed.loc[site_names]

        if not pd.api.types.is_numeric_dtype(supply):
            raise TypeError(
                f"supply_col={supply_col!r} must contain numeric values. "
                "Hint: try pd.to_numeric() on this column before calling "
                "add_sites()."
            )

        null_sites = supply[supply.isna()].index.tolist()
        if null_sites:
            raise ValueError(
                f"The following sites are missing a value in "
                f"supply_col={supply_col!r}: {null_sites}."
            )

        negative_sites = supply[supply < 0].index.tolist()
        if negative_sites:
            raise ValueError(
                f"The following sites have a negative value in "
                f"supply_col={supply_col!r}, which is not a valid supply "
                f"quantity: {negative_sites}."
            )

        return supply

    @staticmethod
    def _two_step_floating_catchment(
        cost_frame, demand, supply, catchment_size, per_capita, return_site_ratios
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
        catchment_size : float
            Hard catchment threshold d0. Membership is inclusive (`<=`),
            unlike the library's coverage metrics, which use strict `<` --
            deliberate, matching the inclusive convention in the 2SFCA
            literature.
        per_capita : float
            Multiplier applied to `accessibility`, e.g. 1_000 to express
            supply per 1,000 head instead of raw supply units per head.
        return_site_ratios : bool
            If True, also return the step-1 per-site table.

        Returns
        -------
        pandas.DataFrame or tuple of (pandas.DataFrame, pandas.DataFrame)
        """
        in_catchment = cost_frame <= catchment_size

        # Step 1: supply-to-demand ratio per site.
        catchment_demand = in_catchment.mul(demand, axis=0).sum(axis=0)
        n_regions_in_catchment = in_catchment.sum(axis=0)

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
                f"demand within catchment_size={catchment_size}, so their "
                "supply-to-demand ratio is undefined (NaN) and they are "
                f"excluded from every region's accessibility score: "
                f"{empty_sites}.",
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

        # Step 2: sum reachable sites' ratios per demand region. A NaN
        # ratio (empty-catchment site) must drop out of this sum rather
        # than propagate -- an unreachable site's undefined ratio
        # shouldn't poison every other region that happens to be near it
        # too.
        usable_ratio = ratio.fillna(0.0)
        accessibility = in_catchment.mul(usable_ratio, axis=1).sum(axis=1) * per_capita
        n_sites_in_catchment = in_catchment.sum(axis=1)

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
        catchment_size,
        site_names=None,
        site_indices=None,
        matrix=None,
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
        catchment_size : float
            Hard catchment threshold d0, in the travel matrix's registered
            units. A site is "in catchment" for a demand region if the
            travel cost between them is `<= catchment_size`.
        site_names : list of str, optional
        site_indices : list of int, optional
            The site set to score. At most one may be given. If neither is
            given, every candidate site is scored.
        matrix : str, optional
            Label of a secondary travel matrix registered via
            `add_secondary_travel_matrix()`. Scores accessibility under
            that matrix's travel costs instead of the primary matrix's.
        per_capita : float, default 1
            Multiplier applied to the `accessibility` column, e.g. 1_000
            to express supply per 1,000 head instead of raw supply units
            per head. Does not affect `site_frame`.
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
            If both `site_names` and `site_indices` are given, if no demand
            data is registered, if `supply_col` is missing or contains a
            null/negative value for a scored site, or if `matrix` is not a
            registered secondary travel matrix label.
        TypeError
            If `supply_col` is not numeric.
        IndexError
            If `site_indices` are out of range.
        KeyError
            If `site_names` are not found in the travel matrix or
            `candidate_sites`.

        Notes
        -----
        A site with no demand within `catchment_size` has an undefined
        (`NaN`) supply-to-demand ratio and is excluded from every region's
        accessibility score, with a warning naming it. A demand region with
        no site in `catchment_size` correctly scores `accessibility == 0`
        (a real "no supply available" result, not a missing value) --
        these two zero-like cases are deliberately kept distinguishable.

        `accessibility` obeys `sum(demand * accessibility) == sum(supply)`
        (before `per_capita` scaling) whenever every scored site has at
        least one region in its catchment, since the sum of demand-weighted
        ratios is exactly the supply that produced them.
        """
        if site_names is not None and site_indices is not None:
            raise ValueError(
                "Please provide at most one of 'site_names' or "
                "'site_indices', not both."
            )

        if self._demand_data_demand_col is None:
            raise ValueError(
                "two_step_floating_catchment requires demand data. Call "
                "add_demand() before using this method."
            )

        if self.travel_and_demand_df is None:
            self._create_joined_demand_travel_df(index_col=self._demand_data_id_col)
            self._build_secondary_travel_frames()

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

        demand = self.travel_and_demand_df[self._demand_data_demand_col].reindex(
            cost_frame.index
        )

        supply = self._resolve_supply(
            self.candidate_sites,
            self._candidate_sites_candidate_id_col,
            supply_col,
            site_names_resolved,
        )

        return self._two_step_floating_catchment(
            cost_frame=cost_frame,
            demand=demand,
            supply=supply,
            catchment_size=catchment_size,
            per_capita=per_capita,
            return_site_ratios=return_site_ratios,
        )
