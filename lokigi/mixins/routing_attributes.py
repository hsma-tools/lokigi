import pandas as pd
from typing import Literal, Optional, Union, Callable


class RoutingAttributeMixin:
    """A mixin class providing geographic routing and service configuration
    for community healthcare modeling.
    """

    def add_resources_same_starts(
        self, count: int, starting_location: str, return_to_start: bool = True
    ) -> None:
        """Define a fleet of identical resources starting from the same central hub.

        Parameters
        ----------
        count : int
            The number of individual resources (e.g., nurses, cars) to deploy.
        starting_location : str
            The location identifier (e.g., postcode sector or matrix ID) where
            all resources begin their shift.
        return_to_start : bool, default True
            If True, the heuristic closes the loop back to the starting hub.
            If False, the shift ends at the final patient location (open loop).
        """
        pass

    def add_resources_varying_starts(
        self,
        resource_df: pd.DataFrame,
        id_col: str,
        location_col: str,
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
