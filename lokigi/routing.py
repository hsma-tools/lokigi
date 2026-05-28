from lokigi.problem import _Problem
from lokigi.mixins.routing_attributes import RoutingAttributeMixin


class RoutingProblem(_Problem, RoutingAttributeMixin):
    def __init__(self, preferred_crs="EPSG:27700", debug_mode=True):

        self.num_resources = None

        self.resource_data = None
        self._resources_data_type = "geopandas"

        # Used only for simple resource approach, where all resources are assumed to have
        # a single start/end location and a single pattern
        self.resource_max_shift_duration_mins = None
        self.resource_shift_duration_flex_mins = None
        self.resource_lunch_duration_mins = None

        super().__init__(preferred_crs, debug_mode)
