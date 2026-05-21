from lokigi.problem import _Problem
from lokigi.mixins.routing_attributes import RoutingAttributeMixin


class RoutingProblem(_Problem, RoutingAttributeMixin):
    def __init__(self, preferred_crs="EPSG:27700", debug_mode=True):

        self.num_resources = None

        self.resource_same_starts = None  # Bool, will be true/false

        self.resource_data = None

        self.resource_constraints = None

        self.resource_

        super().__init__(preferred_crs, debug_mode)
