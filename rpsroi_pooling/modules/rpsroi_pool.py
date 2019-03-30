from torch.nn.modules.module import Module
import sys
from ..functions.rpsroi_pooling import RPSRoIPoolingFunction


class RPSRoIPool(Module):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        super(RPSRoIPool, self).__init__()

        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

    def forward(self, features, rois):
        input_dim = features.size(1)
        assert(input_dim == self.pooled_width*self.pooled_height*self.output_dim)
        return RPSRoIPoolingFunction(self.pooled_height, self.pooled_width, self.spatial_scale, self.group_size, self.output_dim)(features, rois)
