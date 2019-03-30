import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        self.num_priors = len(cfg['scales'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.steps = cfg['steps']
        self.scales = cfg['scales']
        self.clip = cfg['clip']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')


    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i in range(f[0]):
                for j in range(f[1]):
                    f_k_y = self.image_size[0] / self.steps[k][0]
                    f_k_x = self.image_size[1] / self.steps[k][1]
                    # unit center x,y
                    cx = (j + 0.5) / f_k_x
                    cy = (i + 0.5) / f_k_y

                    for q in self.scales[k]:
                        mean += [cx, cy, q*1.0/self.image_size[1], q*1.0/self.image_size[0]]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output