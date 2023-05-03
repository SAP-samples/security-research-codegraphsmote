from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from .pooling import Pooling


class AbstractParameterlessPooling(Pooling):
    def __init__(self, params):
        super(AbstractParameterlessPooling, self).__init__()
        self.params = params

    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def loss_keys(self):
        return []
    
    def loss(self, representation, batch):
        return (self.pool(representation, batch), {})
    
    def get_params(self):
        return self.params


class MeanPooling(AbstractParameterlessPooling):
    def pool(self, representation, batch):
        return global_mean_pool(representation, batch)


class SumPooling(AbstractParameterlessPooling):
    def pool(self, representation, batch):
        return global_add_pool(representation, batch)


class MaxPooling(AbstractParameterlessPooling):
    def pool(self, representation, batch):
        return global_max_pool(representation, batch)