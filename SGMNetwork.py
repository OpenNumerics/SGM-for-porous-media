import torch as pt
import torch.nn as nn

from collections import OrderedDict

class Score(nn.Module):
    def __init__(self, layers=[]):
        super().__init__()
        
        # Create all feed-forward layers
        self.depth = len(layers) - 1
        self.activation = nn.ReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, pt.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, self.activation()))
        layer_list.append(('layer_%d' % (self.depth-1), pt.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        # Combine all layers in a single Sequential object to keep track of parameter count
        self.layers = pt.nn.Sequential(layerDict)

    def forward(self, x):
        return self.layers(x)