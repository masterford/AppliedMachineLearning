# This is the implementation of the first network architecture. We have
# started it, but you need to finish it. Do not change the class name
# or the name of the data members "fc1" or "fc2"
import torch
class FeedForward(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(3072, 1000)
    # You need to add the second layer's parameters
    self.fc2 = torch.nn.Linear(1000,10)

  def forward(self, X):
    batch_size = X.size(0)
    # This next line reshapes the tensor to be size (B x 3072)
    # so it can be passed through a linear layer.
    X = X.view(batch_size, -1)
    # You need to pass X through the two linear layers and relu
    # then return the final scores
    h1 = self.fc1(X)
    h1 = torch.nn.functional.relu(h1)
    h2 = self.fc2(h1)

    return h2
    
    # This is the implementation of the second network architecture. We have
# started it, but you need to finish it. Do not change the class name
# or the name of the data members "conv1", "pool", "conv2", "fc1", "fc2",
# or "fc3".
class Convolutional(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3,
                                 out_channels=7,
                                 kernel_size=3,
                                 stride=1,
                                 padding=0)
    # TODO
    # You need to add the pooling, second convolution, and
    # three linear modules here
    self.pool = None
    self.conv2 = None
    self.fc1 = None
    self.fc2 = None
    self.fc3 = None

  def forward(self, X):
    batch_size = X.size(0)
    # TODO
    # You need to implement the full network architecture here
    # and return the final scores
    raise NotImplementedError()