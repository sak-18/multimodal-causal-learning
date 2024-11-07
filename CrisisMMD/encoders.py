# encoders.py
import torch
from torch import nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights

class ImageEncoder(nn.Module):
    def __init__(self, output_size):
        super(ImageEncoder, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)

    def forward(self, x):
        return self.resnet(x)

class TextEncoder2D(nn.Module):
    """2D-ConvNet to encode text data from one-hot encoded inputs."""

    def __init__(self, output_size, sequence_length, vocab_size, fbase=25):
        super(TextEncoder2D, self).__init__()
        if sequence_length < 24 or sequence_length > 32:
            raise ValueError("TextEncoder2D expects sequence_length between 24 and 32")
        self.fbase = fbase

        # Convolutional layers
        self.convnet = nn.Sequential(
            nn.Conv2d(1, fbase, (4, vocab_size), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(fbase),
            nn.ReLU(True),
            nn.Conv2d(fbase, fbase * 2, (4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(fbase * 2),
            nn.ReLU(True),
            nn.Conv2d(fbase * 2, fbase * 4, (4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(fbase * 4),
            nn.ReLU(True)
        )

        # Manually calculate the output size after the convolutions
        conv_out_height = self.calculate_conv_output_size(sequence_length, 4, 2, 1, num_layers=3)
        self.linear_input_size = fbase * 4 * conv_out_height * 1  # Height * Width is conv_out_height * 1

        self.linear = nn.Linear(self.linear_input_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv2D
        x = self.convnet(x)
        x = x.view(-1, self.linear_input_size)  # Flatten the tensor
        x = self.linear(x)
        return x

    def calculate_conv_output_size(self, input_size, kernel_size, stride, padding, num_layers):
        """Helper function to calculate the output size after several Conv2D layers."""
        output_size = input_size
        for _ in range(num_layers):
            output_size = (output_size - kernel_size + 2 * padding) // stride + 1
        return output_size


class TaskHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TaskHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)
