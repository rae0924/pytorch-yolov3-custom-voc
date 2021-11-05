import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class ResidualUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
                ConvBlock(in_channels, in_channels//2, kernel_size=1),
                ConvBlock(in_channels//2, in_channels, kernel_size=3, padding=1)
            )

    def forward(self, x):
        x = x + self.layers(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n = n
        self.layers.append(
            ConvBlock(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1)
        )
        for _ in range(n):
            self.layers.append(
                ResidualUnit(in_channels*2)
            )
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Neck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        if in_channels == 1024:
            self.layers = nn.Sequential(
                ConvBlock(in_channels, in_channels//2, kernel_size=1),
                ConvBlock(in_channels//2, in_channels, kernel_size=3, padding=1),
                ConvBlock(in_channels, in_channels//2, kernel_size=1),
                ConvBlock(in_channels//2, in_channels, kernel_size=3, padding=1),
                ConvBlock(in_channels, in_channels//2, kernel_size=1)
            )
        else:
            self.layers = nn.Sequential(
                ConvBlock(in_channels, in_channels//3, kernel_size=1),
                ConvBlock(in_channels//3, in_channels//3*2, kernel_size=3, padding=1),
                ConvBlock(in_channels//3*2, in_channels//3, kernel_size=1),
                ConvBlock(in_channels//3, in_channels//3*2, kernel_size=3, padding=1),
                ConvBlock(in_channels//3*2, in_channels//3, kernel_size=1)
            )

    def forward(self, x):
        x = self.layers(x)
        return x

class ConvUpsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(in_channels, in_channels//2, kernel_size=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ScaledPrediction(nn.Module):
    def __init__(self, in_channels, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.pred = nn.Sequential(
            ConvBlock(in_channels, in_channels*2, kernel_size=3, padding=1),
            ConvBlock(in_channels*2, 3*(5+self.num_classes), kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.pred(x)
        #x = x.reshape(x.shape[0], 3, x.shape[2], x.shape[3], self.num_classes+5)
        return x


class Darknet53(nn.Module):
    def __init__(self):
        super().__init__()
        # input = (batch_size, 3, 416, 416)
        self.layers = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, kernel_size=3, padding=1), # output: (batch_size, 32, 416, 416)
            ResidualBlock(in_channels=32, n=1),     # output: (batch_size, 64, 208, 208)
            ResidualBlock(in_channels=64, n=2),     # output: (batch_size, 128, 104, 104)
            ResidualBlock(in_channels=128, n=8),    # output: (batch_size, 256, 52, 52) -> save for concat with later upsample
            ResidualBlock(in_channels=256, n=8),    # output: (batch_size, 512, 26, 26) -> save for concat with later upsample
            ResidualBlock(in_channels=512, n=4),    # output: (batch_size, 1024, 13, 13)
        )

    def forward(self, x):
        x = self.layers(x)
        return x



class YOLOv3(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = Darknet53()
        self.necks = nn.ModuleList(modules=[
            Neck(in_channels=1024),
            Neck(in_channels=768),
            Neck(in_channels=384) 
        ])
        self.scaled_predictions = nn.ModuleList(modules=[
            ScaledPrediction(in_channels=512),
            ScaledPrediction(in_channels=256),
            ScaledPrediction(in_channels=128)
        ])
        self.upsamples = nn.ModuleList(modules=[
            ConvUpsample(in_channels=512),
            ConvUpsample(in_channels=256)
        ])
        self.scale_concat = list()

    def forward(self, x):
        predictions = []
        # an example input might be: (batch_size, 3, 416, 416)
        # pass through the feature extractor
        for layer in self.backbone.layers:
            x = layer(x)
            if isinstance(layer, ResidualBlock):
                if layer.n == 8:
                    self.scale_concat.append(x)
        # output = (batch_size, 1024, 13, 13)

        # scale 1 predictions
        x = self.necks[0](x)        # output: (batch_size, 512, 13, 13)
        scale_one_pred = self.scaled_predictions[0](x)
        predictions.append(scale_one_pred)

        # scale 2 predictions
        x = self.upsamples[0](x)     # output: (batch_size, 256 (512/2), 26 (13*2), 26 (13*2))
        x = torch.cat([x, self.scale_concat.pop()], dim=1)  # output: (batch_size, 768 (256 + 512), 26, 26)
        x = self.necks[1](x)        # output: (batch_size, 256 (768/3), 26, 26)
        scale_two_pred = self.scaled_predictions[1](x)
        predictions.append(scale_two_pred)

        # scale 3 predictions
        x = self.upsamples[1](x)    # output: (batch_size, 128 (256/2), 52 (26*2), 52 (26*2))
        x = torch.cat([x, self.scale_concat.pop()], dim=1)  # output: (batch_size, 384 (128 + 256), 52, 52)
        x = self.necks[2](x)        # output: (batch_size, 128 (384/3), 52, 52)
        scale_three_pred = self.scaled_predictions[2](x)
        predictions.append(scale_three_pred)

        return predictions

net = YOLOv3()
x = torch.rand([1,3,416,416])
y = net(x)
print([i.shape for i in y])


