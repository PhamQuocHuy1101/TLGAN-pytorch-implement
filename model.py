import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channel = 64, kernel = 3, stride = 1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(channel, channel, kernel_size=kernel, stride=stride, padding=padding, bias=True)
        self.ac_1 = nn.ReLU()
        self.batch_norm_1 = nn.BatchNorm2d(channel)
        self.conv_2 = nn.Conv2d(channel, channel, kernel_size=kernel, stride=stride, padding=padding, bias=True)
        self.batch_norm_2 = nn.BatchNorm2d(channel)
    def forward(self, X):
        output = self.conv_1(X)
        output = self.ac_1(output)
        output = self.batch_norm_1(output)
        output = self.conv_2(output)
        output = self.batch_norm_2(output)
        return X + output
    
class Generator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, n_blocks = 16):
        super(Generator, self).__init__()
        
        hidden_channels = 64
        self.block_input = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=9, stride=1, padding=4, bias=True ),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList([ResidualBlock(channel = hidden_channels) for _ in range(n_blocks)])
        self.residual_final = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=True ),
            nn.BatchNorm2d(hidden_channels)
        )
        self.conv_final = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels*4, kernel_size=3, stride=1, padding=1, bias=True ),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*4, out_channels, kernel_size=9, stride=1, padding=4, bias=True ),
            nn.Sigmoid()
        )
    def forward(self, X):
        output_1 = self.block_input(X)
        out_blocks = output_1
        for layer in self.blocks:
            out_blocks = layer(out_blocks)
        output = self.residual_final(out_blocks) + output_1
        return self.conv_final(output)

class Discriminator(nn.Module):
    def __init__(self, in_channels, input_size = 128):
        super(Discriminator, self).__init__()
        self.hidden_channels = 64
        self.input_size = input_size
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels, kernel_size=3, stride = 2, padding = 1),
            nn.LeakyReLU(0.2)
        ) #(64, 3) > (64, 64)

        #(size, dim)
        self.conv_blocks = nn.ModuleList([
            self.make_inner_blok(self.hidden_channels, self.hidden_channels, kernel = 3, stride = 2, padding = 1), # (64, 64) -> (32, 64)
            self.make_inner_blok(self.hidden_channels, self.hidden_channels*2, kernel = 3, stride = 1, padding = 1), #(32, 64) -> (32, 128)
            self.make_inner_blok(self.hidden_channels*2, self.hidden_channels*2, kernel = 3, stride = 2, padding = 1), #(32, 128) -> (16, 128)
            self.make_inner_blok(self.hidden_channels*2, self.hidden_channels*4, kernel = 3, stride = 1, padding = 1), #(16, 128) -> (16, 256)
            self.make_inner_blok(self.hidden_channels*4, self.hidden_channels*4, kernel = 3, stride = 2, padding = 1), #(16, 256) -> (8, 256)
            self.make_inner_blok(self.hidden_channels*4, self.hidden_channels*8, kernel = 3, stride = 1, padding = 1), #(8, 256) -> (8, 512)
            self.make_inner_blok(self.hidden_channels*8, self.hidden_channels*8, kernel = 3, stride = 2, padding = 1) #(8, 512) -> (4, 512)
        ])

        self.dense = nn.Sequential(
            nn.Linear(self.hidden_channels*8, self.hidden_channels*16),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_channels*16, 1),
            nn.Sigmoid()
        )
    
    def make_inner_blok(self, in_channels, out_channels, kernel=3, stride = 1, padding = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride = stride, padding = padding),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, X):
        out = self.input_block(X)
        for block in self.conv_blocks:
            out = block(out)
        out = out.permute(0, 2, 3, 1)
        out = self.dense(out)
        return torch.flatten(out, start_dim = 1)
