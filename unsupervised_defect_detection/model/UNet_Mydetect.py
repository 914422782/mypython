import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(dim, num_heads)

    def forward(self, x):
        B, C, H, W = x.shape

        # Ensure H and W are divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))

        H_pad, W_pad = x.shape[2], x.shape[3]

        x = x.view(B, C, H_pad // self.window_size, self.window_size, W_pad // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, self.window_size * self.window_size, C)
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.view(B, H_pad // self.window_size, W_pad // self.window_size, self.window_size,
                                 self.window_size, C)
        attn_out = attn_out.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, H_pad, W_pad)

        # Remove padding
        attn_out = attn_out[:, :, :H, :W]
        return attn_out


class TransformerWithWindow(nn.Module):
    def __init__(self, d_model, window_size, nhead, num_encoder_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            WindowAttention(d_model, window_size, nhead)
            for _ in range(num_encoder_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UNet_Mydetect(nn.Module):
    def __init__(self, input=1, output=1):
        super(UNet_Mydetect, self).__init__()
        batchNorm_momentum = 0.1

        self.in_conv1 = nn.Conv2d(input, 64, kernel_size=3, stride=1, padding=1)
        self.in_conv1_bn = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.in_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.in_conv2_bn = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv1_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3_2_bn = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_1_bn = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2_bn = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        # Transformer
        self.transformer = TransformerWithWindow(d_model=512, window_size=4, nhead=8, num_encoder_layers=6)

        self.conv5_1 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv6_1 = nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_1_bn = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2_bn = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv7_1 = nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_1_bn = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2_bn = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_1_bn = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8_2_bn = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.out_conv = nn.Conv2d(64, output, kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        c1 = F.relu(self.in_conv1_bn(self.in_conv1(input)))
        c2 = F.relu(self.in_conv2_bn(self.in_conv2(c1)))

        e1_ = F.max_pool2d(c2, kernel_size=2, stride=2)
        e1_ = F.relu(self.conv1_1_bn(self.conv1_1(e1_)))
        e1 = F.relu(self.conv1_2_bn(self.conv1_2(e1_)))

        e2_ = F.max_pool2d(e1, kernel_size=2, stride=2)
        e2_ = F.relu(self.conv2_1_bn(self.conv2_1(e2_)))
        e2 = F.relu(self.conv2_2_bn(self.conv2_2(e2_)))

        e3_ = F.max_pool2d(e2, kernel_size=2, stride=2)
        e3_ = F.relu(self.conv3_1_bn(self.conv3_1(e3_)))
        e3 = F.relu(self.conv3_2_bn(self.conv3_2(e3_)))

        e4_ = F.max_pool2d(e3, kernel_size=2, stride=2)
        e4_ = F.relu(self.conv4_1_bn(self.conv4_1(e4_)))
        e4 = F.relu(self.conv4_2_bn(self.conv4_2(e4_)))

        # Transformer module
        # b, c, h, w = e4.shape
        # e4_flat = e4.view(b, c, -1).permute(2, 0, 1)  # [H*W, B, C]
        # transformer_out = self.transformer(e4_flat, e4_flat)  # [H*W, B, C]
        # e4_transformed = transformer_out.permute(1, 2, 0).view(b, c, h, w)
        e4_transformed = self.transformer(e4)

        d1_1 = F.interpolate(e4_transformed, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = e3.size()[2] - d1_1.size()[2]
        diffX = e3.size()[3] - d1_1.size()[3]
        d1_1 = F.pad(d1_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d1_2 = torch.cat([d1_1, e3], dim=1)
        d1_ = F.relu(self.conv5_1_bn(self.conv5_1(d1_2)))
        d1 = F.relu(self.conv5_2_bn(self.conv5_2(d1_)))

        d2_1 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = e2.size()[2] - d2_1.size()[2]
        diffX = e2.size()[3] - d2_1.size()[3]
        d2_1 = F.pad(d2_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d2_2 = torch.cat([d2_1, e2], dim=1)
        d2_ = F.relu(self.conv6_1_bn(self.conv6_1(d2_2)))
        d2 = F.relu(self.conv6_2_bn(self.conv6_2(d2_)))

        d3_1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = e1.size()[2] - d3_1.size()[2]
        diffX = e1.size()[3] - d3_1.size()[3]
        d3_1 = F.pad(d3_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d3_2 = torch.cat([d3_1, e1], dim=1)
        d3_ = F.relu(self.conv7_1_bn(self.conv7_1(d3_2)))
        d3 = F.relu(self.conv7_2_bn(self.conv7_2(d3_)))

        d4_1 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        diffY = c2.size()[2] - d4_1.size()[2]
        diffX = c2.size()[3] - d4_1.size()[3]
        d4_1 = F.pad(d4_1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        d4_2 = torch.cat([d4_1, c2], dim=1)
        d4_ = F.relu(self.conv8_1_bn(self.conv8_1(d4_2)))
        d4 = F.relu(self.conv8_2_bn(self.conv8_2(d4_)))
        output = self.out_conv(d4)

        return output
