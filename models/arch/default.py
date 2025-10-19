# Define network components here
import torch
from torch import nn
import torch.nn.functional as F
import math


class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) implementation.
    Creates a top-down pathway with lateral connections for multi-scale feature fusion.
    """
    def __init__(self, in_channels, out_channels, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        
        # Lateral connections (1x1 convs to reduce channel dimensions)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            for _ in range(num_levels)
        ])
        
        # Top-down pathway (3x3 convs for feature refinement)
        self.top_down_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            for _ in range(num_levels)
        ])
        
        self.relu = nn.ReLU(inplace=True)
        
        # Final fusion layer
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, feats):
        """
        Args:
            feats: Input feature map of shape (B, C, H, W)
        Returns:
            Enhanced feature map with multi-scale information
        """
        # Create pyramid levels by downsampling
        pyramid_levels = []
        current_feat = feats
        
        for i in range(self.num_levels):
            pyramid_levels.append(current_feat)
            if i < self.num_levels - 1:  # Don't downsample the last level
                current_feat = F.avg_pool2d(current_feat, kernel_size=2, stride=2)
        
        # Apply lateral connections
        lateral_feats = []
        for i, feat in enumerate(pyramid_levels):
            lateral_feat = self.lateral_convs[i](feat)
            lateral_feat = self.lateral_norms[i](lateral_feat)
            lateral_feats.append(lateral_feat)
        
        # Top-down pathway with lateral connections
        top_down_feats = []
        prev_feat = None
        
        for i in reversed(range(self.num_levels)):
            lateral_feat = lateral_feats[i]
            
            if prev_feat is not None:
                # Upsample previous feature to match current level
                h, w = lateral_feat.size(2), lateral_feat.size(3)
                prev_feat = F.interpolate(prev_feat, size=(h, w), mode='bilinear', align_corners=False)
                # Add lateral connection
                top_down_feat = lateral_feat + prev_feat
            else:
                top_down_feat = lateral_feat
            
            # Apply 3x3 conv for refinement
            top_down_feat = self.top_down_convs[i](top_down_feat)
            top_down_feat = self.top_down_norms[i](top_down_feat)
            top_down_feat = self.relu(top_down_feat)
            
            top_down_feats.insert(0, top_down_feat)  # Insert at beginning to maintain order
            prev_feat = top_down_feat
        
        # Use the finest level (original resolution) as output
        output_feat = top_down_feats[0]
        
        # Final fusion
        output_feat = self.fusion_conv(output_feat)
        output_feat = self.fusion_norm(output_feat)
        output_feat = self.relu(output_feat)
        
        return output_feat


class PyramidPooling(nn.Module):
    """
    Legacy Pyramid Pooling module - kept for backward compatibility.
    Consider using FeaturePyramidNetwork instead for better performance.
    """
    def __init__(self, in_channels, out_channels, scales=(4, 8, 16, 32), ct_channels=1):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels, scale, ct_channels) for scale in scales])
        self.bottleneck = nn.Conv2d(in_channels + len(scales) * ct_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def _make_stage(self, in_channels, scale, ct_channels):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AvgPool2d(kernel_size=(scale, scale))
        conv = nn.Conv2d(in_channels, ct_channels, kernel_size=1, bias=False)
        relu = nn.LeakyReLU(0.2, inplace=True)
        return nn.Sequential(prior, conv, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = torch.cat([F.interpolate(input=stage(feats), size=(h, w), mode='nearest') for stage in self.stages] + [feats], dim=1)
        return self.relu(self.bottleneck(priors))


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA-Net) implementation.
    More efficient than SE-Net by using 1D convolution instead of FC layers
    and adaptively determining the kernel size.
    """
    def __init__(self, channel, k_size=None):
        super(ECALayer, self).__init__()
        if k_size is None:
            # Adaptive kernel size calculation based on channel dimension
            t = int(abs((math.log(channel, 2) + 1) / 2))
            k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average pooling
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # (b, c, 1, 1)
        
        # Remove spatial dimensions and transpose for 1D conv
        y = y.view(b, 1, c)  # (b, 1, c)
        
        # 1D convolution for channel attention
        y = self.conv(y)  # (b, 1, c)
        
        # Apply sigmoid and reshape back
        y = self.sigmoid(y)  # (b, 1, c)
        y = y.view(b, c, 1, 1)  # (b, c, 1, 1)
        
        return x * y


class SELayer(nn.Module):
    """
    Legacy Squeeze-and-Excitation layer - kept for backward compatibility.
    Consider using ECALayer instead for better efficiency.
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y        
     

class DRNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_feats, n_resblocks, norm=nn.BatchNorm2d, 
    se_reduction=None, res_scale=1, bottom_kernel_size=3, pyramid=False, use_fpn=True, use_eca=True):
        super(DRNet, self).__init__()
        # Initial convolution layers
        conv = nn.Conv2d
        deconv = nn.ConvTranspose2d
        act = nn.ReLU(True)
        
        self.pyramid_module = None
        self.conv1 = ConvLayer(conv, in_channels, n_feats, kernel_size=bottom_kernel_size, stride=1, norm=None, act=act)
        self.conv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
        self.conv3 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=2, norm=norm, act=act)

        # Residual layers
        dilation_config = [1] * n_resblocks

        self.res_module = nn.Sequential(*[ResidualBlock(
            n_feats, dilation=dilation_config[i], norm=norm, act=act, 
            se_reduction=se_reduction, res_scale=res_scale, use_eca=use_eca) for i in range(n_resblocks)])

        # Upsampling Layers
        self.deconv1 = ConvLayer(deconv, n_feats, n_feats, kernel_size=4, stride=2, padding=1, norm=norm, act=act)

        if not pyramid:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        else:
            self.deconv2 = ConvLayer(conv, n_feats, n_feats, kernel_size=3, stride=1, norm=norm, act=act)
            # Use FPN by default, fallback to PyramidPooling if use_fpn=False
            if use_fpn:
                self.pyramid_module = FeaturePyramidNetwork(n_feats, n_feats, num_levels=4)
            else:
                self.pyramid_module = PyramidPooling(n_feats, n_feats, scales=(4,8,16,32), ct_channels=n_feats//4)
            self.deconv3 = ConvLayer(conv, n_feats, out_channels, kernel_size=1, stride=1, norm=None, act=act)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.res_module(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        if self.pyramid_module is not None:
            x = self.pyramid_module(x)
        x = self.deconv3(x)

        return x


class ConvLayer(torch.nn.Sequential):
    def __init__(self, conv, in_channels, out_channels, kernel_size, stride, padding=None, dilation=1, norm=None, act=None):
        super(ConvLayer, self).__init__()
        # padding = padding or kernel_size // 2
        padding = padding or dilation * (kernel_size - 1) // 2
        self.add_module('conv2d', conv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation))
        if norm is not None:
            self.add_module('norm', norm(out_channels))
            # self.add_module('norm', norm(out_channels, track_running_stats=True))
        if act is not None:
            self.add_module('act', act)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels, dilation=1, norm=nn.BatchNorm2d, act=nn.ReLU(True), se_reduction=None, res_scale=1, use_eca=True):
        super(ResidualBlock, self).__init__()
        conv = nn.Conv2d
        self.conv1 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=act)
        self.conv2 = ConvLayer(conv, channels, channels, kernel_size=3, stride=1, dilation=dilation, norm=norm, act=None)
        self.attention_layer = None
        self.res_scale = res_scale
        
        if se_reduction is not None:
            if use_eca:
                # Use ECA-Net (more efficient, no reduction parameter needed)
                self.attention_layer = ECALayer(channels)
            else:
                # Use SE-Net (legacy)
                self.attention_layer = SELayer(channels, se_reduction)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.attention_layer:
            out = self.attention_layer(out)
        out = out * self.res_scale
        out = out + residual
        return out

    def extra_repr(self):
        return 'res_scale={}'.format(self.res_scale)
