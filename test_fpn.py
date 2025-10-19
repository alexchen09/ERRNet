#!/usr/bin/env python3
"""
Test script to verify the Feature Pyramid Network implementation.
"""

import torch
import torch.nn as nn
from models.arch.default import FeaturePyramidNetwork, PyramidPooling, DRNet
from models.arch import errnet, basenet

def test_fpn_module():
    """Test the FeaturePyramidNetwork module independently."""
    print("Testing FeaturePyramidNetwork module...")
    
    # Test parameters
    batch_size = 2
    in_channels = 256
    out_channels = 256
    height, width = 64, 64
    
    # Create test input
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Create FPN module
    fpn = FeaturePyramidNetwork(in_channels, out_channels, num_levels=4)
    
    # Forward pass
    with torch.no_grad():
        output = fpn(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {out_channels}, {height}, {width})")
    
    # Verify output shape
    assert output.shape == (batch_size, out_channels, height, width), f"Shape mismatch: {output.shape}"
    print("✓ FPN module test passed!")
    
    return fpn

def test_drnet_with_fpn():
    """Test DRNet with FPN integration."""
    print("\nTesting DRNet with FPN...")
    
    # Test parameters
    batch_size = 1
    in_channels = 3
    height, width = 224, 224
    
    # Create test input
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Create DRNet with FPN
    drnet_fpn = DRNet(in_channels, 3, 256, 13, norm=None, res_scale=0.1, 
                      se_reduction=8, bottom_kernel_size=1, pyramid=True, use_fpn=True)
    
    # Forward pass
    with torch.no_grad():
        output = drnet_fpn(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 3, {height}, {width})")
    
    # Verify output shape
    assert output.shape == (batch_size, 3, height, width), f"Shape mismatch: {output.shape}"
    print("✓ DRNet with FPN test passed!")
    
    return drnet_fpn

def test_drnet_with_pyramid_pooling():
    """Test DRNet with legacy PyramidPooling for comparison."""
    print("\nTesting DRNet with PyramidPooling...")
    
    # Test parameters
    batch_size = 1
    in_channels = 3
    height, width = 224, 224
    
    # Create test input
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Create DRNet with PyramidPooling
    drnet_pp = DRNet(in_channels, 3, 256, 13, norm=None, res_scale=0.1, 
                     se_reduction=8, bottom_kernel_size=1, pyramid=True, use_fpn=False)
    
    # Forward pass
    with torch.no_grad():
        output = drnet_pp(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 3, {height}, {width})")
    
    # Verify output shape
    assert output.shape == (batch_size, 3, height, width), f"Shape mismatch: {output.shape}"
    print("✓ DRNet with PyramidPooling test passed!")
    
    return drnet_pp

def test_errnet_function():
    """Test the errnet function which should now use FPN by default."""
    print("\nTesting errnet function...")
    
    # Test parameters
    batch_size = 1
    in_channels = 3
    height, width = 224, 224
    
    # Create test input
    x = torch.randn(batch_size, in_channels, height, width)
    print(f"Input shape: {x.shape}")
    
    # Create network using errnet function
    net = errnet(in_channels, 3)
    
    # Forward pass
    with torch.no_grad():
        output = net(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 3, {height}, {width})")
    
    # Verify output shape
    assert output.shape == (batch_size, 3, height, width), f"Shape mismatch: {output.shape}"
    print("✓ errnet function test passed!")
    
    return net

def compare_parameter_counts():
    """Compare parameter counts between FPN and PyramidPooling."""
    print("\nComparing parameter counts...")
    
    # Create both networks
    drnet_fpn = DRNet(3, 3, 256, 13, norm=None, res_scale=0.1, 
                      se_reduction=8, bottom_kernel_size=1, pyramid=True, use_fpn=True)
    drnet_pp = DRNet(3, 3, 256, 13, norm=None, res_scale=0.1, 
                     se_reduction=8, bottom_kernel_size=1, pyramid=True, use_fpn=False)
    
    # Count parameters
    fpn_params = sum(p.numel() for p in drnet_fpn.parameters())
    pp_params = sum(p.numel() for p in drnet_pp.parameters())
    
    print(f"DRNet with FPN parameters: {fpn_params:,}")
    print(f"DRNet with PyramidPooling parameters: {pp_params:,}")
    print(f"Difference: {fpn_params - pp_params:,}")
    
    return fpn_params, pp_params

if __name__ == "__main__":
    print("=" * 60)
    print("Feature Pyramid Network Implementation Test")
    print("=" * 60)
    
    try:
        # Run all tests
        test_fpn_module()
        test_drnet_with_fpn()
        test_drnet_with_pyramid_pooling()
        test_errnet_function()
        compare_parameter_counts()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! FPN implementation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
