#!/usr/bin/env python3
"""
Image Quality Assessment Script
Evaluates image quality using BRISQUE and NIQE metrics
"""

import cv2
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.stats import gamma
import argparse


def estimate_aggd_parameters(block):
    """Estimate AGGD parameters for BRISQUE"""
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)
    r_gam = (gamma_func(1/gam) ** 2) / (gamma_func(2/gam) * gamma_func(1/gam))
    
    mean_abs = np.mean(np.abs(block))
    std_abs = np.sqrt(np.mean(block**2))
    
    if std_abs == 0:
        return 1, 0.1
    
    rho = mean_abs / std_abs
    rho_array = np.abs(rho - r_gam)
    min_idx = np.argmin(rho_array)
    alpha = gam[min_idx]
    
    beta_l = np.sqrt(gamma_func(1/alpha) / gamma_func(3/alpha))
    beta_r = np.sqrt(gamma_func(1/alpha) / gamma_func(3/alpha))
    
    return alpha, (beta_l + beta_r) / 2


def compute_brisque_features(img):
    """Compute BRISQUE features from image"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.astype(np.float64)
    
    # Compute MSCN coefficients
    mu = cv2.GaussianBlur(img, (7, 7), 1.16)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(img * img, (7, 7), 1.16)
    sigma = np.sqrt(np.abs(sigma - mu_sq))
    
    # Avoid division by zero
    sigma[sigma < 1] = 1
    mscn = (img - mu) / sigma
    
    # Compute features
    features = []
    
    # Shape parameters
    alpha, beta = estimate_aggd_parameters(mscn)
    features.extend([alpha, beta])
    
    # Compute pairwise products
    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
    for shift in shifts:
        shifted = np.roll(np.roll(mscn, shift[0], axis=0), shift[1], axis=1)
        pair = mscn * shifted
        alpha, beta = estimate_aggd_parameters(pair)
        
        # Also compute for left and right tails
        left = pair[pair < 0]
        right = pair[pair > 0]
        
        if len(left) > 0:
            alpha_l, beta_l = estimate_aggd_parameters(left)
        else:
            alpha_l, beta_l = 1, 0.1
            
        if len(right) > 0:
            alpha_r, beta_r = estimate_aggd_parameters(right)
        else:
            alpha_r, beta_r = 1, 0.1
            
        mean = np.mean(pair)
        features.extend([alpha, beta, alpha_l, beta_l, alpha_r, beta_r, mean])
    
    return np.array(features)


def calculate_brisque(img_path):
    """
    Calculate BRISQUE score for an image
    Note: This is a simplified version. For accurate results, 
    use pre-trained models from libraries like pyiqa or image-quality
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Extract features at two scales
        features1 = compute_brisque_features(img)
        
        # Downsample for second scale
        img_half = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        features2 = compute_brisque_features(img_half)
        
        # Combine features
        all_features = np.concatenate([features1, features2])
        
        # Simple scoring based on feature statistics
        # Note: This is a placeholder. Real BRISQUE uses SVM regression
        score = np.mean(np.abs(all_features - 1)) * 20
        
        return min(100, max(0, score))
        
    except Exception as e:
        print(f"Error calculating BRISQUE: {e}")
        return None


def calculate_niqe(img_path):
    """
    Calculate NIQE score for an image
    Note: This is a simplified version. For accurate results,
    use pre-trained models from libraries like pyiqa or image-quality
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert to grayscale
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        gray = gray.astype(np.float64)
        
        # Extract patches
        patch_size = 96
        patches = []
        
        h, w = gray.shape
        for i in range(0, h - patch_size, patch_size//2):
            for j in range(0, w - patch_size, patch_size//2):
                patch = gray[i:i+patch_size, j:j+patch_size]
                
                # Check if patch has sufficient variance
                if np.std(patch) > 0.5:
                    patches.append(patch)
        
        if len(patches) == 0:
            return 10.0  # Default score for low-variance images
        
        # Compute features for each patch
        features_list = []
        for patch in patches[:100]:  # Limit number of patches
            # Compute local normalized luminance
            mu = cv2.GaussianBlur(patch, (7, 7), 1.5)
            # Ensure non-negative variance
            var = cv2.GaussianBlur(patch**2, (7, 7), 1.5) - mu**2
            var = np.maximum(var, 0)  # Clip negative values to 0
            sigma = np.sqrt(var)
            sigma[sigma < 1] = 1
            
            normalized = (patch - mu) / sigma
            
            # Extract simple statistics as features
            features = [
                np.mean(normalized),
                np.std(normalized),
                np.percentile(normalized, 25),
                np.percentile(normalized, 75),
                np.mean(np.abs(normalized)),
                np.max(normalized),
                np.min(normalized)
            ]
            features_list.append(features)
        
        # Compute score based on feature statistics
        if len(features_list) == 0:
            return 5.0  # Default middle score if no valid features
            
        features_array = np.array(features_list)
        mean_features = np.mean(features_array, axis=0)
        std_features = np.std(features_array, axis=0)
        
        # Improved scoring based on deviation from expected natural image statistics
        # Natural images typically have certain statistical properties
        expected_mean = np.array([0, 1, -0.7, 0.7, 0.8, 3, -3])  # Expected values for natural images
        expected_std = np.array([0.1, 0.5, 0.5, 0.5, 0.3, 1, 1])  # Expected standard deviations
        
        # Calculate deviation from expected values
        mean_deviation = np.abs(mean_features - expected_mean)
        std_deviation = np.abs(std_features - expected_std)
        
        # Combine deviations with weights
        weights = np.array([1, 2, 1, 1, 1.5, 1, 1])  # Weight importance of different features
        score = np.sum(weights * (mean_deviation + std_deviation * 0.5)) / np.sum(weights)
        
        # Scale to 0-10 range
        score = score * 1.5
        
        return min(10, max(0.1, score))  # Ensure score is between 0.1 and 10
        
    except Exception as e:
        print(f"Error calculating NIQE: {e}")
        return None


def assess_image_quality(img_path):
    """
    Assess image quality using BRISQUE and NIQE metrics
    
    Args:
        img_path: Path to the image file
        
    Returns:
        Dictionary with BRISQUE and NIQE scores
    """
    print(f"\nAssessing image quality for: {img_path}")
    print("-" * 50)
    
    # Calculate BRISQUE score
    brisque_score = calculate_brisque(img_path)
    if brisque_score is not None:
        print(f"BRISQUE Score: {brisque_score:.2f}")
        print(f"  (0-100, lower is better)")
    else:
        print("BRISQUE Score: Failed to calculate")
    
    # Calculate NIQE score
    niqe_score = calculate_niqe(img_path)
    if niqe_score is not None:
        print(f"NIQE Score: {niqe_score:.2f}")
        print(f"  (0-10, lower is better)")
    else:
        print("NIQE Score: Failed to calculate")
    
    print("-" * 50)
    
    # Quality interpretation
    if brisque_score is not None:
        if brisque_score < 20:
            quality = "Excellent"
        elif brisque_score < 40:
            quality = "Good"
        elif brisque_score < 60:
            quality = "Fair"
        elif brisque_score < 80:
            quality = "Poor"
        else:
            quality = "Very Poor"
        print(f"Overall Quality Assessment: {quality}")
    
    return {
        "brisque": brisque_score,
        "niqe": niqe_score,
        "image_path": img_path
    }


def main():
    parser = argparse.ArgumentParser(
        description="Assess image quality using BRISQUE and NIQE metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python image.py photo.jpg
  python image.py /path/to/image.png

Note: This is a simplified implementation for demonstration.
For production use, consider installing specialized libraries:
  pip install pyiqa
  pip install image-quality
        """
    )
    
    parser.add_argument(
        "image_path",
        help="Path to the image file"
    )
    
    parser.add_argument(
        "--accurate",
        action="store_true",
        help="Use accurate implementation (requires pyiqa library)"
    )
    
    args = parser.parse_args()
    
    # Check if accurate mode is requested
    if args.accurate:
        try:
            import pyiqa
            print("\nUsing PyIQA for accurate assessment...")
            
            # Create metrics
            brisque_metric = pyiqa.create_metric('brisque')
            niqe_metric = pyiqa.create_metric('niqe')
            
            # Calculate scores
            brisque_score = brisque_metric(args.image_path).item()
            niqe_score = niqe_metric(args.image_path).item()
            
            print(f"\nAccurate BRISQUE Score: {brisque_score:.2f}")
            print(f"Accurate NIQE Score: {niqe_score:.2f}")
            
        except ImportError:
            print("\nPyIQA not installed. Install it with:")
            print("  pip install pyiqa torch torchvision")
            print("\nFalling back to simplified implementation...\n")
            assess_image_quality(args.image_path)
    else:
        # Use simplified implementation
        assess_image_quality(args.image_path)


if __name__ == "__main__":
    main()