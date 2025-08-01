import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

def find_smooth_contours(
    boolean_image: np.ndarray,
    min_contour_area: int = 50,
    gaussian_sigma: float = 0.5,
    method: str = 'interpolate',  # 'interpolate', 'subsample', or 'none'
    num_points: int = 100,
    epsilon_factor: float = 0.001,
    closing_kernel_size: int = 3
) -> List[np.ndarray]:
    """
    Find smooth borders around contiguous regions in a boolean image.
    
    Parameters:
    -----------
    boolean_image : np.ndarray
        2D boolean array where True/1 represents regions of interest
    min_contour_area : int
        Minimum area threshold for contours to be included
    gaussian_sigma : float
        Standard deviation for Gaussian smoothing before contour detection (light smoothing)
    method : str
        Smoothing method: 'interpolate' (uniform sampling), 'subsample' (Douglas-Peucker), 'none' (original points)
    num_points : int
        Number of points for 'interpolate' method
    epsilon_factor : float
        Approximation accuracy factor for 'subsample' method (as fraction of perimeter)
    closing_kernel_size : int
        Kernel size for morphological closing to fill small gaps
    
    Returns:
    --------
    List[np.ndarray]
        List of smooth contours, each as Nx2 array of (x,y) coordinates normalized to [-1,1]
        First and last points are identical for loop closure
    """
    
    # Convert boolean to uint8 if needed
    if boolean_image.dtype == bool:
        image = boolean_image.astype(np.uint8) * 255
    else:
        image = (boolean_image > 0).astype(np.uint8) * 255
    
    # Apply morphological closing to fill small gaps
    if closing_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (closing_kernel_size, closing_kernel_size))
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Apply very light Gaussian smoothing to reduce pixelation
    if gaussian_sigma > 0:
        image = gaussian_filter(image.astype(float), sigma=gaussian_sigma)
        image = (image > 127).astype(np.uint8) * 255
    
    # Find all contours including holes (enclaves)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    
    # Get image dimensions for normalization
    height, width = boolean_image.shape
    max_dim = max(height, width)
    
    smooth_contours = []
    
    for i, contour in enumerate(contours):
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        
        # Convert to proper format
        points = contour.reshape(-1, 2).astype(float)
        
        if len(points) < 3:
            continue
        
        # Determine if this is an external contour or a hole
        # hierarchy[0][i] = [next, previous, first_child, parent]
        is_hole = hierarchy[0][i][3] != -1  # Has a parent = it's a hole
        
        # Apply smoothing method
        if method == 'subsample':
            # Douglas-Peucker algorithm for intelligent subsampling
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            points = simplified.reshape(-1, 2).astype(float)
            processed_contour = points
            
        elif method == 'interpolate':
            # Uniform interpolation along the contour
            processed_contour = uniform_resample_contour(points, num_points)
            
        else:  # method == 'none'
            processed_contour = points
        
        # For holes, reverse the point order to maintain consistent winding
        if is_hole:
            processed_contour = processed_contour[::-1]
        
        # Normalize coordinates to [-1, 1] range
        center_x, center_y = width / 2, height / 2
        normalized_contour = np.zeros_like(processed_contour)
        normalized_contour[:, 0] = (processed_contour[:, 0] - center_x) / (max_dim / 2)
        normalized_contour[:, 1] = (processed_contour[:, 1] - center_y) / (max_dim / 2)
        
        # Ensure loop closure by repeating first point
        normalized_contour = np.vstack([normalized_contour, normalized_contour[0:1]])
        
        smooth_contours.append(normalized_contour)
    
    return smooth_contours


def uniform_resample_contour(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Resample contour points to have uniform spacing along the perimeter.
    Preserves shape while providing consistent point density.
    """
    # Calculate cumulative distance along contour
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
    
    # Add closing distance to complete the loop
    closing_dist = np.sqrt(np.sum((points[-1] - points[0])**2))
    total_perimeter = cumulative_dist[-1] + closing_dist
    
    # Create uniform sampling points along perimeter
    uniform_distances = np.linspace(0, total_perimeter, num_points + 1)[:-1]  # Exclude last point (same as first)
    
    # Handle the wrap-around by extending the contour
    extended_points = np.vstack([points, points[0:1]])  # Add first point at end
    extended_cumulative = np.concatenate([cumulative_dist, [total_perimeter]])
    
    # Interpolate to get uniform points
    interp_x = interp1d(extended_cumulative, extended_points[:, 0], kind='linear')
    interp_y = interp1d(extended_cumulative, extended_points[:, 1], kind='linear')
    
    uniform_x = interp_x(uniform_distances)
    uniform_y = interp_y(uniform_distances)
    
    return np.column_stack([uniform_x, uniform_y])


def visualize_contours(
    boolean_image: np.ndarray, 
    smooth_contours: List[np.ndarray],
    show_original: bool = True
) -> None:
    """
    Visualize the original image and extracted smooth contours.
    
    Parameters:
    -----------
    boolean_image : np.ndarray
        Original boolean image
    smooth_contours : List[np.ndarray]
        List of smooth contours from find_smooth_contours()
    show_original : bool
        Whether to show the original image as background
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original image
    ax1.imshow(boolean_image, cmap='gray', origin='lower')
    ax1.set_title('Original Boolean Image')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    
    # Plot normalized smooth contours
    if show_original:
        # Show original image in normalized coordinates for reference
        height, width = boolean_image.shape
        max_dim = max(height, width)
        extent = [-width/(max_dim), width/(max_dim), -height/(max_dim), height/(max_dim)]
        ax2.imshow(boolean_image, cmap='gray', alpha=0.3, extent=extent, origin='lower')
    
    # Plot smooth contours
    colors = plt.cm.tab10(np.linspace(0, 1, len(smooth_contours)))
    for i, contour in enumerate(smooth_contours):
        ax2.plot(contour[:, 0], contour[:, 1], 
                color=colors[i], linewidth=2, 
                label=f'Contour {i+1}')
    
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Smooth Normalized Contours')
    ax2.set_xlabel('Normalized X')
    ax2.set_ylabel('Normalized Y')
    
    if len(smooth_contours) <= 10:  # Only show legend if not too many contours
        ax2.legend()
    
    plt.tight_layout()
    plt.show()