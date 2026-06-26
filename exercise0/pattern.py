# pattern.py
import numpy as np
import matplotlib.pyplot as plt
import copy

class Checker:
    """
    Creates a checkerboard pattern with alternating black (0) and white (1) tiles.
    """
    
    def __init__(self, resolution, tile_size):
        """
        Initialize the checkerboard.
        
        Args:
            resolution: Total size of the image (e.g., 200 means 200x200 pixels)
            tile_size: Size of each tile (e.g., 25 means each tile is 25x25 pixels)
        """
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
        
    def draw(self):
        """
        Create the checkerboard pattern.
        """
        unit_size = 2 * self.tile_size
        if self.resolution % unit_size != 0:
            raise ValueError(f"Resolution {self.resolution} must be divisible by 2 * tile_size = {unit_size}")
        
        # 1. Create the basic 2x2 pattern (Top left must be black/0)
        basic_pattern = np.array([[0, 1], 
                                  [1, 0]])
        
        # 2. Tile the pattern
        num_repetitions = self.resolution // unit_size
        tiled_pattern = np.tile(basic_pattern, (num_repetitions, num_repetitions))
        
        # 3. Expand elements to tile_size x tile_size
        expanded = np.repeat(tiled_pattern, self.tile_size, axis=0)
        expanded = np.repeat(expanded, self.tile_size, axis=1)
        
        # Store and return a copy (Requirement for unit tests)
        self.output = expanded
        return self.output.copy()
    
    def show(self):
        """Display the checkerboard pattern."""
        if self.output is None: return
        
        plt.figure(figsize=(6, 6))
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Checkerboard: {self.resolution}x{self.resolution}, tile={self.tile_size}')
        plt.axis('off')
        plt.show()


class Circle:
    """
    Creates a binary circle pattern.
    """
    
    def __init__(self, resolution, radius, position):
        """
        Initialize the circle.
        
        Args:
            resolution: Size of the image (resolution x resolution)
            radius: Radius of the circle in pixels
            position: Tuple (x, y) for the center position
        """
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
        
    def draw(self):
        """
        Create a binary circle using the distance formula.
        """
        # 1. Create coordinate arrays
        x_coords = np.arange(self.resolution)
        y_coords = np.arange(self.resolution)
        
        # 2. Create meshgrid
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # 3. Calculate distance from center
        center_x, center_y = self.position
        distance = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        
        # 4. Create binary mask (distance <= radius)
        # NOTE: Unit tests expect a BOOLEAN array, not an integer array (1s/0s).
        self.output = (distance <= self.radius) 
        
        # Store and return a copy (Requirement for unit tests)
        return self.output.copy()
    
    def show(self):
        """Display the circle pattern."""
        if self.output is None: return
        
        # Displaying a boolean array uses 0 and 1 for vmin/vmax
        plt.figure(figsize=(6, 6))
        plt.imshow(self.output, cmap='gray', vmin=0, vmax=1)
        plt.title(f'Circle: radius={self.radius}, center={self.position}')
        plt.axis('off')
        plt.show()


class Spectrum:
    """
    Creates an RGB color spectrum.
    """
    
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None
        
    def draw(self):
        """
        Create an RGB spectrum with R=L-R, G=T-B, B=1-R.
        """
        
        self.output = np.zeros((self.resolution, self.resolution, 3), dtype=float)
        
        gradient = np.linspace(0.0, 1.0, self.resolution, dtype=float)
        
        # Red Channel (0): Increases left-to-right (X-axis)
        self.output[:, :, 0] = gradient[np.newaxis, :]
        
        # Green Channel (1): Increases top-to-bottom (Y-axis)
        self.output[:, :, 1] = gradient[:, np.newaxis]
        
        # Blue Channel (2) FIX: Inverse of Red channel to achieve the required corner colors
        # TL (B=1), TR (B=0), BL (B=1), BR (B=0).
        self.output[:, :, 2] = 1.0 - self.output[:, :, 0]
        
        return self.output.copy()
    
    def show(self):
        if self.output is None: return
        plt.figure(figsize=(8, 6))
        plt.imshow(self.output)
        plt.title('RGB Spectrum')
        plt.axis('off')
        plt.show()