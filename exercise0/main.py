# main.py
import numpy as np
import matplotlib.pyplot as plt

# Assuming the required classes are available in 'pattern.py' and 'generator.py'
from pattern import Checker, Circle, Spectrum 
from generator import ImageGenerator 


def run_checker_tests():
   
    # Example 1: Simple 4x4 checkerboard (Minimal output)
    print("Testing 4x4 image, 1-pixel tiles...")
    checker1 = Checker(resolution=4, tile_size=1)
    checker1.draw()
    checker1.show() # Uncomment to show this small plot
    
    # Example 2: Larger checkerboard (Primary test)
    print("Testing 200x200 image, 25-pixel tiles...")
    checker2 = Checker(resolution=200, tile_size=25)
    pattern2 = checker2.draw()
    print(f"Shape: {pattern2.shape}, Unique values: {np.unique(pattern2)}")
    print(f"Corner values: {pattern2[0, 0]} (TL), {pattern2[0, 25]} (Start of 2nd tile)")
    checker2.show()
    
    # Example 3: Different tile size
    print("Testing 100x100 image, 10-pixel tiles...")
    checker3 = Checker(resolution=100, tile_size=10)
    checker3.draw()
    checker3.show()
    
    print("✓ Checkerboard tests completed!\n")


def run_circle_tests():
    """Test the Circle class and display results."""
    print("=" * 40)
    print("RUNNING CIRCLE TESTS")
    print("=" * 40)
    
    # Example 1: Circle in the center
    print("Testing circle in center (100x100, r=30, pos=50,50)...")
    circle1 = Circle(resolution=100, radius=30, position=(50, 50))
    pattern1 = circle1.draw()
    print(f"Pixel count: {np.sum(pattern1)}")
    circle1.show()
    
    # Example 2: Circle off-center
    print("Testing circle off-center (200x200, r=50, pos=75,150)...")
    circle2 = Circle(resolution=200, radius=50, position=(75, 150))
    circle2.draw()
    circle2.show()
    
    # Example 3: Small circle with area check
    print("Testing small circle (50x50, r=10, pos=25,25) with area check...")
    circle3 = Circle(resolution=50, radius=10, position=(25, 25))
    pattern3 = circle3.draw()
    theoretical_area = np.pi * (10**2)
    actual_area = np.sum(pattern3)
    print(f"Theoretical area (π*r²): {theoretical_area:.1f}, Actual pixel count: {actual_area}")
    circle3.show()
    
    print("✓ Circle tests completed!\n")


def run_spectrum_tests():
    """Test the Spectrum class and display results."""
    print("=" * 40)
    print("RUNNING RGB SPECTRUM TESTS")
    print("=" * 40)
    
    # Example 1: Full resolution spectrum
    print("Testing 200x200 RGB spectrum...")
    spectrum = Spectrum(resolution=200)
    pattern = spectrum.draw()
    
    print(f"Shape: {pattern.shape}, Data type: {pattern.dtype}")
    print(f"Value range: [{pattern.min():.2f}, {pattern.max():.2f}]")
    
    # Analyze corners
    corners = {
        'TL (0,0)': pattern[0, 0, :],
        'BR (199,199)': pattern[-1, -1, :]
    }
    
    # Define a simplified helper for corner analysis
    def get_corner_color_description(rgb):
        r, g, b = rgb
        if r < 0.1 and g < 0.1 and b < 0.1: return "Black/Dark"
        if r > 0.9 and g < 0.1 and b < 0.1: return "Red"
        if r < 0.1 and g > 0.9 and b < 0.1: return "Green"
        if r > 0.9 and g > 0.9 and b > 0.9: return "White/Light"
        if r > 0.9 and g > 0.9: return "Yellow"
        if r < 0.1 and g < 0.1 and b > 0.9: return "Blue"
        return "Mixed/Grey"

    for name, rgb in corners.items():
        print(f"{name}: R={rgb[0]:.2f}, G={rgb[1]:.2f}, B={rgb[2]:.2f} -> {get_corner_color_description(rgb)}")
    
    spectrum.show()
    
    print("✓ Spectrum tests completed!\n")


def run_generator_verification():
    """Test the ImageGenerator class and display a sample batch."""
    print("=" * 40)
    print("RUNNING IMAGE GENERATOR VERIFICATION")
    print("=" * 40)

    # Define dummy paths and parameters (consistent with generator.py's internal structure)
    IMG_PATH = 'data/exercise_data'
    LABEL_PATH = 'data/Labels.json'
    IMAGE_SIZE = [32, 32, 3] # Used in TestGen

    # Create generator with full augmentation enabled
    generator = ImageGenerator(
        file_path=IMG_PATH, 
        label_path=LABEL_PATH, 
        batch_size=12, 
        image_size=IMAGE_SIZE, 
        rotation=True, 
        mirroring=True, 
        shuffle=True
    )

    # Generate and plot a batch
    print(f"Generator initialized. Current Epoch: {generator.current_epoch()}")
    print("Generating and plotting a batch with augmentations (shuffle/rotation/mirroring enabled)...")
    generator.show()
    
    print("✓ ImageGenerator verification initiated (Check plot).\n")


if __name__ == "__main__":
    
    print("\n" + "█" * 40)
    print("█  PATTERN & GENERATOR MAIN TEST SCRIPT  █")
    print("█" * 40 + "\n")
    
    # Run all pattern tests
    run_checker_tests()
    run_circle_tests()
    run_spectrum_tests()
    
    # Run generator test
    run_generator_verification()
    
    