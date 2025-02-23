import cv2
import numpy as np
import os

# Global lists to store selected canopy and no-tree colors
global canopy_colors, no_tree_colors
canopy_colors = []  # List to store colors selected as part of the tree canopy
no_tree_colors = []  # List to store colors selected as non-tree areas

def load_image(image_path):
    """Loads an image from the specified path."""
    return cv2.imread(image_path)

def pick_color(event, x, y, flags, param):
    """Mouse callback function to select colors for tree and non-tree areas."""
    global canopy_colors, no_tree_colors
    image, selection_type = param  # Get the image and selection type (canopy/no_tree)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get a 5x5 region around the click point
        region = image[max(0, y-2):min(y+3, image.shape[0]), max(0, x-2):min(x+3, image.shape[1])]
        avg_color = np.mean(region.reshape(-1, 3), axis=0)  # Calculate the average color in the region
        
        # Append the average color to the respective list based on the selection type
        if selection_type == "canopy" and len(canopy_colors) < 4:
            canopy_colors.append(avg_color)
            print(f"Selected canopy color {len(canopy_colors)}: {avg_color}")
        elif selection_type == "no_tree" and len(no_tree_colors) < 3:
            no_tree_colors.append(avg_color)
            print(f"Selected no-tree color {len(no_tree_colors)}: {avg_color}")
        
        # Stop further selection when the required number of colors is selected
        if len(canopy_colors) >= 4 and selection_type == "canopy":
            cv2.destroyAllWindows()
        if len(no_tree_colors) >= 3 and selection_type == "no_tree":
            cv2.destroyAllWindows()

def get_colors(image, selection_type):
    """Displays an image and allows the user to click on multiple areas to set colors."""
    global canopy_colors, no_tree_colors
    cv2.imshow(f"Select {selection_type.capitalize()} Colors (Click 4 for Canopy, 3 for No-Tree)", image)
    cv2.setMouseCallback(f"Select {selection_type.capitalize()} Colors (Click 4 for Canopy, 3 for No-Tree)", pick_color, (image, selection_type))
    cv2.waitKey(0)  # Wait for user input (mouse click)

def is_tree_pixel(pixel, reference_colors, threshold=50):
    """Determines if a pixel is likely part of a tree based on similarity to any of the selected canopy colors."""
    return any(np.linalg.norm(pixel - color) < threshold for color in reference_colors)

def analyze_grid(image, grid_size=(10, 10)):
    """Divides the image into a grid, analyzes tree coverage, and assigns a density value."""
    global canopy_colors, no_tree_colors
    
    # If colors are not selected yet, prompt the user to select them
    if not canopy_colors:
        print("Please select 4 tree canopy colors.")
        get_colors(image, "canopy")
    if not no_tree_colors:
        print("Please select 3 no-tree colors.")
        get_colors(image, "no_tree")
    
    height, width, _ = image.shape
    cell_height = height // grid_size[1]
    cell_width = width // grid_size[0]
    
    density_map = np.zeros(grid_size)  # Initialize a density map
    
    # Loop over each grid cell and calculate tree density
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            x_start = col * cell_width
            y_start = row * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height
            
            cell = image[y_start:y_end, x_start:x_end]  # Extract the cell
            # Count pixels that match canopy colors
            tree_pixels = np.sum([is_tree_pixel(pixel, canopy_colors) for pixel in cell.reshape(-1, 3)])
            tree_density = tree_pixels / (cell_width * cell_height)  # Calculate density as the ratio of tree pixels
            
            density_map[row, col] = tree_density  # Store the density value for the current cell
    
    return density_map

def apply_heatmap(image, density_map, grid_size):
    """Applies a heatmap overlay to indicate tree density."""
    height, width, _ = image.shape
    cell_height = height // grid_size[1]
    cell_width = width // grid_size[0]
    
    heatmap = np.zeros((height, width, 3), dtype=np.uint8)  # Initialize heatmap
    
    # Loop over each grid cell and apply color based on density
    for row in range(grid_size[1]):
        for col in range(grid_size[0]):
            x_start = col * cell_width
            y_start = row * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height
            
            intensity = int(density_map[row, col] * 255)  # Scale the density value to 0-255
            # Set color for the current cell: blue for low density, red for high density
            heatmap[y_start:y_end, x_start:x_end] = (0, intensity, 255 - intensity)
    
    # Blend the original image with the heatmap
    blended = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return blended

def process_directory(image_dir, grid_size=(10, 10)):
    """Processes all images in a directory and applies a heatmap."""
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            image = load_image(image_path)  # Load the image
            density_map = analyze_grid(image, grid_size)  # Analyze tree density
            result_image = apply_heatmap(image, density_map, grid_size)  # Apply heatmap to the image
            
            # Display the result
            cv2.imshow(f'Tree Coverage Heatmap - {filename}', result_image)
            cv2.waitKey(0)  # Wait for user input
            cv2.destroyAllWindows()  # Close the image window

def main(image_dir, grid_size=(10, 10)):
    """Runs the tree coverage analysis on all images in a directory with a heatmap."""
    process_directory(image_dir, grid_size)

if __name__ == "__main__":
    main("images")  # Replace with your actual directory path
