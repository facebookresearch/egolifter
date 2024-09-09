import argparse
from PIL import Image

argparser = argparse.ArgumentParser()
argparser.add_argument("--img_path", type=str, required=True)
args = argparser.parse_args()
 
img_path = args.img_path

# Load the image
img = Image.open(img_path)

# Dimensions of the grid image
width, height = img.size

# Assuming the grid is 2x2 for the top row and 1x1 for the bottom row
top_img_width = width // 3
top_img_height = height // 2
bottom_img_width = width // 3
bottom_img_height = height // 2

# Coordinates of each sub-image in the grid
coords = [
    (0, 0, top_img_width, top_img_height),  # Top-left
    (top_img_width, 0, 2 * top_img_width, top_img_height),  # Top-middle
    (2 * top_img_width, 0, width, top_img_height),  # Top-right
    (0, top_img_height, bottom_img_width, height)  # Bottom
]

# Extract and save each sub-image
sub_images = []
for i, (left, upper, right, lower) in enumerate(coords):
    sub_img = img.crop((left, upper, right, lower))
    if img_path.endswith(".jpg"):
        sub_img_path = img_path.replace(".jpg", f"_{i}.jpg")
    elif img_path.endswith(".png"):
        sub_img_path = img_path.replace(".png", f"_{i}.png")
    else:
        raise ValueError("Image format not supported")
    sub_img.save(sub_img_path)
    sub_images.append(sub_img_path)