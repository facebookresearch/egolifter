import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import subprocess
import concurrent.futures
from tqdm import tqdm
from torch import nn


def load_image(image_path, device, rot_back=False):
    # Load an image using PIL and convert it to a tensor
    image = Image.open(image_path).convert("RGB")
    # rotate the image 90 degrees clockwise
    if rot_back:
        image = image.rotate(-90)
    image = transforms.ToTensor()(image).unsqueeze(0)
    return image.to(device)


def compute_difference(img1, img2):
    # Compute per-pixel mean of the per-channel absolute difference
    num_channels = img1.shape[1]
    return (
        torch.abs(img1 - img2).mean(dim=1, keepdim=True).repeat(1, num_channels, 1, 1)
    )


def save_image(tensor, path):
    # Convert tensor to PIL image and save
    img = transforms.ToPILImage()(tensor.squeeze().cpu())
    img.save(path)


def create_video(input_dir, output_video_path, fps=10, image_width=1408):
    # Use FFmpeg to create a video from the images. Label "Ground Truth", "Rendered", and "Difference"
    # as text at column 704, 704+1408, 704+2*1408, respectively.
    # command = [
    #     "ffmpeg",
    #     "-y",
    #     "-framerate",
    #     str(fps),
    #     "-i",
    #     os.path.join(input_dir, "%05d_combined.png"),
    #     "-vf",
    #     f"drawtext=fontfile=Arial.ttf:fontsize=48:fontcolor=white:x={int(0.5*image_width)}:y=104:text='Ground Truth',"
    #     f"drawtext=fontfile=Arial.ttf:fontsize=48:fontcolor=white:x={int(1.5*image_width)}:y=104:text='Rendered',"
    #     f"drawtext=fontfile=Arial.ttf:fontsize=48:fontcolor=white:x={int(2.5*image_width)}:y=104:text='Difference'",
    #     # "-c:v",
    #     # "libopenh264",
    #     "-profile:v",
    #     "high",
    #     "-pix_fmt",
    #     "yuv420p",
    #     output_video_path,
    # ]
    
    # subprocess.run(command)
    
    # Load all images, and save them as a video using imageio.mimsave
    import imageio
    images = []
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".png"):
            images.append(imageio.imread(os.path.join(input_dir, filename)))
    imageio.mimsave(output_video_path, images, fps=fps)
    

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=0,
            bias=False,
        )
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x
        
        
def main(base_dir, gradient=False, rot_back=False):
    gt_dir = os.path.join(base_dir, "gt")
    rendered_dir = os.path.join(base_dir, "renders")
    
    if gradient:
        output_dir = os.path.join(base_dir, "output_gradient")
    else:
        output_dir = os.path.join(base_dir, "output")
        
    image_width = [640]

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sobel = Sobel().to(device)

    def process_image(gt_filename):
        if gt_filename.endswith(".png"):
            gt_path = os.path.join(gt_dir, gt_filename)
            rendered_path = os.path.join(rendered_dir, gt_filename)
            output_path = os.path.join(
                output_dir, gt_filename.split(".")[0] + "_combined.png"
            )
            gt_image = load_image(gt_path, device)
            rendered_image = load_image(rendered_path, device, rot_back=rot_back)
            image_width[0] = gt_image.shape[3]
            if gradient:
                gt_image = sobel(gt_image.mean(dim=1, keepdim=True))
                rendered_image = sobel(rendered_image.mean(dim=1, keepdim=True))
            diff_image = compute_difference(gt_image, rendered_image)
            combined_image = torch.cat([gt_image, rendered_image, diff_image], axis=3)
            save_image(combined_image, output_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_image, sorted(os.listdir(gt_dir))),
                total=len(os.listdir(gt_dir)),
            )
        )

    create_video(output_dir, os.path.join(output_dir, "output_video.mp4"), image_width=image_width[0])
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input_folder', type=str, required=True)
    parser.add_argument("-g", '--gradient', action='store_true')
    parser.add_argument("-r", '--rot_back', action='store_true')
    args = parser.parse_args()

    main(args.input_folder, gradient=args.gradient, rot_back=args.rot_back)
