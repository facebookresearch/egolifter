import argparse
from moviepy.editor import VideoFileClip


def convert_mp4_to_gif(input_file, output_file):
    clip = VideoFileClip(input_file)
    clip.write_gif(output_file, fps=20, program='ffmpeg', fuzz=1, opt='OptimizePlus')

    
def main():
    parser = argparse.ArgumentParser(description='Convert MP4 to GIF.')
    parser.add_argument('input_file', type=str, help='Input MP4 file')
    parser.add_argument('output_file', type=str, help='Output GIF file')
    args = parser.parse_args()
    convert_mp4_to_gif(args.input_file, args.output_file)

    
if __name__ == "__main__":
    main()