from moviepy.editor import VideoFileClip

# Load the MKV video file
input_file_path = './videos/clutter-2d-seg-interactive.mkv'
output_file_path = input_file_path.replace('.mkv', '.mp4')

# Convert MKV to MP4 without altering the frame rate, resolution, or quality
clip = VideoFileClip(input_file_path)
clip.write_videofile(output_file_path, codec='libx264', audio_codec='aac', remove_temp=True)