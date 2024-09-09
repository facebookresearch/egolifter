import numpy as np
import argparse
import os

def concatenate_npy_files(file_paths, output_file, axis=0):
    # List to hold data from all .npy files
    data_list = []

    # Iterate over all file paths provided
    for file_path in file_paths:
        if os.path.isfile(file_path) and file_path.endswith('.npy'):
            # Load the .npy file and add to the list
            data = np.load(file_path)
            data_list.append(data)
        else:
            print(f"Skipping invalid file: {file_path}")

    if not data_list:
        raise ValueError("No valid .npy files were provided.")
    
    # Concatenate all arrays along the specified axis
    concatenated_data = np.concatenate(data_list, axis=axis)

    # Save the concatenated array into a single .npy file
    np.save(output_file, concatenated_data)
    print(f"Concatenated array of shape {concatenated_data.shape} saved to {output_file}")

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Concatenate multiple .npy files into a single .npy file.")
    parser.add_argument('file_paths', nargs='+', help='Paths to the .npy files to concatenate.')
    parser.add_argument('--output', '-o', required=True, help='Path to save the concatenated .npy file.')
    parser.add_argument('--axis', '-a', type=int, default=0, help='Axis along which to concatenate.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the concatenate function
    concatenate_npy_files(args.file_paths, args.output, axis=args.axis)

if __name__ == '__main__':
    main()