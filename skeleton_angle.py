
from src.utils import *
from src.skeletonize import *
from src.angles import *
from pathlib import Path


def run_for_image(input_image, output_path):
    fil, fil_pruned = skeletonize(input_image)
    lines_data = find_lines(fil_pruned.skeleton_longpath)
    visualize_all(input_image, fil_pruned.skeleton_longpath, lines_data, output_path)


def run_for_file(input_file, output_file):
    image = load_image(input_file)
    run_for_image(image, output_file)


def run_for_dir(input_dir, out_dir):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for image_path, image in load_images(input_dir):
        output_path = os.path.join(out_dir, f"result_{Path(image_path).stem}.png") if out_dir else None
        run_for_image(image, output_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find skeleton and angle in hallux images')
    parser.add_argument('--input', required=True, help='directory of hallux images or a single image file')
    parser.add_argument('--output', required=False, help='output directory path or a file in case of single file input')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        run_for_dir(args.input, args.output)
    else:
        run_for_file(args.input, args.output)
