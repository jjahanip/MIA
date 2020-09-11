import os
import argparse
from utils import write_memmap
from thumbnail import Thumbnail


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thumbnail_image', type=str, default=r'E:\jahandar\generate_dataset\images\DAPI.tif')
    parser.add_argument('--channels', type=str, nargs='+', default=[r'E:\jahandar\generate_dataset\images\IBA1.tif',
                                                                    r'E:\jahandar\generate_dataset\images\CD31.tif',
                                                                    r'E:\jahandar\generate_dataset\images\NeuN.tif',
                                                                    r'E:\jahandar\generate_dataset\images\S100.tif'])
    parser.add_argument('--probability_table', type=str, default=r'E:\jahandar\generate_dataset\images\probability_table.csv')
    parser.add_argument('--output_dir', type=str, default=r'E:\jahandar\generate_dataset\out')
    parser.add_argument('--thumbnail_downscale', type=int, default=5)

    args = parser.parse_args()

    all_images = args.channels + [args.thumbnail_image]

    if not os.path.exists(os.path.join(args.output_dir, 'memmap')):
        write_memmap(all_images, args.output_dir)
    thumbnail_box = Thumbnail(args.thumbnail_image, args.thumbnail_downscale, args.channels, args.probability_table, args.output_dir)

