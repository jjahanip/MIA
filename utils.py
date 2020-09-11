import os
import numpy as np
import pandas as pd
from PIL import Image
import PIL.ImageDraw as ImageDraw
from tifffile import memmap, imread, imsave
from skimage.util import img_as_ubyte


def write_memmap(image_filenames, output_dir, dtype='uint8'):
    if not os.path.exists(os.path.join(output_dir, 'memmap')):
        os.makedirs(os.path.join(output_dir, 'memmap'))

    output_dir = os.path.join(output_dir, 'memmap')

    for file in image_filenames:

        image = imread(file)

        if dtype == 'uint8':
            image = img_as_ubyte(image)

        new_file = os.path.join(output_dir, os.path.split(file)[1])
        memmap_image = memmap(new_file, shape=image.shape, dtype=image.dtype, bigtiff=True)
        memmap_image[:] = image[:]
        memmap_image.flush()
        del memmap_image


def center_image(file_name, centers, image_size, r=2, color=None):
    '''
    Save RGB image with centers
    :param file_name: tifffile to be saved
    :param centers: np.array [centroid_x centroid_y]
    :param image_size: [width height]
    :param r : radius of center
    :param color: color of center 'red', 'blue', None: gray image
    :return:
    '''

    if color is None:
        image = Image.new('L', image_size)
        color = 'white'
    else:
        image = Image.new('RGB', image_size)
    center_draw = ImageDraw.Draw(image)

    for center in centers:
        center_draw.ellipse((center[0] - r, center[1] - r, center[0] + r, center[1] + r), fill=color)

    try:
        image.save(file_name)
    except:
        imsave(file_name, np.array(image), bigtiff=True)


def crop_image_table(INPUT_DIR, OUTPUT_DIR, CHANNELS, PROB_TABLE, CROP_POSITION):

    # create dir to save images if not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    xmin, ymin, xmax, ymax = CROP_POSITION

    # read images -> crop -> save in output_dir
    for ch in CHANNELS:
        image = imread(os.path.join(INPUT_DIR, ch), plugin='tifffile')
        image = image[ymin:ymax, xmin:xmax]
        imsave(os.path.join(OUTPUT_DIR, ch), image, plugin='tifffile', bigtiff=True)

    # read table -> crop -> save in output_dir
    table = pd.read_csv(PROB_TABLE)


    new_table = table.loc[(table['xmin'] > xmin) & (table['xmax'] < xmax) & (table['ymin'] > ymin) & (table['ymax'] < ymax)].copy()
    new_table['ID'] = np.arange(1, len(new_table) + 1)
    new_table.set_index('ID', inplace=True)
    new_table.loc[:, ['centroid_x', 'xmin', 'xmax']] -= xmin
    new_table.loc[:, ['centroid_y', 'ymin', 'ymax']] -= ymin
    new_table.to_csv(os.path.join(OUTPUT_DIR, 'probability_table.csv'))