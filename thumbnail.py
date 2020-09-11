import os
import numpy as np
import pandas as pd
from tifffile import memmap
import matplotlib.pyplot as plt
from skimage.exposure import rescale_intensity
from matplotlib.widgets import RectangleSelector, Button, Slider

from annotator import Annotator

# plt.style.use('dark_background')


class Thumbnail:
    def __init__(self, image, downscale, channel_names, probability_table, output_dir):

        # [x0, y0] -> coordinates of top left selected box
        # [x1, y1] -> coordinates of top left selected box
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None

        # handles for fig, ax, rectangle selector
        self.fig = None         # figure object
        self.ax = None          # axes object
        self.RS = None          # rectangle selector object
        self.dsr = downscale    # DownScale Ratio
        self.thumbnail_width = 3

        self.channel_fnames = channel_names
        self.channels = [os.path.splitext(os.path.split(channel)[1])[0] for channel in self.channel_fnames]

        # read probability table
        self.probability_table = self.read_probability_table(probability_table)

        # generate groundtruth table based on probability table and set values of channels to NaN
        self.create_groundtruth_table()

        self.output_dir = output_dir

        # read thumbnail image
        self.image = None
        self.image_size = None

        self.read_image(image)

        self.plot_thumbnail()

    def read_image(self, image_filename):
        # read image
        self.image = memmap(os.path.join(self.output_dir, 'memmap', os.path.split(image_filename)[1]))
        self.image_size = self.image.shape[::-1]     # (width, height)

        # downscale image
        self.image = self.image[::self.dsr, ::self.dsr]

        # adjust intensity to 2% and 98% percentile
        p2, p98 = np.percentile(self.image, (2, 98))
        self.image = rescale_intensity(self.image, in_range=(p2, p98))

    def read_probability_table(self, probability_table_fname):
        probability_table = pd.read_csv(probability_table_fname)

        # check the table contains 'ID',  'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax'
        must_have_columns = ['ID', 'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax']
        assert all(item in probability_table.columns for item in must_have_columns), "table must contain 'ID', " \
                                                                                     "'centroid_x', 'centroid_y'," \
                                                                                     "'xmin', 'ymin', 'xmax', 'ymax'"

        # set ID as index
        probability_table.set_index('ID', inplace=True)

        # get column names of the selected channels
        column_indices = [list(probability_table.columns.str.lower()).index(im.lower()) for im in self.channels]
        self.column_names = [list(probability_table.columns)[i] for i in column_indices]

        # Just keep 'ID' (index), 'centroid_x', 'centroid_y', 'xmin', 'ymin', 'xmax', 'ymax' and selected channels
        columns_to_keep = list(probability_table.columns[:6]) + self.column_names
        return probability_table.loc[:, columns_to_keep]

    def create_groundtruth_table(self):
        self.groundtruth_table = self.probability_table.copy()

        # set the values of the channel columns to nan
        self.groundtruth_table.loc[:, self.column_names] = np.nan

    def postprocess_groundtruth_table(self):

        # drop rows with na
        self.groundtruth_table = self.groundtruth_table.dropna()

        # change to int
        self.groundtruth_table = self.groundtruth_table.astype(int)

        # drop rows with no class assigned (no biomarker)
        all_zeros = (self.groundtruth_table[self.column_names] != 0).sum(1) == 0
        self.groundtruth_table.drop(all_zeros[all_zeros].index, inplace=True)

        # drop rows with more than 1 class assigned (dual biomarker)
        dual_markers = (self.groundtruth_table[self.column_names] != 0).sum(1) > 1
        self.groundtruth_table.drop(dual_markers[dual_markers].index, inplace=True)

    def plot_thumbnail(self):

        # plot figure and set size
        fig_size_inch = self.thumbnail_width, self.thumbnail_width * self.image.shape[0] / self.image.shape[1]
        self.fig, self.ax = plt.subplots(figsize=fig_size_inch, dpi=150)

        # plot image
        plt.ion()
        plt.imshow(self.image, cmap='gray')
        plt.axis('off')
        plt.tight_layout()

        # add function when any key pressed
        plt.connect('key_press_event', self.toggle_selector)

        # create rectangle selector
        rectprops = dict(facecolor='white', edgecolor='white', linewidth=2, alpha=0.2, fill=True)
        self.RS = RectangleSelector(self.ax, self.onselect, drawtype='box', rectprops=rectprops, interactive=True)
        self.RS.set_active(False)

        # add button to initiate annotator
        axbtn = plt.axes([0.7, 0.05, 0.2, 0.075])
        annotate_btn = Button(axbtn, 'Annotate')
        annotate_btn.on_clicked(self.generate_annotator)

        # show plot
        plt.show(block=True)

        # postprocess and save groundtruth table
        self.postprocess_groundtruth_table()
        self.groundtruth_table.to_csv(os.path.join(self.output_dir, 'groundtruth_table.csv'))

        # temp
        # check the saved table
        from utils import center_image
        centers = self.groundtruth_table[['centroid_x', 'centroid_y']].values
        for ch in self.column_names:
            center_image(os.path.join(self.output_dir, ch + '.tif'),
                         centers[self.groundtruth_table[ch].values == 1, :],
                                 self.image_size)

    def generate_annotator(self, event):
        annotation = Annotator(self.channel_fnames, self.probability_table, self.output_dir)
        new_table = annotation.annotate([self.x0, self.y0], [self.x1, self.y1])

        # add the new table to the groundtruth table
        self.groundtruth_table.loc[new_table.index, self.column_names] = new_table.loc[:, self.column_names]

    def onselect(self, eclick, erelease):
        self.x0 = int(min(eclick.xdata, erelease.xdata)) * self.dsr
        self.x1 = int(max(eclick.xdata, erelease.xdata)) * self.dsr
        self.y0 = int(min(eclick.ydata, erelease.ydata)) * self.dsr
        self.y1 = int(max(eclick.ydata, erelease.ydata)) * self.dsr
        print('selected box = [{}, {}, {}, {}]'.format(self.x0, self.x1, self.y0, self.y1))

    def toggle_selector(self, event):
        if event.key in ['A', 'a'] and not self.RS.active:
            print(' RectangleSelector activated.')
            self.RS.set_active(True)
        elif event.key in ['A', 'a'] and self.RS.active:
            print(' RectangleSelector deactivated.')
            self.RS.set_active(False)
