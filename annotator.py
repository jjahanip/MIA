import os
import numpy as np
import pandas as pd
from tifffile import memmap
from skimage.exposure import rescale_intensity
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider


class Annotator:
    def __init__(self, images, probability_table, output_dir):

        # read probability table
        self.table = probability_table

        # reorder images alphabetically
        image_names = [os.path.splitext(os.path.split(im)[1])[0].lower() for im in images]
        image_indices = sorted(range(len(image_names)), key=lambda k: image_names[k])
        images = [images[i] for i in image_indices]

        # get image names and column names alphabetically
        self.image_names = [os.path.splitext(os.path.split(im)[1])[0] for im in images]
        column_indices = [list(self.table.columns.str.lower()).index(im.lower()) for im in self.image_names]
        self.column_names = [list(self.table.columns)[i] for i in column_indices]

        # read images from memory-mapped files
        memmap_image_filenames = [os.path.join(output_dir, 'memmap', os.path.split(file)[1]) for file in images]
        self.read_memmap_images(memmap_image_filenames)

        # create thresholds array to generate classification table based on probability table
        self.thresholds = np.zeros(len(images))
        #TODO: change to dict
        # self.thresholds = dict(zip(self.column_names, np.zeros(len(images))))

        # color map for image
        color = [0, 1, 0]
        self.cmap = ListedColormap(np.vstack((np.linspace(0, color[0], 256), np.linspace(0, color[1], 256),
                                              np.linspace(0, color[2], 256), np.ones(256))).T)

    def read_memmap_images(self, image_filenames):
        self.memmap_images = []
        for image_filename in image_filenames:
            self.memmap_images.append(memmap(image_filename))

    def get_all_images(self):
        self.images = []
        for memmap_image in self.memmap_images:
            image = memmap_image[self.y0:self.y1, self.x0:self.x1]
            p2, p98 = np.percentile(image, (.1, 99.99))
            self.images.append(rescale_intensity(image, in_range=(p2, p98)))

    def get_new_table(self):
        self.table = self.table.loc[(self.table['xmin'] > self.x0) & (self.table['xmax'] < self.x1) &
                                    (self.table['ymin'] > self.y0) & (self.table['ymax'] < self.y1)].copy()
        self.table.loc[:, ['centroid_x', 'xmin', 'xmax']] -= self.x0
        self.table.loc[:, ['centroid_y', 'ymin', 'ymax']] -= self.y0

    def next_btn(self, event):
        # update index of current image
        self.idx += 1
        self.idx %= len(self.images)

        # update the image array
        self.aximg.set_array(self.images[self.idx])
        self.ax.set_title(self.image_names[self.idx])

        # update slider
        self.sldr.set_val(self.thresholds[self.idx])

    def prev_btn(self, event):
        # update index of current image
        self.idx -= 1
        self.idx %= len(self.images)

        # update the image array
        self.aximg.set_array(self.images[self.idx])
        self.ax.set_title(self.image_names[self.idx])

        # update slider
        self.sldr.set_val(self.thresholds[self.idx])

    def update_slider(self, val):
        # update the threshold for selected images
        self.thresholds[self.idx] = val

        self.axdots.set_xdata(self.table.loc[self.table[self.column_names[self.idx]] > val, 'centroid_x'].values)
        self.axdots.set_ydata(self.table.loc[self.table[self.column_names[self.idx]] > val, 'centroid_y'].values)
        self.fig.canvas.draw()

    def save_btn(self, event):
        for i in range(len(self.column_names)):
            self.table[self.column_names[i]] = (self.table[self.column_names[i]] > self.thresholds[0]).astype(int)

        plt.close()

    def annotate(self, start, end):
        self.x0, self.y0 = start
        self.x1, self.y1 = end

        # load all images based
        self.get_all_images()

        # set index of current figure to 0 (first image)
        self.idx = 0

        # create figure and axes
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(self.image_names[self.idx])

        # plot image
        self.aximg = self.ax.imshow(self.images[self.idx], cmap=self.cmap)
        self.aximg.axes.set_axis_off()

        # plot dots on top of image
        self.get_new_table()
        self.axdots, = self.ax.plot(self.table['centroid_x'].values, self.table['centroid_y'].values, 'r.')

        # SLIDER
        self.axslider = plt.axes([0.1, 0.07, 0.8, 0.03])
        self.sldr = Slider(self.axslider, ' ', 0, 1, valinit=0)
        self.sldr.on_changed(self.update_slider)

        # Next/Previous Button
        self.axprev = plt.axes([0.7, 0.01, 0.1, 0.05])
        self.axnext = plt.axes([0.81, 0.01, 0.1, 0.05])
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next_btn)
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.prev_btn)

        # Save Button
        self.axsave = plt.axes([0.1, 0.01, 0.1, 0.05])
        self.bsave = Button(self.axsave, 'Save')
        self.bsave.on_clicked(self.save_btn)

        plt.show(block=True)

        return self.table
