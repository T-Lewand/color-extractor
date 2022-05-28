import numpy as np
import pandas as pd
import extcolors
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import squarify
from colormap import rgb2hex

class Picture:
    def __init__(self, directory=None, image=None):
        if directory is None:
            self.image = image
        elif image is None:
            self.directory = directory
            self.format = directory.split('.')[-1]
            self.image = cv.imread(self.directory)
        else:
            print('No image passed, please give path to image or image array')

    def show_properties(self):
        shape = self.image.shape
        size = self.image.size
        bands = shape[-1]
        print(f'Size = {size}')
        print(f'Shape = {shape[0:2]}')
        print(f'Bands = {bands}')

    def show(self):
        cv.imshow('image', self.image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def resample(self, scale=None, width=None, height=None ):
        if scale is not None:
            width = int(self.image.shape[1] * scale)
            height = int(self.image.shape[0] * scale)
            size = (width, height)
        elif (width & height) is not None:
            size = (width, height)

        resampled = cv.resize(self.image, size, interpolation=cv.INTER_LINEAR)
        resmapled_dir = f'{self.directory}_resampled.{self.format}'
        cv.imwrite(resmapled_dir, resampled)
        return Picture(directory=resmapled_dir)

    def _extract_colors(self, tol=30, drop_biggest=False):
        pil_image = Image.fromarray(self.image, 'RGB')
        colors, pixel_count = extcolors.extract_from_image(pil_image, tolerance=tol)
        rgb_colors = [color[0] for color in colors]
        count = [count[-1] for count in colors]
        if drop_biggest:
            rgb_colors.pop(0)
            count.pop(0)
        return rgb_colors, count

    def get_rgb(self, tol=50, drop_biggest=False):
        rgb_colors, count = self._extract_colors(tol=tol, drop_biggest=drop_biggest)
        colors_df = pd.DataFrame(zip(rgb_colors, count), columns=['RGB', 'count'])
        return colors_df

    def get_hex(self, tol=50, drop_biggest=False):
        rgb_colors, count = self._extract_colors(tol=tol, drop_biggest=drop_biggest)
        hex_colors = [rgb2hex(color[0], color[1], color[2]) for color in rgb_colors]
        colors_df = pd.DataFrame(zip(hex_colors, count), columns=['Hex', 'count'])
        return colors_df

class Pil_Picture(Picture):
    def __init__(self, image=None, pil_image=None):
        if image is not None:
            self.pil_image = Image.open(image)
        elif pil_image is not None:
            self.pil_image = pil_image

        red, green, blue = self.pil_image.getchannel(0), self.pil_image.getchannel(1), self.pil_image.getchannel(2)
        self.image = np.dstack([np.array(red), np.array(green), np.array(blue)]).astype(int)

    def resample(self, scale=None, width=None, height=None):
        if scale is not None:
            width = int(self.image.shape[1] * scale)
            height = int(self.image.shape[0] * scale)
            size = (width, height)
        elif (width & height) is not None:
            size = (width, height)

        resampled = self.pil_image.resize(size, Image.LINEAR)

        return Pil_Picture(pil_image=resampled)

    def _extract_colors(self, tol=30, drop_biggest=False):

        colors, pixel_count = extcolors.extract_from_image(self.pil_image, tolerance=tol)
        rgb_colors = [color[0] for color in colors]
        count = [count[-1] for count in colors]
        if drop_biggest:
            rgb_colors.pop(0)
            count.pop(0)

        return rgb_colors, count

class Visualization:
    def __init__(self, colors):
        """

        :param colors: Data Frame with colors in hex code and pixel count
        """
        self.colors = colors

    def treemap(self):
        sns.set_style('whitegrid')
        color_palette = self.colors['Hex']
        label = self.colors['Hex']
        pixel_count = self.colors["count"]
        fig, ax = plt.subplots()
        squarify.plot(pixel_count, label=label, color=color_palette)
        plt.axis('off')
        return fig

    def color_list(self):

        colors_count = self.colors.shape[0]
        fig, ax = plt.subplots(figsize=(10, colors_count))
        single_color_space = 1/colors_count
        print(single_color_space)
        box_width = 0.4
        hspace = single_color_space/4
        box_height = single_color_space - hspace
        print(box_height)

        anchor = [0, 0]
        self.colors.sort_values('count', axis=0, ascending=True, inplace=True)

        for i in range(self.colors.shape[0]):
            color_name = self.colors.iloc[i, 0]

            color_box = patches.Rectangle(anchor, box_width, box_height, facecolor=color_name)
            ax.add_patch(color_box)
            ax.text(x=anchor[0]+box_width+0.05, y=anchor[1]+box_height/2, s=color_name, fontsize=15)
            anchor[1] += box_height + hspace + hspace/colors_count

        ax.axis('off')
        plt.tight_layout()

        return fig
