from classes import Picture, Visualization

image = Picture('test_image.jpg')
image.show_properties()


resampled = image.resample(scale=0.4)
resampled.show_properties()

resampled.get_hex()


colors = resampled.get_hex(tol=60)
chart = Visualization(colors)
chart.color_list()
