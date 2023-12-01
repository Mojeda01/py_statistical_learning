import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# A SIMPLE EXAMPLE
#------------------
#   Matplotlib graphs your data on Figures (e.g. windows, Jupyter widgets, etc), eac of which can
#   contain one or more Axes, or an area where points can be specified in terms of x-y coordinates
#   (or theta-r in polar plot, x-y-z in a 3D plot, etc.). The simplest way of creating a Figure
#   with an Axes is using pyplot.subplots. We can then use Axes.plot to draw some data on the
#   Axes:  

fig = plt.figure() # an empty figures with no Axes
fig, ax = plt.subplots() # Create a figure containing a single axes.
fig, axs = plt.subplots(2, 2) # A figure with a 2x2 grid of axes.

# A figure with one on the left, and two on the right:
fig, axs = plt.subplot_mosaic([["left", "right_top"],
                               ["left", "right_bottom"]])
plt.show()