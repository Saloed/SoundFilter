import matplotlib.pyplot as plotter
import numpy as np

from DA.Parameters import NUM_EPOCH


def new_figure(num):
    x = np.arange(0, NUM_EPOCH, 1)
    y = np.full(NUM_EPOCH, -1.1)
    fig = plotter.figure(num)
    fig.set_size_inches(1920, 1080)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(0, NUM_EPOCH)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation error')
    ax.autoscale(True, 'Y')
    ax.grid(True)
    line, = ax.plot(x, y, '.', color='r')
    fig.show(False)
    fig.canvas.draw()
    return line, fig


def update_figure(plot, axes, x, y):
    new_data = axes.get_ydata()
    new_data[x] = y
    axes.set_ydata(new_data)
    plot.canvas.draw()
