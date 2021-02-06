"""The image module contains functions for plotting"""
from IPython import display
from matplotlib import pyplot as plt

__all__ = ['set_figsize', 'use_svg_display']


def set_figsize(figsize=(3.5, 2.5)):
    """Set matplotlib figure size."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def use_svg_display():
    """Use svg format to display plot in jupyter."""
    display.set_matplotlib_formats('svg')


# class Animator(object):
#     def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
#                  ylim=None, xscale='linear', yscale='linear', fmts=None,
#                  nrows=1, ncols=1, figsize=(3.5, 2.5)):
#         """Incrementally plot multiple lines."""
#         if legend is None:
#             legend = []
#         use_svg_display()
#         self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1: self.axes = [self.axes, ]
#         # use a lambda to capture arguments
#         self.config_axes = lambda: set_axes(
#             self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts
