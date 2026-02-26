import torch
from carle.env import CARLE

import matplotlib
import matplotlib.animation
import matplotlib.pyplot as plt

# for type hints
from matplotlib.axes import Axes
from matplotlib.figure import Figure 
from matplotlib.colors import LinearSegmentedColormap

def get_fig(grid, env, my_cmap="bone", invert=False, annotate_fig=True):

  global subplot_0

  fig, ax = plt.subplots(1,1, facecolor="black", figsize=(6,6))

  if invert:
      display_grid = 1.0 - plt.get_cmap(my_cmap)(grid.squeeze())
  else:
      display_grid = plt.get_cmap(my_cmap)(grid.squeeze())

  subplot_0 = ax.imshow(display_grid, cmap=my_cmap)

  ax.set_xticklabels("")
  ax.set_yticklabels("")

  def update_frame(grid, fig, ax):

    global subplot_0

    if invert:
        display_grid = 1.0 - plt.get_cmap(my_cmap)(grid.squeeze())
    else:
        display_grid = plt.get_cmap(my_cmap)(grid.squeeze())

    subplot_0.set_array(display_grid)
    plt.tight_layout()

  return fig, ax, update_frame


def save_vid(grid, env, number_frames: int, interval: int=30, my_cmap: str="bone", name: str="default"):

  MyWriter = matplotlib.animation.writers["ffmpeg"]
  metadata = dict(title="default", artist="Matplotlib", comment="")
  my_writer = MyWriter(fps=1000./interval, metadata=metadata)

  fig, ax, update_frame = get_fig(grid, env, my_cmap = my_cmap)

  filename = f"vid00_{name}.mp4"

  with my_writer.saving(fig, filename, number_frames):
    for frame in range(number_frames):

      my_writer.grab_frame()
      update_frame(grid, fig, ax)

      with torch.no_grad():
        grid = env(grid)
