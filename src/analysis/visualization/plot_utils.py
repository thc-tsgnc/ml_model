# src/visualization/plot_utils.py

import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

def create_figure(figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create and return a new figure and axis.

    Args:
    figsize (Tuple[int, int]): The width and height of the figure in inches.

    Returns:
    Tuple[plt.Figure, plt.Axes]: A tuple containing the created Figure and Axes objects.
    """
    return plt.subplots(figsize=figsize)

def add_labels(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """
    Add labels to the given axes.

    Args:
    ax (plt.Axes): The axes object to add labels to.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def save_figure(fig: plt.Figure, filename: str, output_dir: str = 'results/plots') -> str:
    """
    Save the figure to a file and return the file path.

    Args:
    fig (plt.Figure): The figure to save.
    filename (str): The name of the file to save the figure as.
    output_dir (str): The directory to save the figure in. Default is 'results/plots'.

    Returns:
    str: The full path to the saved figure.
    """
    _create_directory(output_dir)
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return file_path

def add_grid(ax: plt.Axes, axis: str = 'both', linestyle: str = '--', alpha: float = 0.7) -> None:
    """
    Add a grid to the given axes.

    Args:
    ax (plt.Axes): The axes object to add the grid to.
    axis (str): Which axes to apply the grid to. Can be 'x', 'y', or 'both'.
    linestyle (str): The style of the grid lines.
    alpha (float): The transparency of the grid lines.
    """
    ax.grid(axis=axis, linestyle=linestyle, alpha=alpha)

def add_colorbar(fig: plt.Figure, mappable: plt.cm.ScalarMappable, label: Optional[str] = None) -> None:
    """
    Add a colorbar to the figure.

    Args:
    fig (plt.Figure): The figure to add the colorbar to.
    mappable (plt.cm.ScalarMappable): The mappable object to create the colorbar for.
    label (Optional[str]): The label for the colorbar. Default is None.
    """
    cbar = fig.colorbar(mappable)
    if label:
        cbar.set_label(label)

def set_axis_tick_params(ax: plt.Axes, axis: str = 'both', rotation: int = 0, ha: str = 'center') -> None:
    """
    Set tick parameters for the given axes.

    Args:
    ax (plt.Axes): The axes object to set tick parameters for.
    axis (str): Which axes to apply the parameters to. Can be 'x', 'y', or 'both'.
    rotation (int): The rotation angle of the tick labels.
    ha (str): The horizontal alignment of the tick labels.
    """
    ax.tick_params(axis=axis, rotation=rotation)
    if axis in ['x', 'both']:
        ax.set_xticklabels(ax.get_xticklabels(), ha=ha)

def _create_directory(directory: str) -> None:
    """
    Create a directory if it does not exist.

    Args:
    directory (str): The path to the directory to create.
    """
    os.makedirs(directory, exist_ok=True)
