import matplotlib.pyplot as plt

from typing import List, Optional


def plot_data(
    x_data: List[float],
    y_data: List[List[float]] ,
    labels: List[str],
    title: str,
    x_label: str,
    y_label: str,
    save_path: Optional[str] = None
):
    """
    Plots the given data with appropriate labels and title.

    Args:
        x_data: The data for the x-axis.
        y_data: A list of lists, where each inner list contains the y-axis data for a specific line on the plot.
        labels: A list of labels corresponding to each line on the plot.
        title: The title of the plot.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        save_path: An optional path to save the generated plot image. If None, the plot will not be saved.
    """
    plt.figure(figsize=(10, 6))
    for y, label in zip(y_data, labels):
        plt.plot(x_data, y, label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    if save_path:
        plt.savefig(save_path)
    plt.show()


class GridObserver:
    """
    Class that represents an observer of the grid simulation, can be used to record
    and plot various grid metrics such as carbon intensity and emissions. 
    """
    
    def __init__(self, num_grids: int, grid_names: List[str]) -> None:
        """
        Constructs an instance of GridObserver.

        Args:
            num_grids:  The number of grids in the simulation, used to initialize data structures for recording metrics.
            grid_names: A list of names for each grid.
        """
        self._ci_data: List[List[float]] = [
            [] for _ in range(num_grids)
        ]
        self._ce_data: List[List[float]] = [
            [] for _ in range(num_grids)
        ]
        self._grid_names = grid_names
        

    def record_metrics(self, grid_idx: int, carbon_intensity: float, carbon_emission_rate: float) -> None:
        """
        Records the carbon intensity and carbon emission rate for a given grid at a specific time step.

        Args:
            grid_idx:               The index of the grid to record metrics for
            carbon_intensity:       The carbon intensity value to record
            carbon_emission_rate:   The carbon emission rate value to record
        """
        self._ci_data[grid_idx].append(carbon_intensity)
        self._ce_data[grid_idx].append(carbon_emission_rate)


