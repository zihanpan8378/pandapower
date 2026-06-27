import csv
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
from typing import Any, List, Optional, TextIO


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


def plot_metric_over_time(
    metrics_df: pd.DataFrame,
    metric: str,
    title: str,
    y_label: str,
    x_label: str = "Date",
    grid_column: str = "grid",
    time_column: str = "timestamp",
    save_path: Optional[str] = None
) -> None:
    """
    Plots a recorded metric over time, one line per grid, on a single figure.

    Intended for the DataFrame returned by GridObserver.to_dataframe(), where each
    row is one grid at one timestamp.

    Args:
        metrics_df:  The recorded metrics, one row per grid per timestamp.
        metric:      The name of the metric column to plot on the y-axis.
        title:       The title of the plot.
        y_label:     The label for the y-axis.
        x_label:     The label for the x-axis.
        grid_column: The column identifying each grid/region (one line per value).
        time_column: The column holding the timestamp for the x-axis.
        save_path:   Optional path to save the figure to. If None, it is not saved.
    """
    plt.figure(figsize=(10, 6))
    # One line per grid, with points ordered along the time axis.
    for grid_name, group in metrics_df.groupby(grid_column):
        ordered = group.sort_values(time_column)
        plt.plot(
            pd.to_datetime(ordered[time_column]),
            ordered[metric],
            label=str(grid_name)
        )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.gcf().autofmt_xdate()
    if save_path:
        plt.savefig(save_path)
    plt.show()


class GridObserver:

    """
    Class that represents an observer of the grid simulation, used to record and
    persist grid metrics over time.

    Metrics are stored dynamically: each call to record_metrics() accepts an
    arbitrary set of named metric values, so new metrics can be tracked without
    changing this class. Every record is a single row (one grid at one timestamp)
    and is streamed straight to a CSV file as it is recorded, so partial results
    survive an interrupted run and memory use stays flat. The full history is also
    kept in memory and exposed as a DataFrame via to_dataframe() for plotting.
    """

    def __init__(self, grid_names: List[str], csv_path: str = "simulation_metrics.csv") -> None:
        """
        Constructs an instance of GridObserver.

        Args:
            grid_names: A list of names for each grid, indexed by grid id.
            csv_path:   Path of the CSV file metrics are streamed to. The file is
                        created (truncating any existing file) when the first record
                        is written, and its header is derived from that record's keys.
        """
        self._grid_names = grid_names
        self._csv_path = csv_path

        # Full history kept in memory for to_dataframe(); each entry is one row.
        self._records: List[dict[str, Any]] = []

        # CSV handle/writer are opened lazily on the first record so the header can
        # be derived from the metrics actually supplied.
        self._csv_file: Optional[TextIO] = None
        self._writer: Optional["csv.DictWriter"] = None
        self._fieldnames: Optional[List[str]] = None


    def record_metrics(self, grid_idx: int, timestamp: datetime, **metrics: float) -> None:
        """
        Records an arbitrary set of metrics for a given grid at a specific timestamp.

        Args:
            grid_idx:  The index of the grid to record metrics for.
            timestamp: The simulation time of this record; used as the row's index.
            **metrics: Named metric values to record (e.g. carbon_intensity=...,
                       carbon_emission_rate=...). The metric names must be consistent
                       across calls, since they define the CSV columns.
        """
        row: dict[str, Any] = {
            "timestamp": timestamp,
            "grid": self._grid_names[grid_idx],
            **metrics,
        }
        self._records.append(row)
        self.__write_row(row)


    def __write_row(self, row: dict[str, Any]) -> None:
        """
        Streams a single record to the CSV file, opening it and writing the header on
        first use.
        """
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._csv_file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        assert self._fieldnames is not None
        if set(row.keys()) != set(self._fieldnames):
            raise ValueError(
                f"Metric keys {sorted(row.keys())} do not match the CSV columns "
                f"{sorted(self._fieldnames)} established by the first record. The set "
                f"of metrics must be consistent across record_metrics() calls."
            )

        # Order values to match the header, then flush so a crash leaves the CSV
        # complete up to the last recorded step.
        self._writer.writerow({field: row[field] for field in self._fieldnames})
        assert self._csv_file is not None
        self._csv_file.flush()


    def to_dataframe(self) -> pd.DataFrame:
        """
        Returns all recorded metrics as a DataFrame (one row per grid per timestamp).
        """
        return pd.DataFrame(self._records)


    def close(self) -> None:
        """
        Closes the underlying CSV file. Safe to call multiple times; records flush
        after every write, so closing is optional but releases the file handle.
        """
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._writer = None


