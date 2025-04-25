"""
Tests for the plotting module.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from src.visualization.plotting import (
    setup_figure,
    create_bar_chart,
    create_pie_chart,
    create_line_chart,
    create_scatter_plot,
    create_heatmap,
    create_histogram,
    create_box_plot,
    create_grouped_bar_chart,
    create_stacked_bar_chart
)

@pytest.fixture
def mock_plt(monkeypatch):
    """
    Mock matplotlib.pyplot to avoid actual rendering.
    """
    mock = MagicMock()
    monkeypatch.setattr("src.visualization.plotting.plt", mock)
    return mock

def test_setup_figure(mock_plt):
    """
    Test setting up a figure.
    """
    # Create a figure with default size
    fig = setup_figure()
    
    # Check that plt.figure was called with the default size
    mock_plt.figure.assert_called_once_with(figsize=(10, 6))
    
    # Create a figure with custom size
    fig = setup_figure(figsize=(12, 8))
    
    # Check that plt.figure was called with the custom size
    mock_plt.figure.assert_called_with(figsize=(12, 8))

def test_create_bar_chart(mock_plt):
    """
    Test creating a bar chart.
    """
    # Create a bar chart
    labels = ["A", "B", "C"]
    values = [1, 2, 3]
    fig = create_bar_chart(
        labels=labels,
        values=values,
        title="Test Bar Chart",
        xlabel="Categories",
        ylabel="Values",
        color="blue",
        rotation=90,
        figsize=(8, 6)
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(8, 6))
    
    # Check that plt.bar was called with the correct arguments
    mock_plt.bar.assert_called_once()
    args, kwargs = mock_plt.bar.call_args
    assert args[0] == labels
    assert args[1] == values
    assert kwargs["color"] == "blue"
    
    # Check that other plt functions were called
    mock_plt.title.assert_called_with("Test Bar Chart")
    mock_plt.xlabel.assert_called_with("Categories")
    mock_plt.ylabel.assert_called_with("Values")
    mock_plt.xticks.assert_called_with(rotation=90, ha='right')
    mock_plt.tight_layout.assert_called_once()

def test_create_pie_chart(mock_plt):
    """
    Test creating a pie chart.
    """
    # Create a pie chart
    values = [30, 20, 50]
    labels = ["A", "B", "C"]
    fig = create_pie_chart(
        values=values,
        labels=labels,
        title="Test Pie Chart",
        colors=["red", "green", "blue"],
        explode=[0.1, 0, 0],
        figsize=(8, 8)
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(8, 8))
    
    # Check that plt.pie was called with the correct arguments
    mock_plt.pie.assert_called_once()
    args, kwargs = mock_plt.pie.call_args
    assert args[0] == values
    assert kwargs["labels"] == labels
    assert kwargs["colors"] == ["red", "green", "blue"]
    assert kwargs["explode"] == [0.1, 0, 0]
    
    # Check that other plt functions were called
    mock_plt.title.assert_called_with("Test Pie Chart")
    mock_plt.axis.assert_called_with('equal')
    mock_plt.tight_layout.assert_called_once()

def test_create_line_chart(mock_plt):
    """
    Test creating a line chart.
    """
    # Create a line chart
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 15, 25, 30]
    fig = create_line_chart(
        x=x,
        y=y,
        title="Test Line Chart",
        xlabel="X Values",
        ylabel="Y Values",
        color="red",
        marker="x",
        figsize=(10, 5)
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(10, 5))
    
    # Check that plt.plot was called with the correct arguments
    mock_plt.plot.assert_called_once()
    args, kwargs = mock_plt.plot.call_args
    assert args[0] == x
    assert args[1] == y
    assert kwargs["color"] == "red"
    assert kwargs["marker"] == "x"
    
    # Check that other plt functions were called
    mock_plt.title.assert_called_with("Test Line Chart")
    mock_plt.xlabel.assert_called_with("X Values")
    mock_plt.ylabel.assert_called_with("Y Values")
    mock_plt.grid.assert_called_with(alpha=0.3)
    mock_plt.tight_layout.assert_called_once()

def test_create_scatter_plot(mock_plt):
    """
    Test creating a scatter plot.
    """
    # Create a scatter plot
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 15, 25, 30]
    fig = create_scatter_plot(
        x=x,
        y=y,
        title="Test Scatter Plot",
        xlabel="X Values",
        ylabel="Y Values",
        color="green",
        alpha=0.5,
        figsize=(10, 5),
        add_trendline=True
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(10, 5))
    
    # Check that plt.scatter was called with the correct arguments
    mock_plt.scatter.assert_called_once()
    args, kwargs = mock_plt.scatter.call_args
    assert args[0] == x
    assert args[1] == y
    assert kwargs["color"] == "green"
    assert kwargs["alpha"] == 0.5
    
    # Check that plt.plot was called for the trendline
    mock_plt.plot.assert_called_once()
    
    # Check that other plt functions were called
    mock_plt.title.assert_called_with("Test Scatter Plot")
    mock_plt.xlabel.assert_called_with("X Values")
    mock_plt.ylabel.assert_called_with("Y Values")
    mock_plt.grid.assert_called_with(alpha=0.3)
    mock_plt.tight_layout.assert_called_once()

def test_create_scatter_plot_no_trendline(mock_plt):
    """
    Test creating a scatter plot without a trendline.
    """
    # Create a scatter plot without a trendline
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 15, 25, 30]
    fig = create_scatter_plot(
        x=x,
        y=y,
        title="Test Scatter Plot",
        xlabel="X Values",
        ylabel="Y Values",
        add_trendline=False
    )
    
    # Check that plt.plot was not called (no trendline)
    mock_plt.plot.assert_not_called()

def test_create_heatmap(mock_plt, monkeypatch):
    """
    Test creating a heatmap.
    """
    # Mock seaborn
    mock_sns = MagicMock()
    monkeypatch.setattr("src.visualization.plotting.sns", mock_sns)
    
    # Create a heatmap
    data = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })
    fig = create_heatmap(
        data=data,
        title="Test Heatmap",
        cmap="viridis",
        annot=False,
        figsize=(12, 10)
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(12, 10))
    
    # Check that sns.heatmap was called with the correct arguments
    mock_sns.heatmap.assert_called_once()
    args, kwargs = mock_sns.heatmap.call_args
    assert args[0].equals(data)
    assert kwargs["cmap"] == "viridis"
    assert kwargs["annot"] is False
    
    # Check that other plt functions were called
    mock_plt.title.assert_called_with("Test Heatmap")
    mock_plt.tight_layout.assert_called_once()

def test_create_histogram(mock_plt):
    """
    Test creating a histogram.
    """
    # Create a histogram
    data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    fig = create_histogram(
        data=data,
        title="Test Histogram",
        xlabel="Values",
        ylabel="Frequency",
        bins=5,
        color="purple",
        figsize=(10, 6),
        show_mean=True,
        show_median=True
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(10, 6))
    
    # Check that plt.hist was called with the correct arguments
    mock_plt.hist.assert_called_once()
    args, kwargs = mock_plt.hist.call_args
    assert args[0] == data
    assert kwargs["bins"] == 5
    assert kwargs["color"] == "purple"
    
    # Check that plt.axvline was called twice (mean and median)
    assert mock_plt.axvline.call_count == 2
    
    # Check that other plt functions were called
    mock_plt.title.assert_called_with("Test Histogram")
    mock_plt.xlabel.assert_called_with("Values")
    mock_plt.ylabel.assert_called_with("Frequency")
    mock_plt.grid.assert_called_with(axis='y', alpha=0.3)
    mock_plt.legend.assert_called_once()
    mock_plt.tight_layout.assert_called_once()

def test_create_box_plot(mock_plt):
    """
    Test creating a box plot.
    """
    # Create a box plot
    data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    fig = create_box_plot(
        data=data,
        title="Test Box Plot",
        xlabel="Values",
        vertical=True,
        color="orange",
        figsize=(8, 6)
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(8, 6))
    
    # Check that plt.boxplot was called with the correct arguments
    mock_plt.boxplot.assert_called_once()
    args, kwargs = mock_plt.boxplot.call_args
    assert args[0] == data
    assert kwargs["vert"] is True
    
    # Check that other plt functions were called
    mock_plt.title.assert_called_with("Test Box Plot")
    mock_plt.xlabel.assert_called_with("Values")
    mock_plt.grid.assert_called_with(axis='y', alpha=0.3)
    mock_plt.tight_layout.assert_called_once()

def test_create_grouped_bar_chart(mock_plt):
    """
    Test creating a grouped bar chart.
    """
    # Create a grouped bar chart
    data = {
        "Group 1": [10, 20, 30],
        "Group 2": [15, 25, 35]
    }
    categories = ["A", "B", "C"]
    fig = create_grouped_bar_chart(
        data=data,
        categories=categories,
        title="Test Grouped Bar Chart",
        xlabel="Categories",
        ylabel="Values",
        colors=["blue", "red"],
        figsize=(12, 6)
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(12, 6))
    
    # Check that plt.bar was called twice (once for each group)
    assert mock_plt.bar.call_count == 2
    
    # Check that other plt functions were called
    mock_plt.xlabel.assert_called_with("Categories")
    mock_plt.ylabel.assert_called_with("Values")
    mock_plt.title.assert_called_with("Test Grouped Bar Chart")
    mock_plt.xticks.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.tight_layout.assert_called_once()

def test_create_stacked_bar_chart(mock_plt):
    """
    Test creating a stacked bar chart.
    """
    # Create a stacked bar chart
    data = {
        "Group 1": [10, 20, 30],
        "Group 2": [15, 25, 35]
    }
    categories = ["A", "B", "C"]
    fig = create_stacked_bar_chart(
        data=data,
        categories=categories,
        title="Test Stacked Bar Chart",
        xlabel="Categories",
        ylabel="Values",
        colors=["blue", "red"],
        figsize=(12, 6)
    )
    
    # Check that plt.figure was called with the correct size
    mock_plt.figure.assert_called_with(figsize=(12, 6))
    
    # Check that plt.bar was called twice (once for each group)
    assert mock_plt.bar.call_count == 2
    
    # Check that other plt functions were called
    mock_plt.xlabel.assert_called_with("Categories")
    mock_plt.ylabel.assert_called_with("Values")
    mock_plt.title.assert_called_with("Test Stacked Bar Chart")
    mock_plt.xticks.assert_called_once()
    mock_plt.legend.assert_called_once()
    mock_plt.tight_layout.assert_called_once()
