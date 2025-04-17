"""
Common plotting functions for visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

def setup_figure(figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Sets up a matplotlib figure with the specified size.
    
    Args:
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    return plt.figure(figsize=figsize)

def create_bar_chart(
    labels: List[str], 
    values: List[float], 
    title: str, 
    xlabel: str, 
    ylabel: str,
    color: str = 'skyblue',
    rotation: int = 45,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Creates a bar chart.
    
    Args:
        labels: Labels for the x-axis
        values: Values for the y-axis
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color
        rotation: X-axis label rotation
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.bar(labels, values, color=color, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha='right')
    plt.tight_layout()
    return fig

def create_pie_chart(
    values: List[float], 
    labels: List[str], 
    title: str,
    colors: Optional[List[str]] = None,
    explode: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Creates a pie chart.
    
    Args:
        values: Values for the pie slices
        labels: Labels for the pie slices
        title: Chart title
        colors: Colors for the pie slices
        explode: Explode values for the pie slices
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.pie(
        values, 
        labels=labels, 
        autopct='%1.1f%%', 
        colors=colors,
        explode=explode,
        startangle=90
    )
    plt.title(title)
    plt.axis('equal')
    plt.tight_layout()
    return fig

def create_line_chart(
    x: List[Any], 
    y: List[float], 
    title: str, 
    xlabel: str, 
    ylabel: str,
    color: str = 'blue',
    marker: str = 'o',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Creates a line chart.
    
    Args:
        x: X-axis values
        y: Y-axis values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Line color
        marker: Marker style
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y, color=color, marker=marker)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def create_scatter_plot(
    x: List[float], 
    y: List[float], 
    title: str, 
    xlabel: str, 
    ylabel: str,
    color: str = 'blue',
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (10, 6),
    add_trendline: bool = False
) -> plt.Figure:
    """
    Creates a scatter plot.
    
    Args:
        x: X-axis values
        y: Y-axis values
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Point color
        alpha: Point transparency
        figsize: Figure size as (width, height) in inches
        add_trendline: Whether to add a trendline
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.scatter(x, y, color=color, alpha=alpha, edgecolors='w')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    
    # Add trendline if requested
    if add_trendline and len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8)
    
    plt.tight_layout()
    return fig

def create_heatmap(
    data: pd.DataFrame,
    title: str,
    cmap: str = 'coolwarm',
    annot: bool = True,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Creates a heatmap.
    
    Args:
        data: DataFrame containing the data
        title: Chart title
        cmap: Colormap
        annot: Whether to annotate cells
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    sns.heatmap(data, annot=annot, cmap=cmap, vmin=-1, vmax=1, fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    return fig

def create_histogram(
    data: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    bins: int = 20,
    color: str = 'skyblue',
    figsize: Tuple[int, int] = (10, 6),
    show_mean: bool = False,
    show_median: bool = False
) -> plt.Figure:
    """
    Creates a histogram.
    
    Args:
        data: Data to plot
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of bins
        color: Bar color
        figsize: Figure size as (width, height) in inches
        show_mean: Whether to show the mean line
        show_median: Whether to show the median line
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, alpha=0.7, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.3)
    
    # Add mean line
    if show_mean:
        mean_value = np.mean(data)
        plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {mean_value:.2f}')
    
    # Add median line
    if show_median:
        median_value = np.median(data)
        plt.axvline(median_value, color='green', linestyle='dashed', linewidth=1, 
                   label=f'Median: {median_value:.2f}')
    
    if show_mean or show_median:
        plt.legend()
    
    plt.tight_layout()
    return fig

def create_box_plot(
    data: List[float],
    title: str,
    xlabel: str,
    vertical: bool = True,
    color: str = 'lightblue',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Creates a box plot.
    
    Args:
        data: Data to plot
        title: Chart title
        xlabel: X-axis label (or Y-axis if vertical=False)
        vertical: Whether the box plot is vertical
        color: Box color
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    plt.boxplot(
        data, 
        vert=vertical, 
        patch_artist=True, 
        boxprops=dict(facecolor=color)
    )
    plt.title(title)
    
    if vertical:
        plt.xlabel(xlabel)
        plt.grid(axis='y', alpha=0.3)
    else:
        plt.ylabel(xlabel)
        plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_grouped_bar_chart(
    data: Dict[str, List[float]],
    categories: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Creates a grouped bar chart.
    
    Args:
        data: Dictionary mapping group names to lists of values
        categories: Category labels
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Colors for each group
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Set width of bars
    bar_width = 0.8 / len(data)
    
    # Set position of bars on x axis
    positions = np.arange(len(categories))
    
    # Create bars
    for i, (group, values) in enumerate(data.items()):
        offset = (i - len(data) / 2 + 0.5) * bar_width
        color = colors[i] if colors and i < len(colors) else None
        plt.bar(positions + offset, values, bar_width, label=group, color=color)
    
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(positions, categories)
    plt.legend()
    
    plt.tight_layout()
    return fig

def create_stacked_bar_chart(
    data: Dict[str, List[float]],
    categories: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Creates a stacked bar chart.
    
    Args:
        data: Dictionary mapping group names to lists of values
        categories: Category labels
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        colors: Colors for each group
        figsize: Figure size as (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Set position of bars on x axis
    positions = np.arange(len(categories))
    
    # Create bars
    bottom = np.zeros(len(categories))
    for i, (group, values) in enumerate(data.items()):
        color = colors[i] if colors and i < len(colors) else None
        plt.bar(positions, values, bottom=bottom, label=group, color=color)
        bottom += values
    
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(positions, categories)
    plt.legend()
    
    plt.tight_layout()
    return fig
