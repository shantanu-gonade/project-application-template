"""
Tests for the timeline analysis module.

"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.analysis.timeline_analysis import TimelineAnalysis
from src.core.model import Issue, Event, State

def test_timeline_analysis_initialization():
    """
    Test that TimelineAnalysis is correctly initialized.
    """
    analysis = TimelineAnalysis()
    
    assert analysis.name == "timeline_analysis"
    assert analysis.results == {}

def test_analyze(sample_issues, monkeypatch):
    """
    Test the analyze method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Mock the helper methods
    analysis._analyze_creation_closing_patterns = MagicMock(return_value={
        'creation_by_month': {'2023-01': 2, '2023-02': 1},
        'closing_by_month': {'2023-01': 1, '2023-02': 1},
        'cumulative_creation': {'2023-01': 2, '2023-02': 3},
        'cumulative_closing': {'2023-01': 1, '2023-02': 2},
        'open_issues': {'2023-01': 1, '2023-02': 1}
    })
    
    analysis._analyze_issue_lifecycle = MagicMock(return_value={
        'first_response_times': [1.0, 2.0],
        'resolution_times': [5.0, 10.0],
        'avg_first_response': 1.5,
        'median_first_response': 1.5,
        'avg_resolution': 7.5,
        'median_resolution': 7.5
    })
    
    analysis._analyze_activity_trends = MagicMock(return_value={
        'day_of_week': {
            'labels': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
            'counts': [2, 3, 1, 4, 2, 0, 0]
        },
        'hour_of_day': {
            'labels': list(range(24)),
            'counts': [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    })
    
    # Run the analysis
    results = analysis.analyze(sample_issues)
    
    # Check that the helper methods were called
    analysis._analyze_creation_closing_patterns.assert_called_once_with(sample_issues)
    analysis._analyze_issue_lifecycle.assert_called_once_with(sample_issues)
    analysis._analyze_activity_trends.assert_called_once_with(sample_issues)
    
    # Check the results
    assert results == {
        'creation_closing_patterns': analysis._analyze_creation_closing_patterns.return_value,
        'issue_lifecycle': analysis._analyze_issue_lifecycle.return_value,
        'activity_trends': analysis._analyze_activity_trends.return_value
    }
    
    # Check that the results were stored
    assert analysis.results == results

def test_analyze_creation_closing_patterns(sample_issues, monkeypatch):
    """
    Test the _analyze_creation_closing_patterns method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Mock the group_by_time_period function to return predictable results
    def mock_group_by_time_period(dates, period):
        if period == 'month':
            return {'2023-01': 2, '2023-02': 1}
        return {}
    
    monkeypatch.setattr("src.analysis.timeline_analysis.group_by_time_period", mock_group_by_time_period)
    
    # Run the analysis
    results = analysis._analyze_creation_closing_patterns(sample_issues)
    
    # Check the results structure
    assert 'creation_by_month' in results
    assert 'closing_by_month' in results
    assert 'cumulative_creation' in results
    assert 'cumulative_closing' in results
    assert 'open_issues' in results
    
    # Check specific values
    assert results['creation_by_month'] == {'2023-01': 2, '2023-02': 1}
    assert '2023-01' in results['closing_by_month']
    assert '2023-02' in results['closing_by_month']
    
    # Check cumulative counts
    assert results['cumulative_creation']['2023-01'] == 2
    assert results['cumulative_creation']['2023-02'] == 3
    
    # Check open issues
    assert results['open_issues']['2023-01'] == results['cumulative_creation']['2023-01'] - results['cumulative_closing']['2023-01']
    assert results['open_issues']['2023-02'] == results['cumulative_creation']['2023-02'] - results['cumulative_closing']['2023-02']

def test_analyze_issue_lifecycle(sample_issues, monkeypatch):
    """
    Test the _analyze_issue_lifecycle method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Mock helper functions
    def mock_get_issue_first_response_time(issue):
        if issue.number == 1:
            return 1.0  # 1 day for issue #1
        elif issue.number == 2:
            return 0.5  # 0.5 days for issue #2
        else:
            return None
    
    def mock_get_issue_resolution_time(issue):
        if issue.number == 1:
            return 9.0  # 9 days for issue #1
        elif issue.number == 4:
            return 1.0  # 1 day for issue #4
        else:
            return None
    
    monkeypatch.setattr("src.analysis.timeline_analysis.get_issue_first_response_time", mock_get_issue_first_response_time)
    monkeypatch.setattr("src.analysis.timeline_analysis.get_issue_resolution_time", mock_get_issue_resolution_time)
    
    # Run the analysis
    results = analysis._analyze_issue_lifecycle(sample_issues)
    
    # Check the results structure
    assert 'first_response_times' in results
    assert 'resolution_times' in results
    assert 'avg_first_response' in results
    assert 'median_first_response' in results
    assert 'avg_resolution' in results
    assert 'median_resolution' in results
    
    # Check that the lists contain values
    assert len(results['first_response_times']) > 0
    assert len(results['resolution_times']) > 0
    
    # Check averages and medians
    #print(results['first_response_times'])  # Print out the list of response times

    assert results['avg_first_response'] == 0.75  # (1.0 + 0.5) / 2
    assert results['median_first_response'] == 0.75  # (1.0 + 0.5) / 2
    assert results['avg_resolution'] == 5.0  # (9.0 + 1.0) / 2
    assert results['median_resolution'] == 5.0  # (9.0 + 1.0) / 2

def test_analyze_activity_trends(sample_issues, monkeypatch):
    """
    Test the _analyze_activity_trends method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Create a fixed date for testing
    monday = datetime(2023, 1, 2, 10, 0, 0)  # Monday at 10:00 AM
    tuesday = datetime(2023, 1, 3, 14, 0, 0)  # Tuesday at 2:00 PM
    wednesday = datetime(2023, 1, 4, 9, 0, 0)  # Wednesday at 9:00 AM
    
    # Mock the sample issues to have fixed dates
    for issue in sample_issues:
        issue.created_date = monday
    
    # Add some events with fixed dates
    sample_issues[0].events[0].event_date = tuesday
    sample_issues[1].events[0].event_date = wednesday
    
    # Run the analysis
    results = analysis._analyze_activity_trends(sample_issues)
    
    # Check the results structure
    assert 'day_of_week' in results
    assert 'hour_of_day' in results
    assert 'labels' in results['day_of_week']
    assert 'counts' in results['day_of_week']
    assert 'labels' in results['hour_of_day']
    assert 'counts' in results['hour_of_day']
    
    # Check day of week counts
    day_counts = results['day_of_week']['counts']
    assert day_counts[0] > 0  # Monday
    assert day_counts[1] > 0  # Tuesday
    assert day_counts[2] > 0  # Wednesday
    
    # Check hour of day counts
    hour_counts = results['hour_of_day']['counts']
    assert hour_counts[9] > 0  # 9:00 AM
    assert hour_counts[10] > 0  # 10:00 AM
    assert hour_counts[14] > 0  # 2:00 PM

def test_visualize(sample_issues, monkeypatch):
    """
    Test the visualize method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Set up the results
    analysis.results = {
        'creation_closing_patterns': {
            'creation_by_month': {'2023-01': 2, '2023-02': 1},
            'closing_by_month': {'2023-01': 1, '2023-02': 1},
            'cumulative_creation': {'2023-01': 2, '2023-02': 3},
            'cumulative_closing': {'2023-01': 1, '2023-02': 2},
            'open_issues': {'2023-01': 1, '2023-02': 1}
        },
        'issue_lifecycle': {
            'first_response_times': [1.0, 2.0],
            'resolution_times': [5.0, 10.0],
            'avg_first_response': 1.5,
            'median_first_response': 1.5,
            'avg_resolution': 7.5,
            'median_resolution': 7.5
        },
        'activity_trends': {
            'day_of_week': {
                'labels': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'counts': [2, 3, 1, 4, 2, 0, 0]
            },
            'hour_of_day': {
                'labels': list(range(24)),
                'counts': [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        }
    }
    
    # Mock the helper methods
    analysis._visualize_creation_closing_patterns = MagicMock()
    analysis._visualize_issue_lifecycle = MagicMock()
    analysis._visualize_activity_trends = MagicMock()
    
    # Run the visualization
    analysis.visualize(sample_issues)
    
    # Check that the helper methods were called
    analysis._visualize_creation_closing_patterns.assert_called_once()
    analysis._visualize_issue_lifecycle.assert_called_once()
    analysis._visualize_activity_trends.assert_called_once()

def test_visualize_creation_closing_patterns(monkeypatch):
    """
    Test the _visualize_creation_closing_patterns method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Set up the results
    analysis.results = {
        'creation_closing_patterns': {
            'creation_by_month': {'2023-01': 2, '2023-02': 1},
            'closing_by_month': {'2023-01': 1, '2023-02': 1},
            'cumulative_creation': {'2023-01': 2, '2023-02': 3},
            'cumulative_closing': {'2023-01': 1, '2023-02': 2},
            'open_issues': {'2023-01': 1, '2023-02': 1}
        }
    }
    
    # Mock plt.figure and plt.plot
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("matplotlib.pyplot.plot", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xticks", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.legend", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_creation_closing_patterns()
    
    # Check that save_figure was called twice
    assert analysis.save_figure.call_count == 2
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "Issue creation and closure over time" in filenames
    assert "Open issues over time" in filenames

def test_visualize_issue_lifecycle(monkeypatch):
    """
    Test the _visualize_issue_lifecycle method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Set up the results
    analysis.results = {
        'issue_lifecycle': {
            'first_response_times': [1.0, 2.0],
            'resolution_times': [5.0, 10.0],
            'avg_first_response': 1.5,
            'median_first_response': 1.5,
            'avg_resolution': 7.5,
            'median_resolution': 7.5
        }
    }
    
    # Mock the create_histogram function
    mock_create_histogram = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.timeline_analysis.create_histogram", mock_create_histogram)
    
    # Mock plt.figure and plt.bar
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("matplotlib.pyplot.bar", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xticks", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_issue_lifecycle()
    
    # Check that create_histogram was called twice
    assert mock_create_histogram.call_count == 2
    
    # Check that save_figure was called three times
    assert analysis.save_figure.call_count == 3
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "Time to first response" in filenames
    assert "Time to resolution" in filenames
    assert "Issue lifecycle" in filenames

def test_visualize_issue_lifecycle_no_data(monkeypatch):
    """
    Test the _visualize_issue_lifecycle method with no data.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Set up the results with no data
    analysis.results = {
        'issue_lifecycle': {
            'first_response_times': [],
            'resolution_times': [],
            'avg_first_response': None,
            'median_first_response': None,
            'avg_resolution': None,
            'median_resolution': None
        }
    }
    
    # Mock the create_histogram function
    mock_create_histogram = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.timeline_analysis.create_histogram", mock_create_histogram)
    
    # Mock plt.figure and plt.bar
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("matplotlib.pyplot.bar", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xticks", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_issue_lifecycle()
    
    # Check that create_histogram was not called
    assert mock_create_histogram.call_count == 0
    
    # Check that save_figure was called once (for the combined visualization)
    assert analysis.save_figure.call_count == 0

def test_analyze_empty_issues(sample_issues, monkeypatch):
    """
    Test the analyze method with an empty issue list.
    """
    analysis = TimelineAnalysis()
    
    # Mock the helper methods
    analysis._analyze_creation_closing_patterns = MagicMock(return_value={})
    analysis._analyze_issue_lifecycle = MagicMock(return_value={})
    analysis._analyze_activity_trends = MagicMock(return_value={})
    
    # Run the analysis
    results = analysis.analyze([])  # Empty list of issues
    
    # Check that the helper methods were called
    analysis._analyze_creation_closing_patterns.assert_called_once_with([])
    analysis._analyze_issue_lifecycle.assert_called_once_with([])
    analysis._analyze_activity_trends.assert_called_once_with([])
    
    # Ensure the results are empty
    assert results == {
        'creation_closing_patterns': {},
        'issue_lifecycle': {},
        'activity_trends': {}
    }



def test_analyze_issue_lifecycle_missing_data(sample_issues, monkeypatch):
    """
    Test the _analyze_issue_lifecycle method with missing first response and resolution times.
    """
    analysis = TimelineAnalysis()
    
    # Mock helper functions for missing data
    def mock_get_issue_first_response_time(issue):
        return None  # No response time for this issue
    
    def mock_get_issue_resolution_time(issue):
        return None  # No resolution time for this issue
    
    monkeypatch.setattr("src.analysis.timeline_analysis.get_issue_first_response_time", mock_get_issue_first_response_time)
    monkeypatch.setattr("src.analysis.timeline_analysis.get_issue_resolution_time", mock_get_issue_resolution_time)
    
    # Run the analysis
    results = analysis._analyze_issue_lifecycle(sample_issues)
    
    # Check that the lists contain None for missing data
    assert all(x is None for x in results['first_response_times'])
    assert all(x is None for x in results['resolution_times'])
    
    # Check averages and medians for missing data
    assert results['avg_first_response'] is None
    assert results['avg_resolution'] is None

def test_visualize_empty_results(sample_issues, monkeypatch):
    """
    Test the visualize method when results are empty.
    """
    analysis = TimelineAnalysis()
    
    # Set empty results
    analysis.results = {}
    
    # Mock the visualization helper methods
    analysis._visualize_creation_closing_patterns = MagicMock()
    analysis._visualize_issue_lifecycle = MagicMock()
    analysis._visualize_activity_trends = MagicMock()
    
    # Run the visualization
    analysis.visualize([])
    
    # Check that the helper methods were not called
    analysis._visualize_creation_closing_patterns.assert_not_called()
    analysis._visualize_issue_lifecycle.assert_not_called()
    analysis._visualize_activity_trends.assert_not_called()
def test_save_results_no_directory(sample_issues, monkeypatch):
    """
    Test the save_results method when results_dir is not set.
    """
    analysis = TimelineAnalysis()
    
    # Set up the results
    analysis.results = {
        'creation_closing_patterns': {'creation_by_month': {'2023-01': 2}},
        'issue_lifecycle': {'avg_first_response': 1.5},
        'activity_trends': {'day_of_week': {'counts': [1, 2, 3]}}
    }
    
    # Set results_dir to None
    analysis.results_dir = None
    
    # Mock the save_json method
    analysis.save_json = MagicMock()
    
    # Run save_results and expect no error
    analysis.save_results()
    
    # Ensure save_json was not called
    analysis.save_json.assert_not_called()

def test_generate_report_no_results(sample_issues, monkeypatch):
    """
    Test the _generate_report method with no results.
    """
    analysis = TimelineAnalysis()

    # Set empty results
    analysis.results = {}

    # Mock the Report class
    mock_report = MagicMock()
    mock_report_class = MagicMock(return_value=mock_report)
    monkeypatch.setattr("src.analysis.timeline_analysis.Report", mock_report_class)

    # Run the _generate_report method
    analysis._generate_report()

    # Check that the Report class was called
    mock_report_class.assert_called_once()  # Ensure the report was instantiated
    mock_report.add_section.assert_called()  # Ensure that sections were added to the report


def test_visualize_activity_trends(monkeypatch):
    """
    Test the _visualize_activity_trends method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Set up the results
    analysis.results = {
        'activity_trends': {
            'day_of_week': {
                'labels': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'counts': [2, 3, 1, 4, 2, 0, 0]
            },
            'hour_of_day': {
                'labels': list(range(24)),
                'counts': [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        }
    }
    
    # Mock the create_bar_chart function
    mock_create_bar_chart = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.timeline_analysis.create_bar_chart", mock_create_bar_chart)
    
    # Mock plt.figure and plt.subplot
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("matplotlib.pyplot.subplot", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.bar", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_activity_trends()
    
    # Check that create_bar_chart was called twice
    assert mock_create_bar_chart.call_count == 2
    
    # Check that save_figure was called three times
    assert analysis.save_figure.call_count == 3
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "Activity by day of week" in filenames
    assert "Activity by hour of day" in filenames
    assert "Activity trends" in filenames

def test_save_results(monkeypatch, tmpdir):
    """
    Test the save_results method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Set up the results
    analysis.results = {
        'creation_closing_patterns': {
            'creation_by_month': {'2023-01': 2, '2023-02': 1},
            'closing_by_month': {'2023-01': 1, '2023-02': 1}
        },
        'issue_lifecycle': {
            'avg_first_response': 1.5,
            'avg_resolution': 7.5
        },
        'activity_trends': {
            'day_of_week': {
                'labels': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'counts': [2, 3, 1, 4, 2, 0, 0]
            }
        }
    }
    
    # Set the results directory to a temporary directory
    analysis.results_dir = str(tmpdir)
    
    # Mock the save_json method
    analysis.save_json = MagicMock()
    
    # Mock the _generate_report method
    analysis._generate_report = MagicMock()
    
    # Run the save_results method
    analysis.save_results()
    
    # Check that save_json was called for each result
    assert analysis.save_json.call_count == 3
    
    # Check that _generate_report was called
    analysis._generate_report.assert_called_once()

def test_generate_report(monkeypatch):
    """
    Test the _generate_report method.
    """
    # Create a TimelineAnalysis instance
    analysis = TimelineAnalysis()
    
    # Set up the results
    analysis.results = {
        'creation_closing_patterns': {
            'creation_by_month': {'2023-01': 2, '2023-02': 1},
            'closing_by_month': {'2023-01': 1, '2023-02': 1}
        },
        'issue_lifecycle': {
            'avg_first_response': 1.5,
            'median_first_response': 1.5,
            'avg_resolution': 7.5,
            'median_resolution': 7.5
        },
        'activity_trends': {
            'day_of_week': {
                'labels': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                'counts': [2, 3, 1, 4, 2, 0, 0]
            },
            'hour_of_day': {
                'labels': list(range(24)),
                'counts': [0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        }
    }
    
    # Mock the Report class
    mock_report = MagicMock()
    mock_report_class = MagicMock(return_value=mock_report)
    monkeypatch.setattr("src.analysis.timeline_analysis.Report", mock_report_class)
    
    # Run the _generate_report method
    analysis._generate_report()
    
    # Check that Report was initialized correctly
    mock_report_class.assert_called_once_with("Timeline Analysis Report", analysis.results_dir)
    
    # Check that add_section was called for each section
    assert mock_report.add_section.call_count >= 4  # At least 4 sections
    
    # Check that save_text_report and save_markdown_report were called
    mock_report.save_text_report.assert_called_once_with("result.txt")
    mock_report.save_markdown_report.assert_called_once_with("report.md")
