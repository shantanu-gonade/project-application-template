"""
Tests for the contributor analysis module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.analysis.contributor_analysis import ContributorAnalysis
from src.core.model import Issue, Event, State

def test_contributor_analysis_initialization():
    """
    Test that ContributorAnalysis is correctly initialized.
    """
    analysis = ContributorAnalysis()
    
    assert analysis.name == "contributor_analysis"
    assert analysis.results == {}

def test_analyze(sample_issues, monkeypatch):
    """
    Test the analyze method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Mock the helper methods
    analysis._identify_top_contributors = MagicMock(return_value={
        'top_issue_creators': {'user1': 2, 'user2': 1},
        'top_commenters': {'user2': 3, 'user1': 2},
        'top_issue_closers': {'user1': 1, 'user4': 1}
    })
    analysis._analyze_contributor_label_focus = MagicMock(return_value={
        'user1': {'bug': 2, 'documentation': 1},
        'user2': {'enhancement': 1}
    })
    analysis._analyze_activity_patterns = MagicMock(return_value={
        'activity_timeline': {'user1': [{'month': '2023-01', 'count': 3}]},
        'top_users': ['user1', 'user2']
    })
    analysis._analyze_response_times = MagicMock(return_value={
        'user1': {'avg_time': 1.5, 'issue_count': 2},
        'user2': {'avg_time': 0.5, 'issue_count': 1}
    })
    
    # Run the analysis
    results = analysis.analyze(sample_issues)
    
    # Check that the helper methods were called
    analysis._identify_top_contributors.assert_called_once_with(sample_issues)
    analysis._analyze_contributor_label_focus.assert_called_once_with(sample_issues)
    analysis._analyze_activity_patterns.assert_called_once_with(sample_issues)
    analysis._analyze_response_times.assert_called_once_with(sample_issues)
    
    # Check the results
    assert results == {
        'top_contributors': analysis._identify_top_contributors.return_value,
        'contributor_label_focus': analysis._analyze_contributor_label_focus.return_value,
        'activity_patterns': analysis._analyze_activity_patterns.return_value,
        'response_times': analysis._analyze_response_times.return_value
    }
    
    # Check that the results were stored
    assert analysis.results == results

def test_identify_top_contributors(sample_issues):
    """
    Test the _identify_top_contributors method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Run the analysis
    results = analysis._identify_top_contributors(sample_issues)
    
    # Check the results structure
    assert 'top_issue_creators' in results
    assert 'top_commenters' in results
    assert 'top_issue_closers' in results
    
    # Check specific results based on sample_issues
    # Issue creators
    assert 'user1' in results['top_issue_creators']
    assert results['top_issue_creators']['user1'] == 1  # user1 created 1 issue
    
    assert 'user2' in results['top_issue_creators']
    assert results['top_issue_creators']['user2'] == 1  # user2 created 1 issue
    
    # Commenters
    assert 'user2' in results['top_commenters']
    assert 'user3' in results['top_commenters']
    
    # Issue closers
    assert 'user1' in results['top_issue_closers']
    assert 'user4' in results['top_issue_closers']

def test_analyze_contributor_label_focus(sample_issues, monkeypatch):
    """
    Test the _analyze_contributor_label_focus method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Mock filter_issues_by_user to return specific issues for each user
    def mock_filter_issues_by_user(issues, user):
        if user == 'user1':
            return [issues[0]]  # user1's issue has bug and documentation labels
        elif user == 'user2':
            return [issues[1]]  # user2's issue has enhancement and good first issue labels
        else:
            return []
    
    monkeypatch.setattr("src.analysis.contributor_analysis.filter_issues_by_user", mock_filter_issues_by_user)
    
    # Run the analysis
    results = analysis._analyze_contributor_label_focus(sample_issues)
    
    # Check the results
    assert 'user1' in results
    assert 'bug' in results['user1']
    assert 'documentation' in results['user1']
    
    assert 'user2' in results
    assert 'enhancement' in results['user2']
    assert 'good first issue' in results['user2']

def test_analyze_activity_patterns(sample_issues):
    """
    Test the _analyze_activity_patterns method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Run the analysis
    results = analysis._analyze_activity_patterns(sample_issues)
    
    # Check the results structure
    assert 'activity_timeline' in results
    assert 'top_users' in results
    
    # Check that we have timelines for top users
    for user in results['top_users']:
        assert user in results['activity_timeline']
        assert isinstance(results['activity_timeline'][user], list)
        
        # Check that each timeline entry has the expected structure
        for entry in results['activity_timeline'][user]:
            assert 'month' in entry
            assert 'count' in entry

def test_analyze_response_times(sample_issues, monkeypatch):
    """
    Test the _analyze_response_times method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Mock filter_issues_by_user to return specific issues for each user
    def mock_filter_issues_by_user(issues, user):
        if user == 'user1':
            return [issues[0]]  # user1's issue has comments
        elif user == 'user2':
            return [issues[1]]  # user2's issue has a comment
        else:
            return []
    
    # Mock get_issue_first_response_time to return specific times
    def mock_get_issue_first_response_time(issue):
        if issue.number == '1':
            return 1.0  # 1 day for issue #1
        elif issue.number == '2':
            return 0.5  # 0.5 days for issue #2
        else:
            return None
    
    monkeypatch.setattr("src.analysis.contributor_analysis.filter_issues_by_user", mock_filter_issues_by_user)
    monkeypatch.setattr("src.utils.helpers.get_issue_first_response_time", mock_get_issue_first_response_time)
    
    # Run the analysis
    results = analysis._analyze_response_times(sample_issues)
    
    # Check the results
    assert 'user1' in results
    assert results['user1']['avg_time'] is not None
    assert results['user1']['issue_count'] == 1
    
    assert 'user2' in results
    assert results['user2']['avg_time'] == 1.0
    assert results['user2']['issue_count'] == 1

def test_visualize(sample_issues, monkeypatch):
    """
    Test the visualize method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Set up the results
    analysis.results = {
        'top_contributors': {
            'top_issue_creators': {'user1': 2, 'user2': 1},
            'top_commenters': {'user2': 3, 'user1': 2},
            'top_issue_closers': {'user1': 1, 'user4': 1}
        },
        'contributor_label_focus': {
            'user1': {'bug': 2, 'documentation': 1},
            'user2': {'enhancement': 1, 'good first issue': 1}
        },
        'activity_patterns': {
            'activity_timeline': {
                'user1': [{'month': '2023-01', 'count': 3}],
                'user2': [{'month': '2023-01', 'count': 2}]
            },
            'top_users': ['user1', 'user2']
        },
        'response_times': {
            'user1': {'avg_time': 1.5, 'issue_count': 2},
            'user2': {'avg_time': 0.5, 'issue_count': 1}
        }
    }
    
    # Mock the helper methods
    analysis._visualize_top_contributors = MagicMock()
    analysis._visualize_contributor_label_focus = MagicMock()
    analysis._visualize_activity_patterns = MagicMock()
    analysis._visualize_response_times = MagicMock()
    
    # Run the visualization
    analysis.visualize(sample_issues)
    
    # Check that the helper methods were called
    analysis._visualize_top_contributors.assert_called_once()
    analysis._visualize_contributor_label_focus.assert_called_once()
    analysis._visualize_activity_patterns.assert_called_once()
    analysis._visualize_response_times.assert_called_once()

def test_visualize_top_contributors(monkeypatch):
    """
    Test the _visualize_top_contributors method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Set up the results
    analysis.results = {
        'top_contributors': {
            'top_issue_creators': {'user1': 2, 'user2': 1},
            'top_commenters': {'user2': 3, 'user1': 2},
            'top_issue_closers': {'user1': 1, 'user4': 1}
        }
    }
    
    # Mock the create_bar_chart function
    mock_create_bar_chart = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.contributor_analysis.create_bar_chart", mock_create_bar_chart)
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_top_contributors()
    
    # Check that create_bar_chart was called three times (for creators, commenters, closers)
    assert mock_create_bar_chart.call_count == 3
    
    # Check that save_figure was called three times
    assert analysis.save_figure.call_count == 3
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "top_issue_creators" in filenames
    assert "top_commenters" in filenames
    assert "top_issue_closers" in filenames

def test_visualize_contributor_label_focus(monkeypatch):
    """
    Test the _visualize_contributor_label_focus method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Set up the results
    analysis.results = {
        'contributor_label_focus': {
            'user1': {'bug': 2, 'documentation': 1},
            'user2': {'enhancement': 1, 'good first issue': 1}
        }
    }
    
    # Mock the create_bar_chart function
    mock_create_bar_chart = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.contributor_analysis.create_bar_chart", mock_create_bar_chart)
    
    # Mock plt.figure and plt.bar
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("matplotlib.pyplot.bar", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xticks", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.legend", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_contributor_label_focus()
    
    # Check that create_bar_chart was called for each user
    assert mock_create_bar_chart.call_count == 2
    
    # Check that save_figure was called for each user plus the combined visualization
    assert analysis.save_figure.call_count == 3
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "user1_label_focus" in filenames
    assert "user2_label_focus" in filenames
    assert "contributor_label_heatmap" in filenames

def test_visualize_activity_patterns(monkeypatch):
    """
    Test the _visualize_activity_patterns method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Set up the results
    analysis.results = {
        'activity_patterns': {
            'activity_timeline': {
                'user1': [
                    {'month': '2023-01', 'count': 3},
                    {'month': '2023-02', 'count': 2}
                ],
                'user2': [
                    {'month': '2023-01', 'count': 1},
                    {'month': '2023-02', 'count': 4}
                ]
            },
            'top_users': ['user1', 'user2']
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
    analysis._visualize_activity_patterns()
    
    # Check that save_figure was called
    analysis.save_figure.assert_called_once()
    
    # Check the filename
    args, _ = analysis.save_figure.call_args
    assert args[1] == "activity_timeline"

def test_visualize_response_times(monkeypatch):
    """
    Test the _visualize_response_times method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Set up the results
    analysis.results = {
        'response_times': {
            'user1': {'avg_time': 1.5, 'issue_count': 2},
            'user2': {'avg_time': 0.5, 'issue_count': 1}
        }
    }
    
    # Mock plt.figure and plt.bar
    mock_figure = MagicMock()
    mock_bars = [MagicMock(), MagicMock()]
    
    def mock_plt_figure(*args, **kwargs):
        return mock_figure
    
    def mock_plt_bar(*args, **kwargs):
        return mock_bars
    
    monkeypatch.setattr("matplotlib.pyplot.figure", mock_plt_figure)
    monkeypatch.setattr("matplotlib.pyplot.bar", mock_plt_bar)
    monkeypatch.setattr("matplotlib.pyplot.text", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_response_times()
    
    # Check that save_figure was called
    analysis.save_figure.assert_called_once()
    
    # Check the filename
    args, _ = analysis.save_figure.call_args
    assert args[1] == "response_times"

def test_visualize_response_times_with_user_parameter(monkeypatch):
    """
    Test the _visualize_response_times method with a user parameter.
    """
    # Create a ContributorAnalysis instance with a user parameter
    analysis = ContributorAnalysis()
    analysis.user = "user1"
    
    # Set up the results
    analysis.results = {
        'response_times': {
            'user1': {'avg_time': 1.5, 'issue_count': 2},
            'user2': {'avg_time': 0.5, 'issue_count': 1}
        }
    }
    
    # Mock plt.figure and plt.bar
    mock_figure = MagicMock()
    mock_bars = [MagicMock(), MagicMock()]
    
    def mock_plt_figure(*args, **kwargs):
        return mock_figure
    
    def mock_plt_bar(*args, **kwargs):
        return mock_bars
    
    monkeypatch.setattr("matplotlib.pyplot.figure", mock_plt_figure)
    monkeypatch.setattr("matplotlib.pyplot.bar", mock_plt_bar)
    monkeypatch.setattr("matplotlib.pyplot.text", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the create_histogram function
    mock_create_histogram = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.contributor_analysis.create_histogram", mock_create_histogram)
    
    # Mock the load_data and filter_issues_by_user methods
    analysis.load_data = MagicMock(return_value=[])
    monkeypatch.setattr("src.analysis.contributor_analysis.filter_issues_by_user", MagicMock(return_value=[]))
    monkeypatch.setattr("src.analysis.contributor_analysis.get_issue_first_response_time", MagicMock(return_value=1.0))
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_response_times()
    
    # Check that save_figure was called at least once
    assert analysis.save_figure.call_count >= 1
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "response_times" in filenames

def test_save_results(monkeypatch, tmpdir):
    """
    Test the save_results method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Set up the results
    analysis.results = {
        'top_contributors': {
            'top_issue_creators': {'user1': 2, 'user2': 1},
            'top_commenters': {'user2': 3, 'user1': 2},
            'top_issue_closers': {'user1': 1, 'user4': 1}
        },
        'contributor_label_focus': {
            'user1': {'bug': 2, 'documentation': 1},
            'user2': {'enhancement': 1}
        },
        'activity_patterns': {
            'activity_timeline': {'user1': [{'month': '2023-01', 'count': 3}]},
            'top_users': ['user1', 'user2']
        },
        'response_times': {
            'user1': {'avg_time': 1.5, 'issue_count': 2},
            'user2': {'avg_time': 0.5, 'issue_count': 1}
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
    assert analysis.save_json.call_count == 4
    
    # Check that _generate_report was called
    analysis._generate_report.assert_called_once()

def test_generate_report(monkeypatch):
    """
    Test the _generate_report method.
    """
    # Create a ContributorAnalysis instance
    analysis = ContributorAnalysis()
    
    # Set up the results
    analysis.results = {
        'top_contributors': {
            'top_issue_creators': {'user1': 2, 'user2': 1},
            'top_commenters': {'user2': 3, 'user1': 2},
            'top_issue_closers': {'user1': 1, 'user4': 1}
        },
        'contributor_label_focus': {
            'user1': {'bug': 2, 'documentation': 1},
            'user2': {'enhancement': 1}
        },
        'activity_patterns': {
            'activity_timeline': {'user1': [{'month': '2023-01', 'count': 3}]},
            'top_users': ['user1', 'user2']
        },
        'response_times': {
            'user1': {'avg_time': 1.5, 'issue_count': 2},
            'user2': {'avg_time': 0.5, 'issue_count': 1}
        }
    }
    
    # Mock the Report class
    mock_report = MagicMock()
    mock_report_class = MagicMock(return_value=mock_report)
    monkeypatch.setattr("src.analysis.contributor_analysis.Report", mock_report_class)
    
    # Run the _generate_report method
    analysis._generate_report()
    
    # Check that Report was initialized correctly
    mock_report_class.assert_called_once_with("Contributor Analysis Report", analysis.results_dir)
    
    # Check that add_section was called for each section
    assert mock_report.add_section.call_count >= 4  # At least 4 sections
    
    # Check that save_text_report was called
    mock_report.save_text_report.assert_called_once_with("overall_analysis_report.txt")

def test_generate_report_with_user_parameter(monkeypatch):
    """
    Test the _generate_report method with a user parameter.
    """
    # Create a ContributorAnalysis instance with a user parameter
    analysis = ContributorAnalysis()
    analysis.user = "user1"
    
    # Set up the results
    analysis.results = {
        'top_contributors': {
            'top_issue_creators': {'user1': 2, 'user2': 1},
            'top_commenters': {'user2': 3, 'user1': 2},
            'top_issue_closers': {'user1': 1, 'user4': 1}
        },
        'contributor_label_focus': {
            'user1': {'bug': 2, 'documentation': 1},
            'user2': {'enhancement': 1}
        },
        'activity_patterns': {
            'activity_timeline': {'user1': [{'month': '2023-01', 'count': 3}]},
            'top_users': ['user1', 'user2']
        },
        'response_times': {
            'user1': {'avg_time': 1.5, 'issue_count': 2},
            'user2': {'avg_time': 0.5, 'issue_count': 1}
        }
    }
    
    # Mock the Report class
    mock_report = MagicMock()
    mock_user_report = MagicMock()
    
    def mock_report_class(title, results_dir):
        if title == "Contributor Analysis Report":
            return mock_report
        else:
            return mock_user_report
    
    monkeypatch.setattr("src.analysis.contributor_analysis.Report", mock_report_class)
    
    # Run the _generate_report method
    analysis._generate_report()
    
    # Check that both reports were created
    assert mock_report.add_section.call_count >= 4  # At least 4 sections in main report
    assert mock_user_report.add_section.call_count >= 2  # At least 2 sections in user report
    
    # Check that save_text_report was called for both reports
    mock_report.save_text_report.assert_called_once_with("overall_analysis_report.txt")
    mock_user_report.save_text_report.assert_called_once_with("user1_analysis_report.txt")
