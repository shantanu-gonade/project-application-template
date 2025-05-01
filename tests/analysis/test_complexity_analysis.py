"""
Tests for the complexity analysis module.
"""
import os
import json
from _pytest.compat import LEGACY_PATH
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from src.analysis.complexity_analysis import IssueComplexityAnalysis
from src.core.model import Issue, Event, State

def test_complexity_analysis_initialization():
    """
    Test that IssueComplexityAnalysis is correctly initialized.
    """
    analysis = IssueComplexityAnalysis()
    
    # Assert that the initialization values are correct
    assert analysis.name == "complexity_analysis"
    assert analysis.results == {}


def test_analyze(sample_issues: list[Issue], monkeypatch: pytest.MonkeyPatch):
    """
    Test the analyze method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Mock the helper methods
    analysis._calculate_complexity_scores = MagicMock(return_value=[
        {'issue_number': '1', 'score': 10.0, 'labels': ['bug', 'documentation']},
        {'issue_number': '2', 'score': 5.0, 'labels': ['enhancement']},
    ])
    analysis._analyze_complexity_resolution_correlation = MagicMock(return_value={
        'correlation': 0.75,
        'resolved_issues_count': 2
    })
    analysis._analyze_complexity_by_label = MagicMock(return_value={
        'label_complexity': {'bug': {'avg_complexity': 10.0, 'issue_count': 1}},
        'sorted_labels': [('bug', 10.0, 1)]
    })
    
    # Run the analysis
    results = analysis.analyze(sample_issues)
    
    # Check that the helper methods were called
    analysis._calculate_complexity_scores.assert_called_once_with(sample_issues)
    analysis._analyze_complexity_resolution_correlation.assert_called_once()
    analysis._analyze_complexity_by_label.assert_called_once()
    
    # Check the results
    assert results == {
        'complexity_scores': analysis._calculate_complexity_scores.return_value,
        'correlation_results': analysis._analyze_complexity_resolution_correlation.return_value,
        'label_complexity': analysis._analyze_complexity_by_label.return_value
    }
    
    # Check that the results were stored
    assert analysis.results == results

def test_calculate_complexity_scores(sample_issues):
    """
    Test the _calculate_complexity_scores method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()

    # Run the analysis
    scores = analysis._calculate_complexity_scores(sample_issues)

    # Check that we have scores for all issues
    assert len(scores) == len(sample_issues)

    # Check that each score has the expected fields
    for score in scores:
        assert 'issue_number' in score
        assert 'title' in score
        assert 'score' in score
        assert 'comment_count' in score
        assert 'discussion_duration' in score
        assert 'participant_count' in score
        assert 'has_code' in score
        assert 'labels' in score
        assert 'resolution_time' in score
        assert 'state' in score

    # Check specific known issue numbers
    print("Issues in scores:")
    for score in scores:
        print(f"issue_number: {score['issue_number']}")

    # Check for issue number '1'
    issue1_score = next((score for score in scores if str(score['issue_number']) == '1'), None)
    if issue1_score:
        assert issue1_score['comment_count'] == 2
        assert issue1_score['participant_count'] >= 2
    else:
        pytest.fail("Issue with issue_number '1' not found in scores")

    # Check for issue number '2'
    issue2_score = next((score for score in scores if str(score['issue_number']) == '2'), None)
    if issue2_score:
        assert issue2_score['comment_count'] == 1
        assert issue2_score['participant_count'] >= 2
    else:
        pytest.fail("Issue with issue_number '2' not found in scores")


def test_analyze_complexity_resolution_correlation_with_data():
    """
    Test the _analyze_complexity_resolution_correlation method with sufficient data.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Create test complexity scores
    complexity_scores = [
        {'issue_number': '1', 'score': 10.0, 'resolution_time': 5.0},
        {'issue_number': '2', 'score': 5.0, 'resolution_time': 2.0},
        {'issue_number': '3', 'score': 15.0, 'resolution_time': 8.0},
        {'issue_number': '4', 'score': 7.0, 'resolution_time': 3.0},
    ]
    
    # Run the analysis
    results = analysis._analyze_complexity_resolution_correlation(complexity_scores)
    
    # Check the results
    assert 'correlation' in results
    assert 'resolved_issues_count' in results
    assert 'scores' in results
    assert 'resolution_times' in results
    assert 'categories' in results
    assert 'avg_resolution_by_category' in results
    
    # Check that the correlation is calculated
    assert results['correlation'] is not None
    assert results['resolved_issues_count'] == 4
    
    # Check that the categories are assigned
    assert len(results['categories']) == 4
    assert set(results['categories']) <= set(['Low', 'Medium', 'High'])
    
    # Check that average resolution times by category are calculated
    assert len(results['avg_resolution_by_category']) > 0
    for category, avg_time in results['avg_resolution_by_category'].items():
        assert category in ['Low', 'Medium', 'High']
        assert avg_time > 0

def test_analyze_complexity_resolution_correlation_insufficient_data():
    """
    Test the _analyze_complexity_resolution_correlation method with insufficient data.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Create test complexity scores with only one resolved issue
    complexity_scores = [
        {'issue_number': '1', 'score': 10.0, 'resolution_time': 5.0},
        {'issue_number': '2', 'score': 5.0, 'resolution_time': None},
    ]
    
    # Run the analysis
    results = analysis._analyze_complexity_resolution_correlation(complexity_scores)
    
    # Check the results
    assert results['correlation'] is None
    assert results['resolved_issues_count'] == 1

def test_analyze_complexity_by_label():
    """
    Test the _analyze_complexity_by_label method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Create test complexity scores
    complexity_scores = [
        {'issue_number': '1', 'score': 10.0, 'labels': ['bug', 'documentation']},
        {'issue_number': '2', 'score': 5.0, 'labels': ['bug', 'enhancement']},
        {'issue_number': '3', 'score': 15.0, 'labels': ['bug', 'high-priority']},
        {'issue_number': '4', 'score': 7.0, 'labels': ['documentation', 'good first issue']},
        {'issue_number': '5', 'score': 12.0, 'labels': ['enhancement', 'feature']},
    ]
    
    # Run the analysis
    results = analysis._analyze_complexity_by_label(complexity_scores)
    
    # Check the results
    assert 'label_complexity' in results
    assert 'sorted_labels' in results
    
    # Check that label complexity is calculated
    label_complexity = results['label_complexity']
    assert 'bug' in label_complexity
    assert label_complexity['bug']['avg_complexity'] == 10.0  # (10 + 5 + 15) / 3
    assert label_complexity['bug']['issue_count'] == 3
    
    # Check that labels are sorted by complexity
    sorted_labels = results['sorted_labels']
    assert len(sorted_labels) > 0
    
    # Check that the first label is the most complex
    assert sorted_labels[0][0] in ['bug', 'high-priority', 'feature']  # One of these should be highest

def test_visualize(sample_issues: list[Issue], monkeypatch: pytest.MonkeyPatch):
    """
    Test the visualize method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results
    analysis.results = {
        'complexity_scores': [
            {'issue_number': '1', 'score': 10.0, 'comment_count': 2, 'discussion_duration': 5.0, 
             'participant_count': 3, 'has_code': True, 'labels': ['bug', 'documentation'], 
             'resolution_time': 5.0, 'state': State.closed},
            {'issue_number': '2', 'score': 5.0, 'comment_count': 1, 'discussion_duration': 0.0, 
             'participant_count': 2, 'has_code': False, 'labels': ['enhancement'], 
             'resolution_time': None, 'state': State.open},
        ],
        'correlation_results': {
            'correlation': 0.75,
            'resolved_issues_count': 1,
            'scores': [10.0],
            'resolution_times': [5.0],
            'categories': ['High'],
            'avg_resolution_by_category': {'High': 5.0}
        },
        'label_complexity': {
            'label_complexity': {
                'bug': {'avg_complexity': 10.0, 'issue_count': 1},
                'documentation': {'avg_complexity': 10.0, 'issue_count': 1},
                'enhancement': {'avg_complexity': 5.0, 'issue_count': 1}
            },
            'sorted_labels': [
                ('bug', 10.0, 1),
                ('documentation', 10.0, 1),
                ('enhancement', 5.0, 1)
            ]
        }
    }
    
    # Mock the helper methods
    analysis._visualize_complexity_distribution = MagicMock()
    analysis._visualize_complexity_resolution_correlation = MagicMock()
    analysis._visualize_complexity_by_label = MagicMock()
    
    # Run the visualization
    analysis.visualize(sample_issues)
    
    # Check that the helper methods were called
    analysis._visualize_complexity_distribution.assert_called_once()
    analysis._visualize_complexity_resolution_correlation.assert_called_once()
    analysis._visualize_complexity_by_label.assert_called_once()

def test_visualize_complexity_distribution(monkeypatch: pytest.MonkeyPatch):
    """
    Test the _visualize_complexity_distribution method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results
    analysis.results = {
        'complexity_scores': [
            {'issue_number': '1', 'score': 10.0, 'comment_count': 2, 'discussion_duration': 5.0, 
             'participant_count': 3, 'has_code': True},
            {'issue_number': '2', 'score': 5.0, 'comment_count': 1, 'discussion_duration': 0.0, 
             'participant_count': 2, 'has_code': False},
        ]
    }
    
    # Mock the create_histogram and create_box_plot functions
    mock_create_histogram = MagicMock(return_value=plt.figure())
    mock_create_box_plot = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.complexity_analysis.create_histogram", mock_create_histogram)
    monkeypatch.setattr("src.analysis.complexity_analysis.create_box_plot", mock_create_box_plot)
    
    # Mock plt.figure and plt.subplot
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("matplotlib.pyplot.subplot", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.hist", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_complexity_distribution()
    
    # Check that create_histogram and create_box_plot were called
    mock_create_histogram.assert_called_once()
    mock_create_box_plot.assert_called_once()
    
    # Check that save_figure was called three times (histogram, box plot, components)
    assert analysis.save_figure.call_count == 3
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "issue_complexity_scores_histogram" in filenames
    assert "issue_complexity_scores_box_plot" in filenames
    assert "complexity_components" in filenames

def test_visualize_complexity_resolution_correlation(monkeypatch: pytest.MonkeyPatch):
    """
    Test the _visualize_complexity_resolution_correlation method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results with correlation data
    analysis.results = {
        'correlation_results': {
            'correlation': 0.75,
            'resolved_issues_count': 3,
            'scores': [10.0, 5.0, 15.0],
            'resolution_times': [5.0, 2.0, 8.0],
            'categories': ['High', 'Low', 'High'],
            'avg_resolution_by_category': {'High': 6.5, 'Low': 2.0}
        }
    }
    
    # Mock the create_scatter_plot and create_bar_chart functions
    mock_create_scatter_plot = MagicMock(return_value=plt.figure())
    mock_create_bar_chart = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.complexity_analysis.create_scatter_plot", mock_create_scatter_plot)
    monkeypatch.setattr("src.analysis.complexity_analysis.create_bar_chart", mock_create_bar_chart)
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_complexity_resolution_correlation()
    
    # Check that create_scatter_plot and create_bar_chart were called
    mock_create_scatter_plot.assert_called_once()
    mock_create_bar_chart.assert_called_once()
    
    # Check that save_figure was called twice
    assert analysis.save_figure.call_count == 2
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "complexity_vs_resolution_time" in filenames
    assert "avg_resolution_time_vs_complexity_category" in filenames

def test_visualize_complexity_resolution_correlation_no_data(monkeypatch: pytest.MonkeyPatch):
    """
    Test the _visualize_complexity_resolution_correlation method with no correlation data.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results with no correlation
    analysis.results = {
        'correlation_results': {
            'correlation': None,
            'resolved_issues_count': 1
        }
    }
    
    # Mock print to check it's called
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    
    # Run the visualization
    analysis._visualize_complexity_resolution_correlation()
    
    # Check that print was called with the expected message
    mock_print.assert_called_with("Not enough data to visualize complexity-resolution correlation")

def test_visualize_complexity_by_label(monkeypatch: pytest.MonkeyPatch):
    """
    Test the _visualize_complexity_by_label method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results
    analysis.results = {
        'label_complexity': {
            'label_complexity': {
                'bug': {'avg_complexity': 10.0, 'issue_count': 3},
                'documentation': {'avg_complexity': 8.0, 'issue_count': 2},
                'enhancement': {'avg_complexity': 5.0, 'issue_count': 1}
            },
            'sorted_labels': [
                ('bug', 10.0, 3),
                ('documentation', 8.0, 2),
                ('enhancement', 5.0, 1)
            ]
        }
    }
    
    # Mock plt.figure and plt.barh
    mock_figure = MagicMock()
    mock_bars = [MagicMock() for _ in range(3)]  # 3 bars for 3 labels
    
    # Create a mock for plt.figure that returns our mock figure
    def mock_plt_figure(*args, **kwargs):
        return mock_figure
    
    # Create a mock for plt.barh that returns our mock bars
    def mock_plt_barh(*args, **kwargs):
        return mock_bars
    
    # Apply the patches
    monkeypatch.setattr("matplotlib.pyplot.figure", mock_plt_figure)
    monkeypatch.setattr("matplotlib.pyplot.barh", mock_plt_barh)
    monkeypatch.setattr("matplotlib.pyplot.text", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_complexity_by_label()
    
    # Check that save_figure was called with the correct arguments
    analysis.save_figure.assert_called_once()
    args, kwargs = analysis.save_figure.call_args
    assert args[0] is mock_figure
    assert args[1] == "avg_issue_complexity_vs_label"

def test_visualize_complexity_by_label_no_data(monkeypatch: pytest.MonkeyPatch):
    """
    Test the _visualize_complexity_by_label method with no label data.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results with no labels
    analysis.results = {
        'label_complexity': {
            'label_complexity': {},
            'sorted_labels': []
        }
    }
    
    # Mock print to check it's called
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    
    # Run the visualization
    analysis._visualize_complexity_by_label()
    
    # Check that print was called with the expected message
    mock_print.assert_called_with("Not enough data to visualize complexity by label")

def test_save_results(monkeypatch: pytest.MonkeyPatch, tmpdir: LEGACY_PATH):
    """
    Test the save_results method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results
    analysis.results = {
        'complexity_scores': [
            {'issue_number': '1', 'score': 10.0, 'state': State.closed},
            {'issue_number': '2', 'score': 5.0, 'state': State.open},
        ],
        'correlation_results': {
            'correlation': 0.75,
            'resolved_issues_count': 1
        },
        'label_complexity': {
            'label_complexity': {'bug': {'avg_complexity': 10.0, 'issue_count': 1}},
            'sorted_labels': [('bug', 10.0, 1)]
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
    
    # Check that the state enum was converted to string in the serialized scores
    serialized_scores = analysis.save_json.call_args_list[0][0][0]
    assert isinstance(serialized_scores[0]['state'], str)
    assert isinstance(serialized_scores[1]['state'], str)

def test_generate_report(monkeypatch: pytest.MonkeyPatch):
    """
    Test the _generate_report method.
    """
    # Create an IssueComplexityAnalysis instance
    analysis = IssueComplexityAnalysis()
    
    # Set up the results
    analysis.results = {
        'complexity_scores': [
            {'issue_number': '1', 'score': 10.0, 'comment_count': 2, 'discussion_duration': 5.0, 
             'participant_count': 3, 'has_code': True, 'title': 'Test Issue 1'},
            {'issue_number': '2', 'score': 5.0, 'comment_count': 1, 'discussion_duration': 0.0, 
             'participant_count': 2, 'has_code': False, 'title': 'Test Issue 2'},
        ],
        'correlation_results': {
            'correlation': 0.75,
            'resolved_issues_count': 1,
            'avg_resolution_by_category': {'High': 5.0}
        },
        'label_complexity': {
            'label_complexity': {
                'bug': {'avg_complexity': 10.0, 'issue_count': 1},
                'documentation': {'avg_complexity': 8.0, 'issue_count': 1},
                'enhancement': {'avg_complexity': 5.0, 'issue_count': 1}
            },
            'sorted_labels': [
                ('bug', 10.0, 1),
                ('documentation', 8.0, 1),
                ('enhancement', 5.0, 1)
            ]
        }
    }
    
    # Mock the Report class
    mock_report = MagicMock()
    mock_report_class = MagicMock(return_value=mock_report)
    monkeypatch.setattr("src.analysis.complexity_analysis.Report", mock_report_class)
    
    # Run the _generate_report method
    analysis._generate_report()
    
    # Check that Report was initialized correctly
    mock_report_class.assert_called_once_with("Issue Complexity Analysis Report", analysis.results_dir)
    
    # Check that add_section was called for each section
    assert mock_report.add_section.call_count >= 4  # At least 4 sections
    
    # Check that save_text_report and save_markdown_report were called
    mock_report.save_text_report.assert_called_once_with("results.txt")
    mock_report.save_markdown_report.assert_called_once_with("report.md")
