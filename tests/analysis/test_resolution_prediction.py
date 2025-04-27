"""
Tests for the resolution prediction module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from src.analysis.resolution_prediction import IssueResolutionPrediction
from src.core.model import Issue, Event, State

@pytest.fixture
def sample_issues():
    return [
        Issue({
            'number': '1',
            'title': 'Fix login bug',
            'text': 'Steps to reproduce the bug...',
            'state': 'closed',
            'labels': ['bug'],
            'creator': 'user1',
            'created_date': '2023-01-01T00:00:00Z',
            'updated_date': '2023-01-10T00:00:00Z',
            'timeline_url': '',
            'events': []
        }),
        Issue({
            'number': '2',
            'title': 'Add dark mode',
            'text': 'It would be great to have a dark mode.',
            'state': 'closed',
            'labels': ['enhancement'],
            'creator': 'user2',
            'created_date': '2023-01-03T00:00:00Z',
            'updated_date': '2023-01-04T00:00:00Z',
            'timeline_url': '',
            'events': []
        }),
    ]

def test_resolution_prediction_initialization():
    """
    Test that IssueResolutionPrediction is correctly initialized.
    """
    analysis = IssueResolutionPrediction()
    
    assert analysis.name == "resolution_prediction"
    assert analysis.results == {}

def test_analyze(sample_issues, monkeypatch):
    """
    Test the analyze method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Create a mock DataFrame for issue features
    mock_df = pd.DataFrame({
        'issue_number': ['1', '2'],
        'title_length': [10, 15],
        'description_length': [100, 150],
        'label_count': [2, 3],
        'has_bug_label': [True, False],
        'has_feature_label': [False, True],
        'has_enhancement_label': [False, True],
        'comment_count': [2, 3],
        'participant_count': [2, 3],
        'has_code': [True, False],
        'resolution_time': [5.0, 10.0],
        'is_quick_resolution': [True, False]
    })
    
    # Mock the helper methods
    analysis._prepare_features = MagicMock(return_value=mock_df)
    analysis._analyze_resolution_time_distribution = MagicMock(return_value={
        'resolution_times': [5.0, 10.0],
        'mean_time': 7.5,
        'median_time': 7.5,
        'min_time': 5.0,
        'max_time': 10.0,
        'quick_resolution_pct': 50.0
    })
    analysis._analyze_feature_correlations = MagicMock(return_value={
        'correlation_matrix': {'resolution_time': {'comment_count': 0.8}},
        'resolution_correlations': {'comment_count': 0.8},
        'sorted_correlations': [('comment_count', 0.8)],
        'categorical_impact': {'has_bug_label': {'avg_time_true': 5.0, 'avg_time_false': 10.0}}
    })
    analysis._build_predictive_model = MagicMock(return_value={
        'success': True,
        'accuracy': 0.8,
        'sorted_importance': [('comment_count', 0.5), ('participant_count', 0.3)]
    })
    analysis._generate_recommendations = MagicMock(return_value={
        'recommendations': [{'feature': 'comment_count', 'recommendation': 'Add more comments', 'importance': 0.5}],
        'significant_factors': [{'factor': 'comment_count', 'difference': 20.0, 'direction': 'higher'}],
        'categorical_factors': [{'factor': 'has_bug_label', 'difference': -15.0, 'direction': 'less'}]
    })
    
    # Run the analysis
    results = analysis.analyze(sample_issues)
    
    # Check that the helper methods were called
    analysis._prepare_features.assert_called_once_with(sample_issues)
    analysis._analyze_resolution_time_distribution.assert_called_once_with(mock_df)
    analysis._analyze_feature_correlations.assert_called_once_with(mock_df)
    analysis._build_predictive_model.assert_called_once_with(mock_df)
    analysis._generate_recommendations.assert_called_once_with(mock_df, analysis._build_predictive_model.return_value)
    
    # Check the results
    assert results == {
        'issue_features': mock_df,
        'resolution_distribution': analysis._analyze_resolution_time_distribution.return_value,
        'feature_correlations': analysis._analyze_feature_correlations.return_value,
        'model_results': analysis._build_predictive_model.return_value,
        'recommendations': analysis._generate_recommendations.return_value
    }
    
    # Check that the results were stored
    assert analysis.results == results

def test_prepare_features(sample_issues, monkeypatch):
    """
    Test the _prepare_features method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Mock helper functions
    def mock_get_issue_resolution_time(issue):
        if issue.number == 1:
            return 9.0  # 9 days for issue #1
        elif issue.number == 2:
            return 1.0  # 1 day for issue #4
        else:
            return None
    
    def mock_get_issue_comment_count(issue):
        if issue.number == 1:
            return 2  # 2 comments for issue #1
        elif issue.number == 2:
            return 1  # 1 comment for issue #4
        else:
            return 0
    
    def mock_get_issue_unique_participants(issue):
        if issue.number == 1:
            return ['user1', 'user2', 'user3']  # 3 participants for issue #1
        elif issue.number == 2:
            return ['user4', 'user1']  # 2 participants for issue #4
        else:
            return [issue.creator]
    
    def mock_has_code_blocks(text):
        return False  # No code blocks for simplicity
    
    monkeypatch.setattr("src.analysis.resolution_prediction.get_issue_resolution_time", mock_get_issue_resolution_time)
    monkeypatch.setattr("src.analysis.resolution_prediction.get_issue_comment_count", mock_get_issue_comment_count)
    monkeypatch.setattr("src.analysis.resolution_prediction.get_issue_unique_participants", mock_get_issue_unique_participants)
    monkeypatch.setattr("src.analysis.resolution_prediction.has_code_blocks", mock_has_code_blocks)
    
    # Run the method
    df = analysis._prepare_features(sample_issues)
    
    # Check that the DataFrame has the expected structure
    assert isinstance(df, pd.DataFrame)
    # Check that at least some of the expected columns are present
    expected_columns = {'issue_number', 'title_length', 'description_length', 'label_count'}
    assert expected_columns.issubset(set(df.columns))

    
    # Check that only closed issues with valid resolution times are included
    assert len(df) == 2  # Only issues 1 and 2 are closed with valid resolution times
    
    # Check specific values for issue #1
    issue1 = df[df['issue_number'] == 1].iloc[0]
    assert issue1['resolution_time'] == 9.0
    assert issue1['is_quick_resolution'] == False  # 9 days > 7 days
    assert issue1['comment_count'] == 2
    assert issue1['participant_count'] == 3
    
    # Check specific values for issue #2
    issue2 = df[df['issue_number'] == 2].iloc[0]
    assert issue2['resolution_time'] == 1.0
    assert issue2['is_quick_resolution'] == True  # 1 day < 7 days
    assert issue2['comment_count'] == 1
    assert issue2['participant_count'] == 2

def test_analyze_resolution_time_distribution():
    """
    Test the _analyze_resolution_time_distribution method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Create a test DataFrame
    df = pd.DataFrame({
        'resolution_time': [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
        'is_quick_resolution': [True, True, True, True, False, False, False, False]
    })
    
    # Run the method
    results = analysis._analyze_resolution_time_distribution(df)
    
    # Check the results
    assert 'resolution_times' in results
    assert 'mean_time' in results
    assert 'median_time' in results
    assert 'min_time' in results
    assert 'max_time' in results
    assert 'quick_resolution_pct' in results
    
    # Check specific values
    assert results['mean_time'] == 8.0
    assert results['median_time'] == 8.0
    assert results['min_time'] == 1.0
    assert results['max_time'] == 15.0
    assert results['quick_resolution_pct'] == 50.0  # 4 out of 8 are quick

def test_analyze_feature_correlations():
    """
    Test the _analyze_feature_correlations method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Create a test DataFrame
    df = pd.DataFrame({
        'title_length': [10, 15, 20, 25, 30],
        'description_length': [100, 150, 200, 250, 300],
        'label_count': [1, 2, 3, 2, 1],
        'comment_count': [1, 2, 3, 4, 5],
        'participant_count': [1, 2, 3, 2, 1],
        'has_bug_label': [True, True, False, False, False],
        'has_feature_label': [False, False, True, True, False],
        'has_enhancement_label': [False, False, False, False, True],
        'has_code': [True, False, True, False, True],
        'resolution_time': [1.0, 3.0, 5.0, 7.0, 9.0]
    })
    
    # Run the method
    results = analysis._analyze_feature_correlations(df)
    
    # Check the results structure
    assert 'correlation_matrix' in results
    assert 'resolution_correlations' in results
    assert 'sorted_correlations' in results
    assert 'categorical_impact' in results
    
    # Check that correlations were calculated
    assert 'comment_count' in results['resolution_correlations']
    assert len(results['sorted_correlations']) > 0
    
    # Check categorical impact
    assert 'has_bug_label' in results['categorical_impact']
    assert 'avg_time_true' in results['categorical_impact']['has_bug_label']
    assert 'avg_time_false' in results['categorical_impact']['has_bug_label']
    assert 'difference' in results['categorical_impact']['has_bug_label']
    assert 'difference_pct' in results['categorical_impact']['has_bug_label']

def test_build_predictive_model_success():
    """
    Test the _build_predictive_model method with sufficient data.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Create a test DataFrame with sufficient data
    df = pd.DataFrame({
        'title_length': np.random.randint(10, 100, 50),
        'description_length': np.random.randint(100, 1000, 50),
        'label_count': np.random.randint(1, 5, 50),
        'has_bug_label': np.random.choice([True, False], 50),
        'has_feature_label': np.random.choice([True, False], 50),
        'has_enhancement_label': np.random.choice([True, False], 50),
        'comment_count': np.random.randint(0, 10, 50),
        'participant_count': np.random.randint(1, 5, 50),
        'has_code': np.random.choice([True, False], 50),
        'resolution_time': np.random.uniform(1, 30, 50),
        'is_quick_resolution': np.random.choice([True, False], 50)
    })
    
    # Mock sklearn functions to avoid actual model training
    mock_train_test_split = MagicMock(return_value=(None, None, None, None))
    mock_rf_classifier = MagicMock()
    mock_rf_classifier.feature_importances_ = np.array([0.1, 0.2, 0.3, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05])
    mock_rf_classifier.fit = MagicMock()
    mock_rf_classifier.predict = MagicMock(return_value=np.array([True, False]))
    
    mock_accuracy_score = MagicMock(return_value=0.8)
    mock_classification_report = MagicMock(return_value={'class1': {'precision': 0.8}})
    mock_confusion_matrix = MagicMock(return_value=np.array([[5, 2], [1, 7]]))
    
    with patch('src.analysis.resolution_prediction.train_test_split', mock_train_test_split), \
         patch('src.analysis.resolution_prediction.RandomForestClassifier', MagicMock(return_value=mock_rf_classifier)), \
         patch('src.analysis.resolution_prediction.accuracy_score', mock_accuracy_score), \
         patch('src.analysis.resolution_prediction.classification_report', mock_classification_report), \
         patch('src.analysis.resolution_prediction.confusion_matrix', mock_confusion_matrix), \
         patch('src.analysis.resolution_prediction.StandardScaler', MagicMock()):
        
        # Run the method
        results = analysis._build_predictive_model(df)
    
    # Check the results
    assert results['success'] == True
    assert 'model' in results
    assert 'accuracy' in results
    assert 'classification_report' in results
    assert 'confusion_matrix' in results
    assert 'feature_importance' in results
    assert 'sorted_importance' in results
    
    # Check specific values
    assert results['accuracy'] == 0.8
    assert len(results['sorted_importance']) == 9  # 9 features

def test_build_predictive_model_insufficient_data():
    """
    Test the _build_predictive_model method with insufficient data.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Create a test DataFrame with insufficient data
    df = pd.DataFrame({
        'title_length': [10, 15],
        'description_length': [100, 150],
        'label_count': [1, 2],
        'has_bug_label': [True, False],
        'has_feature_label': [False, True],
        'has_enhancement_label': [False, False],
        'comment_count': [1, 2],
        'participant_count': [1, 2],
        'has_code': [True, False],
        'resolution_time': [1.0, 3.0],
        'is_quick_resolution': [True, False]
    })
    
    # Run the method
    results = analysis._build_predictive_model(df)
    
    # Check the results
    assert results['success'] == False
    assert 'message' in results
    assert "Not enough data" in results['message']

def test_generate_recommendations():
    """
    Test the _generate_recommendations method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Create a test DataFrame
    df = pd.DataFrame({
        'title_length': [10, 15, 20, 25, 30, 35, 40, 45],
        'description_length': [100, 150, 200, 250, 300, 350, 400, 450],
        'label_count': [1, 2, 3, 2, 1, 2, 3, 2],
        'comment_count': [1, 2, 3, 4, 5, 6, 7, 8],
        'participant_count': [1, 2, 3, 2, 1, 2, 3, 2],
        'has_bug_label': [True, True, False, False, True, True, False, False],
        'has_feature_label': [False, False, True, True, False, False, True, True],
        'has_enhancement_label': [False, False, False, False, True, True, True, True],
        'has_code': [True, False, True, False, True, False, True, False],
        'resolution_time': [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
        'is_quick_resolution': [True, True, True, True, False, False, False, False]
    })
    
    # Create mock model results
    model_results = {
        'success': True,
        'sorted_importance': [
            ('comment_count', 0.3),
            ('participant_count', 0.2),
            ('description_length', 0.15),
            ('has_code', 0.12),
            ('label_count', 0.1),
            ('title_length', 0.05),
            ('has_bug_label', 0.03),
            ('has_feature_label', 0.03),
            ('has_enhancement_label', 0.02)
        ]
    }
    
    # Run the method
    results = analysis._generate_recommendations(df, model_results)
    
    # Check the results structure
    assert 'recommendations' in results
    assert 'significant_factors' in results
    assert 'categorical_factors' in results
    
    # Check that recommendations were generated
    assert len(results['recommendations']) > 0
    
    # Check that significant factors were identified
    for factor in results['significant_factors']:
        assert 'factor' in factor
        assert 'difference' in factor
        assert 'direction' in factor
    
    # Check that categorical factors were identified
    for factor in results['categorical_factors']:
        assert 'factor' in factor
        assert 'difference' in factor
        assert 'direction' in factor

def test_visualize(sample_issues, monkeypatch):
    """
    Test the visualize method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results
    analysis.results = {
        'issue_features': pd.DataFrame({
            'title_length': [10, 15],
            'description_length': [100, 150],
            'label_count': [1, 2],
            'comment_count': [1, 2],
            'participant_count': [1, 2],
            'resolution_time': [1.0, 3.0],
            'is_quick_resolution': [True, False]
        }),
        'resolution_distribution': {
            'resolution_times': [1.0, 3.0],
            'mean_time': 2.0,
            'median_time': 2.0,
            'min_time': 1.0,
            'max_time': 3.0,
            'quick_resolution_pct': 50.0
        },
        'feature_correlations': {
            'correlation_matrix': {'resolution_time': {'comment_count': 0.8}},
            'resolution_correlations': {'comment_count': 0.8},
            'sorted_correlations': [('comment_count', 0.8)],
            'categorical_impact': {'has_bug_label': {'avg_time_true': 5.0, 'avg_time_false': 10.0}}
        },
        'model_results': {
            'success': True,
            'accuracy': 0.8,
            'classification_report': {'class1': {'precision': 0.8}},
            'confusion_matrix': [[5, 2], [1, 7]],
            'feature_importance': {'comment_count': 0.5},
            'sorted_importance': [('comment_count', 0.5)]
        },
        'recommendations': {
            'recommendations': [{'feature': 'comment_count', 'recommendation': 'Add more comments', 'importance': 0.5}],
            'significant_factors': [{'factor': 'comment_count', 'difference': 20.0, 'direction': 'higher'}],
            'categorical_factors': [{'factor': 'has_bug_label', 'difference': -15.0, 'direction': 'less'}]
        }
    }
    
    # Mock the helper methods
    analysis._visualize_resolution_time_distribution = MagicMock()
    analysis._visualize_feature_correlations = MagicMock()
    analysis._visualize_model_results = MagicMock()
    analysis._visualize_recommendations = MagicMock()
    
    # Run the visualization
    analysis.visualize(sample_issues)
    
    # Check that the helper methods were called
    analysis._visualize_resolution_time_distribution.assert_called_once()
    analysis._visualize_feature_correlations.assert_called_once()
    analysis._visualize_model_results.assert_called_once()
    analysis._visualize_recommendations.assert_called_once()

def test_visualize_resolution_time_distribution(monkeypatch):
    """
    Test the _visualize_resolution_time_distribution method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results
    analysis.results = {
        'resolution_distribution': {
            'resolution_times': [1.0, 3.0, 5.0, 7.0, 9.0],
            'mean_time': 5.0,
            'median_time': 5.0,
            'min_time': 1.0,
            'max_time': 9.0,
            'quick_resolution_pct': 60.0
        }
    }
    
    # Mock the create_histogram and create_pie_chart functions
    mock_create_histogram = MagicMock(return_value=plt.figure())
    mock_create_pie_chart = MagicMock(return_value=plt.figure())
    monkeypatch.setattr("src.analysis.resolution_prediction.create_histogram", mock_create_histogram)
    monkeypatch.setattr("src.analysis.resolution_prediction.create_pie_chart", mock_create_pie_chart)
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_resolution_time_distribution()
    
    # Check that create_histogram and create_pie_chart were called
    mock_create_histogram.assert_called_once()
    mock_create_pie_chart.assert_called_once()
    
    # Check that save_figure was called twice
    assert analysis.save_figure.call_count == 2
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "resolution_time" in filenames
    assert "quick_vs_slow_issue_resolution" in filenames

def test_visualize_feature_correlations(monkeypatch):
    """
    Test the _visualize_feature_correlations method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results
    analysis.results = {
        'issue_features': pd.DataFrame({
            'title_length': [10, 15, 20],
            'description_length': [100, 150, 200],
            'comment_count': [1, 2, 3],
            'participant_count': [1, 2, 3],
            'resolution_time': [1.0, 3.0, 5.0]
        }),
        'feature_correlations': {
            'correlation_matrix': {
                'title_length': {'resolution_time': 0.5},
                'description_length': {'resolution_time': 0.6},
                'comment_count': {'resolution_time': 0.8},
                'participant_count': {'resolution_time': 0.7},
                'resolution_time': {
                    'title_length': 0.5,
                    'description_length': 0.6,
                    'comment_count': 0.8,
                    'participant_count': 0.7,
                    'resolution_time': 1.0
                }
            },
            'resolution_correlations': {
                'title_length': 0.5,
                'description_length': 0.6,
                'comment_count': 0.8,
                'participant_count': 0.7
            },
            'sorted_correlations': [
                ('comment_count', 0.8),
                ('participant_count', 0.7),
                ('description_length', 0.6),
                ('title_length', 0.5)
            ]
        }
    }
    
    # Mock plt.figure and sns.heatmap
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("seaborn.heatmap", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.barh", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.scatter", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.plot", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.subplot", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.axvline", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    monkeypatch.setattr("numpy.polyfit", MagicMock(return_value=[1, 0]))
    monkeypatch.setattr("numpy.poly1d", MagicMock(return_value=lambda x: x))
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_feature_correlations()
    
    # Check that save_figure was called three times
    assert analysis.save_figure.call_count == 3
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "issue_feature_correlations" in filenames
    assert "resolution_time_features" in filenames
    assert "issue_resolution_times" in filenames

def test_visualize_model_results(monkeypatch):
    """
    Test the _visualize_model_results method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results with a successful model
    analysis.results = {
        'model_results': {
            'success': True,
            'confusion_matrix': [[5, 2], [1, 7]],
            'sorted_importance': [
                ('comment_count', 0.5),
                ('participant_count', 0.3),
                ('description_length', 0.2)
            ]
        }
    }
    
    # Mock plt.figure and sns.heatmap
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("seaborn.heatmap", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.barh", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.ylabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_model_results()
    
    # Check that save_figure was called twice
    assert analysis.save_figure.call_count == 2
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "confusion_matrix" in filenames
    assert "feature_importance" in filenames

def test_visualize_model_results_no_model(monkeypatch):
    """
    Test the _visualize_model_results method when model building failed.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results with a failed model
    analysis.results = {
        'model_results': {
            'success': False,
            'message': "Not enough data to build a predictive model"
        }
    }
    
    # Mock print to check it's called
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_model_results()
    
    # Check that save_figure was not called
    assert analysis.save_figure.call_count == 0
    
    # Check that print was called with the expected message
    mock_print.assert_called_with("Model wasn't successfully built, skipping visualizations")

def test_visualize_recommendations(monkeypatch):
    """
    Test the _visualize_recommendations method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results
    analysis.results = {
        'recommendations': {
            'recommendations': [
                {'recommendation': 'Add more comments', 'importance': 0.5},
                {'recommendation': 'Use appropriate labels', 'importance': 0.4},
                {'recommendation': 'Include code examples', 'importance': 0.3}
            ]
        }
    }
    
    # Mock plt.figure and plt.barh
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("matplotlib.pyplot.barh", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.xlabel", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.grid", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_recommendations()
    
    # Check that save_figure was called
    analysis.save_figure.assert_called_once()
    
    # Check the filename
    args, _ = analysis.save_figure.call_args
    assert args[1] == "recommendations"

def test_save_results(monkeypatch, tmpdir):
    """
    Test the save_results method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results
    analysis.results = {
        'issue_features': pd.DataFrame({
            'issue_number': ['1', '2'],
            'title_length': [10, 15],
            'resolution_time': [5.0, 10.0]
        }),
        'resolution_distribution': {
            'resolution_times': [5.0, 10.0],
            'mean_time': 7.5
        },
        'feature_correlations': {
            'correlation_matrix': {'resolution_time': {'title_length': 0.8}}
        },
        'model_results': {
            'success': True,
            'accuracy': 0.8,
            'feature_importance': {'title_length': 0.5}
        },
        'recommendations': {
            'recommendations': [{'recommendation': 'Add more comments', 'importance': 0.5}]
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
    assert analysis.save_json.call_count == 5
    
    # Check that _generate_report was called
    analysis._generate_report.assert_called_once()
    
    # Check that the DataFrame was converted to a serializable format
    serializable_features = analysis.save_json.call_args_list[0][0][0]
    assert isinstance(serializable_features, list)
    assert len(serializable_features) == 2
    
    # Check that model results were filtered to remove non-serializable items
    model_results = analysis.save_json.call_args_list[3][0][0]
    assert 'model' not in model_results
    assert 'X_train' not in model_results
    assert 'X_test' not in model_results
    assert 'y_train' not in model_results
    assert 'y_test' not in model_results

def test_generate_report(monkeypatch):
    """
    Test the _generate_report method.
    """
    # Create an IssueResolutionPrediction instance
    analysis = IssueResolutionPrediction()
    
    # Set up the results
    analysis.results = {
        'resolution_distribution': {
            'resolution_times': [5.0, 10.0],
            'mean_time': 7.5,
            'median_time': 7.5,
            'min_time': 5.0,
            'max_time': 10.0,
            'quick_resolution_pct': 50.0
        },
        'feature_correlations': {
            'sorted_correlations': [('comment_count', 0.8), ('participant_count', 0.6)],
            'categorical_impact': {
                'has_bug_label': {
                    'avg_time_true': 5.0,
                    'avg_time_false': 10.0,
                    'difference': -5.0,
                    'difference_pct': -50.0
                }
            }
        },
        'model_results': {
            'success': True,
            'accuracy': 0.8,
            'classification_report': {
                'Slow Resolution': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75, 'support': 10},
                'Quick Resolution': {'precision': 0.7, 'recall': 0.8, 'f1-score': 0.75, 'support': 10}
            },
            'sorted_importance': [('comment_count', 0.5), ('participant_count', 0.3)]
        },
        'recommendations': {
            'recommendations': [
                {'recommendation': 'Add more comments', 'importance': 0.5},
                {'recommendation': 'Use appropriate labels', 'importance': 0.4}
            ],
            'significant_factors': [
                {'factor': 'comment_count', 'difference': 20.0, 'direction': 'higher'}
            ],
            'categorical_factors': [
                {'factor': 'has_bug_label', 'difference': -15.0, 'direction': 'less'}
            ]
        }
    }
    
    # Mock the Report class
    mock_report = MagicMock()
    mock_report_class = MagicMock(return_value=mock_report)
    monkeypatch.setattr("src.analysis.resolution_prediction.Report", mock_report_class)
    
    # Run the _generate_report method
    analysis._generate_report()
    
    # Check that Report was initialized correctly
    mock_report_class.assert_called_once_with("Issue Resolution Prediction Report", analysis.results_dir)
    
    # Check that add_section was called for each section
    assert mock_report.add_section.call_count >= 4  # At least 4 sections
    
    # Check that save_text_report and save_markdown_report were called
    mock_report.save_text_report.assert_called_once_with("result.txt")
    mock_report.save_markdown_report.assert_called_once_with("report.md")


def test_build_predictive_model_exception(monkeypatch):
    """
    Test _build_predictive_model when RandomForestClassifier.fit throws an exception.
    """
    from src.analysis.resolution_prediction import IssueResolutionPrediction
    import pandas as pd
    import numpy as np

    analysis = IssueResolutionPrediction()

    # Minimal DataFrame to reach model training
    df = pd.DataFrame({
        'title_length': np.random.randint(10, 100, 20),
        'description_length': np.random.randint(100, 1000, 20),
        'label_count': np.random.randint(1, 5, 20),
        'has_bug_label': [True] * 10 + [False] * 10,
        'has_feature_label': [False] * 20,
        'has_enhancement_label': [False] * 20,
        'comment_count': np.random.randint(0, 5, 20),
        'participant_count': np.random.randint(1, 3, 20),
        'has_code': [False] * 20,
        'resolution_time': np.random.uniform(1, 10, 20),
        'is_quick_resolution': np.random.choice([True, False], 20)
    })

    # Force RandomForestClassifier.fit to raise an exception
    class FakeClassifier:
        def fit(self, X, y):
            raise ValueError("Training failed!")

    monkeypatch.setattr("src.analysis.resolution_prediction.RandomForestClassifier", lambda *args, **kwargs: FakeClassifier())
    monkeypatch.setattr("src.analysis.resolution_prediction.StandardScaler", lambda: type('FakeScaler', (), {'fit_transform': lambda self, X: X, 'transform': lambda self, X: X})())

    result = analysis._build_predictive_model(df)

    assert result['success'] == False
    assert "Error building predictive model" in result['message']

def test_generate_recommendations_no_quick_or_slow(monkeypatch):
    """
    Test _generate_recommendations when there are only slow-resolved issues.
    """
    from src.analysis.resolution_prediction import IssueResolutionPrediction
    import pandas as pd

    analysis = IssueResolutionPrediction()

    # Only slow issues (is_quick_resolution == False)
    df = pd.DataFrame({
        'title_length': [10, 20],
        'description_length': [100, 200],
        'label_count': [1, 2],
        'comment_count': [0, 1],
        'participant_count': [1, 2],
        'has_bug_label': [False, False],
        'has_feature_label': [False, False],
        'has_enhancement_label': [False, False],
        'has_code': [False, False],
        'resolution_time': [10.0, 15.0],
        'is_quick_resolution': [False, False]  # <=== Only slow
    })

    model_results = {
        'success': False
    }

    recommendations = analysis._generate_recommendations(df, model_results)

    assert 'recommendations' in recommendations
    assert 'significant_factors' in recommendations
    assert 'categorical_factors' in recommendations
def test_analyze_feature_correlations_only_true():
    """
    Test _analyze_feature_correlations when a categorical feature has only True values.
    """
    from src.analysis.resolution_prediction import IssueResolutionPrediction
    import pandas as pd

    analysis = IssueResolutionPrediction()

    # Create a DataFrame where 'has_bug_label' is always True
    df = pd.DataFrame({
        'title_length': [10, 20, 30],
        'description_length': [100, 200, 300],
        'label_count': [1, 2, 3],
        'comment_count': [1, 2, 3],
        'participant_count': [1, 2, 3],
        'has_bug_label': [True, True, True],
        'has_feature_label': [False, False, False],
        'has_enhancement_label': [False, False, False],
        'has_code': [True, False, True],
        'resolution_time': [5.0, 10.0, 15.0]
    })

    results = analysis._analyze_feature_correlations(df)

    assert 'correlation_matrix' in results
    assert 'resolution_correlations' in results
    assert 'sorted_correlations' in results
    assert 'categorical_impact' in results

    # Check that 'has_bug_label' is either missing or partial
    assert 'has_bug_label' not in results['categorical_impact']
def test_prepare_features_negative_resolution_time_and_code(monkeypatch):
    """
    Test _prepare_features skips issues with negative resolution times and correctly detects code blocks.
    """
    from src.analysis.resolution_prediction import IssueResolutionPrediction
    from src.core.model import Issue, State
    import pandas as pd

    analysis = IssueResolutionPrediction()

    # Issue with negative resolution time (should be skipped)
    bad_issue = Issue({
        'number': '10',
        'state': 'closed',
        'title': 'Broken issue',
        'text': 'Bad issue',
        'labels': [],
        'creator': 'user',
        'created_date': '2023-01-01T00:00:00Z',
        'updated_date': '2023-01-02T00:00:00Z',
        'timeline_url': '',
        'events': []
    })

    # Issue with a code block
    good_issue = Issue({
        'number': '11',
        'state': 'closed',
        'title': 'Issue with code',
        'text': '```python\nprint("Hello World")\n```',
        'labels': [],
        'creator': 'user',
        'created_date': '2023-01-01T00:00:00Z',
        'updated_date': '2023-01-03T00:00:00Z',
        'timeline_url': '',
        'events': []
    })

    # Monkeypatch helpers
    def mock_get_issue_resolution_time(issue):
        return -5.0 if issue.number == 10 else 2.0

    def mock_has_code_blocks(text):
        return '```' in text

    monkeypatch.setattr("src.analysis.resolution_prediction.get_issue_resolution_time", mock_get_issue_resolution_time)
    monkeypatch.setattr("src.analysis.resolution_prediction.get_issue_comment_count", lambda issue: 1)
    monkeypatch.setattr("src.analysis.resolution_prediction.get_issue_unique_participants", lambda issue: ['user'])
    monkeypatch.setattr("src.analysis.resolution_prediction.has_code_blocks", mock_has_code_blocks)

    df = analysis._prepare_features([bad_issue, good_issue])

    # Only good_issue should appear
    assert len(df) == 1
    assert df.iloc[0]['issue_number'] == 11
    assert df.iloc[0]['has_code'] == True
def test_generate_recommendations_no_issues(monkeypatch):
    """
    Test _generate_recommendations when there are no issues.
    """
    from src.analysis.resolution_prediction import IssueResolutionPrediction
    import pandas as pd

    analysis = IssueResolutionPrediction()

    df = pd.DataFrame(columns=[
        'title_length', 'description_length', 'label_count',
        'comment_count', 'participant_count', 'has_bug_label',
        'has_feature_label', 'has_enhancement_label', 'has_code',
        'resolution_time', 'is_quick_resolution'
    ])

    model_results = {'success': False}

    recommendations = analysis._generate_recommendations(df, model_results)

    assert 'recommendations' in recommendations
    assert 'significant_factors' in recommendations
    assert 'categorical_factors' in recommendations

def test_generate_report_empty_recommendations(monkeypatch, tmp_path):
    """
    Test _generate_report with empty recommendations.
    """
    from src.analysis.resolution_prediction import IssueResolutionPrediction

    analysis = IssueResolutionPrediction()
    analysis.results_dir = tmp_path

    # Minimal results to trigger report generation
    analysis.results = {
        'resolution_distribution': {
            'resolution_times': [5.0],
            'mean_time': 5.0,
            'median_time': 5.0,
            'min_time': 5.0,
            'max_time': 5.0,
            'quick_resolution_pct': 100.0
        },
        'feature_correlations': {
            'sorted_correlations': [],
            'categorical_impact': {}
        },
        'model_results': {
            'success': False,
            'message': "Not enough data"
        },
        'recommendations': {
            'recommendations': [],
            'significant_factors': [],
            'categorical_factors': []
        }
    }

    # Mock Report class to avoid writing files
    monkeypatch.setattr("src.analysis.resolution_prediction.Report", lambda title, outdir: type('FakeReport', (), {
        'add_section': lambda self, title, content: None,
        'save_text_report': lambda self, filename: None,
        'save_markdown_report': lambda self, filename: None
    })())

    analysis._generate_report()

    # If no exception happens, test passes
    assert True
