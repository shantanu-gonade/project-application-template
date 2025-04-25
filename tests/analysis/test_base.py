"""
Tests for the base analysis module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from src.analysis.base import BaseAnalysis
from src.core.model import Issue, State

class MockAnalysis(BaseAnalysis):
    """
    Mock implementation of BaseAnalysis for testing.
    """
    
    def __init__(self, name="test_analysis"):
        super().__init__(name)
        self.results = {}
    
    def analyze(self, issues):
        self.results = {"count": len(issues)}
        return self.results
    
    def visualize(self, issues):
        pass
    
    def save_results(self):
        self.save_json(self.results, "test_results")

def test_base_analysis_initialization(mock_config):
    """
    Test that BaseAnalysis is correctly initialized.
    """
    analysis = MockAnalysis()
    
    assert analysis.name == "test_analysis"
    assert analysis.label is None
    assert analysis.user is None
    assert "results/test_analysis" in analysis.results_dir
    assert "results/test_analysis/images" in analysis.images_dir

def test_base_analysis_initialization_with_label(monkeypatch):
    """
    Test that BaseAnalysis is correctly initialized with a label.
    """
    # Mock the config.get_parameter function to return a label
    def mock_get_parameter(param_name, default=None):
        if param_name == "label":
            return "bug"
        return None
    
    monkeypatch.setattr("src.core.config.get_parameter", mock_get_parameter)
    
    analysis = MockAnalysis()
    
    assert analysis.name == "test_analysis"
    assert analysis.label == "bug"
    assert analysis.user is None

def test_base_analysis_initialization_with_user(monkeypatch):
    """
    Test that BaseAnalysis is correctly initialized with a user.
    """
    # Mock the config.get_parameter function to return a user
    def mock_get_parameter(param_name, default=None):
        if param_name == "user":
            return "test_user"
        return None
    
    monkeypatch.setattr("src.core.config.get_parameter", mock_get_parameter)
    
    analysis = MockAnalysis()
    
    assert analysis.name == "test_analysis"
    assert analysis.label is None
    assert analysis.user == "test_user"

def test_load_data(sample_issues, monkeypatch):
    """
    Test that load_data returns all issues when no filter is applied.
    """
    # Create a mock DataLoader that returns sample_issues
    class MockDataLoader:
        def get_issues(self):
            return sample_issues
    
    # Patch the DataLoader class
    monkeypatch.setattr("src.analysis.base.DataLoader", MockDataLoader)
    
    analysis = MockAnalysis()
    issues = analysis.load_data()
    
    assert len(issues) == 4
    assert issues == sample_issues

def test_load_data_with_label(sample_issues, monkeypatch):
    """
    Test that load_data filters issues by label when a label is specified.
    """
    # Create a mock DataLoader that returns sample_issues
    class MockDataLoader:
        def get_issues(self):
            return sample_issues
    
    # Patch the DataLoader class
    monkeypatch.setattr("src.analysis.base.DataLoader", MockDataLoader)
    
    # Mock the config.get_parameter function to return a label
    def mock_get_parameter(param_name, default=None):
        if param_name == "label":
            return "bug"
        return None
    
    monkeypatch.setattr("src.core.config.get_parameter", mock_get_parameter)
    
    analysis = MockAnalysis()
    issues = analysis.load_data()
    
    # Only issues 1 and 4 have the "bug" label
    assert len(issues) == 2
    assert issues[0].number == 1
    assert issues[1].number == 4

def test_load_data_with_user(sample_issues, monkeypatch):
    """
    Test that load_data filters issues by user when a user is specified.
    """
    # Create a mock DataLoader that returns sample_issues
    class MockDataLoader:
        def get_issues(self):
            return sample_issues
    
    # Patch the DataLoader class
    monkeypatch.setattr("src.analysis.base.DataLoader", MockDataLoader)
    
    # Mock the config.get_parameter function to return a user
    def mock_get_parameter(param_name, default=None):
        if param_name == "user":
            return "user1"
        return None
    
    monkeypatch.setattr("src.core.config.get_parameter", mock_get_parameter)
    
    analysis = MockAnalysis()
    issues = analysis.load_data()
    
    # Only issue 1 has creator "user1"
    assert len(issues) == 1
    assert issues[0].number == 1

def test_run(mock_data_loader, sample_issues):
    """
    Test that run calls the appropriate methods.
    """
    analysis = MockAnalysis()
    
    # Mock the methods
    analysis.load_data = MagicMock(return_value=sample_issues)
    analysis.analyze = MagicMock(return_value={"count": 4})
    analysis.visualize = MagicMock()
    analysis.save_results = MagicMock()
    
    # Run the analysis
    analysis.run()
    
    # Check that the methods were called
    analysis.load_data.assert_called_once()
    analysis.analyze.assert_called_once_with(sample_issues)
    analysis.visualize.assert_called_once_with(sample_issues)
    analysis.save_results.assert_called_once()

def test_save_figure(temp_results_dir):
    """
    Test that save_figure correctly saves a figure.
    """
    # Create a test analysis with a temporary results directory
    analysis = MockAnalysis()
    analysis.results_dir = str(temp_results_dir)
    analysis.images_dir = str(temp_results_dir.join("images"))
    
    # Create a mock figure
    fig = MagicMock()
    
    # Save the figure
    analysis.save_figure(fig, "test_figure")
    
    # Check that the figure was saved
    fig.savefig.assert_called_once_with(os.path.join(analysis.images_dir, "test_figure.png"))

def test_save_json(temp_results_dir):
    """
    Test that save_json correctly saves JSON data.
    """
    # Create a test analysis with a temporary results directory
    analysis = MockAnalysis()
    analysis.results_dir = str(temp_results_dir)
    
    # Save JSON data
    data = {"key": "value"}
    analysis.save_json(data, "test_data")
    
    # Check that the data was saved
    filepath = os.path.join(analysis.results_dir, "test_data.json")
    assert os.path.exists(filepath)
    
    # Check the contents of the file
    with open(filepath, 'r') as f:
        saved_data = json.load(f)
    
    assert saved_data == data

def test_save_text(temp_results_dir):
    """
    Test that save_text correctly saves text data.
    """
    # Create a test analysis with a temporary results directory
    analysis = MockAnalysis()
    analysis.results_dir = str(temp_results_dir)
    
    # Save text data
    text = "This is a test"
    analysis.save_text(text, "test_text")
    
    # Check that the text was saved
    filepath = os.path.join(analysis.results_dir, "test_text.txt")
    assert os.path.exists(filepath)
    
    # Check the contents of the file
    with open(filepath, 'r') as f:
        saved_text = f.read()
    
    assert saved_text == text

def test_analyze_not_implemented():
    """
    Test that the analyze method raises NotImplementedError if not overridden.
    """
    class IncompleteAnalysis(BaseAnalysis):
        def __init__(self):
            super().__init__("incomplete")
        
        def visualize(self, issues):
            pass
        
        def save_results(self):
            pass
    
    analysis = IncompleteAnalysis()
    
    with pytest.raises(NotImplementedError):
        analysis.analyze([])

def test_visualize_not_implemented():
    """
    Test that the visualize method raises NotImplementedError if not overridden.
    """
    class IncompleteAnalysis(BaseAnalysis):
        def __init__(self):
            super().__init__("incomplete")
        
        def analyze(self, issues):
            return {}
        
        def save_results(self):
            pass
    
    analysis = IncompleteAnalysis()
    
    with pytest.raises(NotImplementedError):
        analysis.visualize([])

def test_save_results_not_implemented():
    """
    Test that the save_results method raises NotImplementedError if not overridden.
    """
    class IncompleteAnalysis(BaseAnalysis):
        def __init__(self):
            super().__init__("incomplete")
        
        def analyze(self, issues):
            return {}
        
        def visualize(self, issues):
            pass
    
    analysis = IncompleteAnalysis()
    
    with pytest.raises(NotImplementedError):
        analysis.save_results()
