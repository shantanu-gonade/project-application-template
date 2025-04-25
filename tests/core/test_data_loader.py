"""
Tests for the data_loader module.
"""

import os
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from src.core.data_loader import DataLoader
from src.core.model import Issue

def test_data_loader_initialization(monkeypatch):
    """
    Test that DataLoader is correctly initialized.
    """
    # Mock the config.get_parameter function
    monkeypatch.setattr("src.core.config.get_parameter", lambda param, default=None: "test_path.json")
    
    # Create a DataLoader instance
    loader = DataLoader()
    
    # Check that the data_path is set correctly
    assert loader.data_path == "test_path.json"

def test_get_issues_singleton(monkeypatch):
    """
    Test that get_issues returns the same issues on subsequent calls (singleton pattern).
    """
    # Create a temporary JSON file with test data
    test_data = [
        {
            "url": "https://github.com/test/test/issues/1",
            "creator": "test_user",
            "labels": ["bug"],
            "state": "open",
            "title": "Test Issue",
            "number": "1",
            "created_date": "2023-01-01T10:00:00Z"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(test_data, temp_file)
    
    try:
        # Mock the config.get_parameter function
        monkeypatch.setattr("src.core.config.get_parameter", lambda param, default=None: temp_file.name)
        
        # Reset the singleton
        import src.core.data_loader
        src.core.data_loader._ISSUES = None
        
        # Create a DataLoader instance
        loader = DataLoader()
        
        # Get issues for the first time
        issues1 = loader.get_issues()
        
        # Check that issues were loaded correctly
        assert len(issues1) == 1
        assert issues1[0].title == "Test Issue"
        
        # Get issues for the second time
        issues2 = loader.get_issues()
        
        # Check that the same object is returned
        assert issues1 is issues2
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)

def test_load_small_file(monkeypatch):
    """
    Test loading a small JSON file.
    """
    # Create a temporary JSON file with test data
    test_data = [
        {
            "url": "https://github.com/test/test/issues/1",
            "creator": "test_user",
            "labels": ["bug"],
            "state": "open",
            "title": "Test Issue 1",
            "number": "1",
            "created_date": "2023-01-01T10:00:00Z"
        },
        {
            "url": "https://github.com/test/test/issues/2",
            "creator": "test_user",
            "labels": ["enhancement"],
            "state": "closed",
            "title": "Test Issue 2",
            "number": "2",
            "created_date": "2023-01-02T10:00:00Z"
        }
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        json.dump(test_data, temp_file)
    
    try:
        # Mock the config.get_parameter function
        monkeypatch.setattr("src.core.config.get_parameter", lambda param, default=None: temp_file.name)
        
        # Reset the singleton
        import src.core.data_loader
        src.core.data_loader._ISSUES = None
        
        # Create a DataLoader instance
        loader = DataLoader()
        
        # Get issues
        issues = loader.get_issues()
        
        # Check that issues were loaded correctly
        assert len(issues) == 2
        assert issues[0].title == "Test Issue 1"
        assert issues[1].title == "Test Issue 2"
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)

def test_load_large_file(monkeypatch):
    """
    Test loading a large JSON file using chunked reading.
    """
    # Mock os.path.getsize to return a large file size
    monkeypatch.setattr("os.path.getsize", lambda path: 20 * 1024 * 1024)  # 20MB
    
    # Create a mock file object that simulates a large JSON file
    class MockFile:
        def __init__(self):
            self.read_count = 0
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def read(self, size=None):
            if self.read_count == 0:
                self.read_count += 1
                return '['
            elif self.read_count == 1:
                self.read_count += 1
                return '{"url": "https://github.com/test/test/issues/1", "creator": "test_user", "labels": ["bug"], "state": "open", "title": "Test Issue 1", "number": "1", "created_date": "2023-01-01T10:00:00Z"}'
            elif self.read_count == 2:
                self.read_count += 1
                return ','
            elif self.read_count == 3:
                self.read_count += 1
                return '{"url": "https://github.com/test/test/issues/2", "creator": "test_user", "labels": ["enhancement"], "state": "closed", "title": "Test Issue 2", "number": "2", "created_date": "2023-01-02T10:00:00Z"}'
            else:
                return ''
    
    # Mock open to return our mock file
    monkeypatch.setattr("builtins.open", lambda path, mode: MockFile())
    
    # Mock the config.get_parameter function
    monkeypatch.setattr("src.core.config.get_parameter", lambda param, default=None: "large_file.json")
    
    # Reset the singleton
    import src.core.data_loader
    src.core.data_loader._ISSUES = None
    
    # Create a DataLoader instance
    loader = DataLoader()
    
    # Get issues
    issues = loader.get_issues()
    
    # Check that issues were loaded correctly
    assert len(issues) == 2
    assert issues[0].title == "Test Issue 1"
    assert issues[1].title == "Test Issue 2"

def test_load_invalid_json(monkeypatch):
    """
    Test handling of invalid JSON.
    """
    # Create a temporary file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write('{"invalid": "json"')  # Missing closing brace
    
    try:
        # Mock the config.get_parameter function
        monkeypatch.setattr("src.core.config.get_parameter", lambda param, default=None: temp_file.name)
        
        # Reset the singleton
        import src.core.data_loader
        src.core.data_loader._ISSUES = None
        
        # Create a DataLoader instance
        loader = DataLoader()
        
        # Attempt to get issues (should raise an exception)
        with pytest.raises(json.JSONDecodeError):
            loader.get_issues()
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)

def test_load_empty_file(monkeypatch):
    """
    Test loading an empty file.
    """
    # Create a temporary empty file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        pass
    
    try:
        # Mock the config.get_parameter function
        monkeypatch.setattr("src.core.config.get_parameter", lambda param, default=None: temp_file.name)
        
        # Reset the singleton
        import src.core.data_loader
        src.core.data_loader._ISSUES = None
        
        # Create a DataLoader instance
        loader = DataLoader()
        
        # Attempt to get issues (should raise an exception)
        with pytest.raises(Exception):
            loader.get_issues()
    finally:
        # Clean up the temporary file
        os.unlink(temp_file.name)
