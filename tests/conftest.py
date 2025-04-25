"""
Shared fixtures for pytest tests.
"""

import os
import pytest
from datetime import datetime
from dateutil import parser

from src.core.model import Issue, Event, State

@pytest.fixture
def sample_event_data():
    """
    Returns sample event data for testing.
    """
    return {
        "event_type": "commented",
        "author": "test_user",
        "event_date": "2023-01-01T12:00:00Z",
        "label": "bug",
        "comment": "Test comment"
    }

@pytest.fixture
def sample_issue_data():
    """
    Returns sample issue data for testing.
    """
    return {
        "url": "https://github.com/test/test/issues/1",
        "creator": "test_creator",
        "labels": ["bug", "documentation"],
        "state": "open",
        "assignees": ["assignee1"],
        "title": "Test Issue",
        "text": "This is a test issue",
        "number": "1",
        "created_date": "2023-01-01T10:00:00Z",
        "updated_date": "2023-01-02T10:00:00Z",
        "timeline_url": "https://github.com/test/test/issues/1/timeline",
        "events": [
            {
                "event_type": "commented",
                "author": "commenter1",
                "event_date": "2023-01-01T11:00:00Z",
                "comment": "First comment"
            },
            {
                "event_type": "labeled",
                "author": "labeler1",
                "event_date": "2023-01-01T12:00:00Z",
                "label": "bug"
            }
        ]
    }

@pytest.fixture
def sample_issues():
    """
    Returns a list of sample issues for testing analysis modules.
    """
    # Create a closed issue with comments and resolution
    closed_issue = Issue({
        "url": "https://github.com/test/test/issues/1",
        "creator": "user1",
        "labels": ["bug", "documentation"],
        "state": "closed",
        "title": "Bug in documentation",
        "number": "1",
        "created_date": "2023-01-01T10:00:00Z",
        "updated_date": "2023-01-10T10:00:00Z",
        "events": [
            {
                "event_type": "commented",
                "author": "user2",
                "event_date": "2023-01-02T10:00:00Z",
                "comment": "I can reproduce this"
            },
            {
                "event_type": "commented",
                "author": "user3",
                "event_date": "2023-01-03T10:00:00Z",
                "comment": "Working on a fix"
            },
            {
                "event_type": "closed",
                "author": "user1",
                "event_date": "2023-01-10T10:00:00Z"
            }
        ]
    })
    
    # Create an open issue with multiple labels
    open_issue = Issue({
        "url": "https://github.com/test/test/issues/2",
        "creator": "user2",
        "labels": ["enhancement", "good first issue"],
        "state": "open",
        "title": "Add new feature",
        "number": "2",
        "created_date": "2023-01-05T10:00:00Z",
        "updated_date": "2023-01-06T10:00:00Z",
        "events": [
            {
                "event_type": "commented",
                "author": "user1",
                "event_date": "2023-01-06T10:00:00Z",
                "comment": "This would be useful"
            }
        ]
    })
    
    # Create an issue with no labels
    no_labels_issue = Issue({
        "url": "https://github.com/test/test/issues/3",
        "creator": "user3",
        "labels": [],
        "state": "open",
        "title": "Question about usage",
        "number": "3",
        "created_date": "2023-01-07T10:00:00Z",
        "updated_date": "2023-01-07T10:00:00Z",
        "events": []
    })
    
    # Create an issue with a different creator
    different_creator_issue = Issue({
        "url": "https://github.com/test/test/issues/4",
        "creator": "user4",
        "labels": ["bug", "high-priority"],
        "state": "closed",
        "title": "Critical bug",
        "number": "4",
        "created_date": "2023-01-08T10:00:00Z",
        "updated_date": "2023-01-09T10:00:00Z",
        "events": [
            {
                "event_type": "commented",
                "author": "user1",
                "event_date": "2023-01-08T11:00:00Z",
                "comment": "This is indeed critical"
            },
            {
                "event_type": "closed",
                "author": "user4",
                "event_date": "2023-01-09T10:00:00Z"
            }
        ]
    })
    
    return [closed_issue, open_issue, no_labels_issue, different_creator_issue]

@pytest.fixture
def mock_data_loader(monkeypatch, sample_issues):
    """
    Mocks the DataLoader to return sample issues.
    """
    class MockDataLoader:
        def __init__(self):
            self.data_path = "mock_path"
            
        def get_issues(self):
            return sample_issues
    
    monkeypatch.setattr("src.core.data_loader.DataLoader", MockDataLoader)
    return MockDataLoader()

@pytest.fixture
def temp_results_dir(tmpdir):
    """
    Creates a temporary directory for test results.
    """
    results_dir = tmpdir.mkdir("results")
    images_dir = results_dir.mkdir("images")
    return results_dir

@pytest.fixture
def mock_config(monkeypatch):
    """
    Mocks the config module.
    """
    def mock_get_parameter(param_name, default=None):
        config_values = {
            "ENPM611_PROJECT_DATA_PATH": "mock_data_path",
            "label": None,
            "user": None
        }
        return config_values.get(param_name, default)
    
    monkeypatch.setattr("src.core.config.get_parameter", mock_get_parameter)
    return mock_get_parameter
