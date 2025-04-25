"""
Tests for the model module.
"""

import pytest
from datetime import datetime
from dateutil import parser

from src.core.model import Issue, Event, State

def test_event_initialization(sample_event_data):
    """
    Test that an Event object is correctly initialized from JSON data.
    """
    event = Event(sample_event_data)
    assert event.event_type == "commented"
    assert event.author == "test_user"
    assert event.event_date.isoformat() == "2023-01-01T12:00:00+00:00"
    assert event.label == "bug"
    assert event.comment == "Test comment"

def test_event_initialization_empty():
    """
    Test that an Event object can be initialized with no data.
    """
    event = Event(None)
    assert event.event_type is None
    assert event.author is None
    assert event.event_date is None
    assert event.label is None
    assert event.comment is None

def test_event_initialization_missing_fields():
    """
    Test that an Event object handles missing fields gracefully.
    """
    event_data = {
        "event_type": "commented",
        # Missing author
        "event_date": "2023-01-01T12:00:00Z",
        # Missing label
        "comment": "Test comment"
    }
    event = Event(event_data)
    assert event.event_type == "commented"
    assert event.author is None
    assert event.event_date.isoformat() == "2023-01-01T12:00:00+00:00"
    assert event.label is None
    assert event.comment == "Test comment"

def test_event_initialization_invalid_date():
    """
    Test that an Event object handles invalid date formats gracefully.
    """
    event_data = {
        "event_type": "commented",
        "author": "test_user",
        "event_date": "invalid-date",
        "label": "bug",
        "comment": "Test comment"
    }
    event = Event(event_data)
    assert event.event_type == "commented"
    assert event.author == "test_user"
    assert event.event_date is None
    assert event.label == "bug"
    assert event.comment == "Test comment"

def test_issue_initialization(sample_issue_data):
    """
    Test that an Issue object is correctly initialized from JSON data.
    """
    issue = Issue(sample_issue_data)
    assert issue.url == "https://github.com/test/test/issues/1"
    assert issue.creator == "test_creator"
    assert issue.labels == ["bug", "documentation"]
    assert issue.state == State.open
    assert issue.assignees == ["assignee1"]
    assert issue.title == "Test Issue"
    assert issue.text == "This is a test issue"
    assert issue.number == 1
    assert issue.created_date.isoformat() == "2023-01-01T10:00:00+00:00"
    assert issue.updated_date.isoformat() == "2023-01-02T10:00:00+00:00"
    assert issue.timeline_url == "https://github.com/test/test/issues/1/timeline"
    assert len(issue.events) == 2
    assert issue.events[0].event_type == "commented"
    assert issue.events[1].event_type == "labeled"

def test_issue_initialization_empty():
    """
    Test that an Issue object can be initialized with no data.
    """
    issue = Issue(None)
    assert issue.url is None
    assert issue.creator is None
    assert issue.labels == []
    assert issue.state is None
    assert issue.assignees == []
    assert issue.title is None
    assert issue.text is None
    assert issue.number == -1
    assert issue.created_date is None
    assert issue.updated_date is None
    assert issue.timeline_url is None
    assert issue.events == []

def test_issue_initialization_missing_fields():
    """
    Test that an Issue object handles missing fields gracefully.
    """
    issue_data = {
        "url": "https://github.com/test/test/issues/1",
        # Missing creator
        "labels": ["bug"],
        "state": "open",
        # Missing assignees
        "title": "Test Issue",
        # Missing text
        "number": "1",
        "created_date": "2023-01-01T10:00:00Z",
        # Missing updated_date
        # Missing timeline_url
        # Missing events
    }
    issue = Issue(issue_data)
    assert issue.url == "https://github.com/test/test/issues/1"
    assert issue.creator is None
    assert issue.labels == ["bug"]
    assert issue.state == State.open
    assert issue.assignees == []
    assert issue.title == "Test Issue"
    assert issue.text is None
    assert issue.number == 1
    assert issue.created_date.isoformat() == "2023-01-01T10:00:00+00:00"
    assert issue.updated_date is None
    assert issue.timeline_url is None
    assert issue.events == []

def test_issue_initialization_invalid_number():
    """
    Test that an Issue object handles invalid number formats gracefully.
    """
    issue_data = {
        "url": "https://github.com/test/test/issues/1",
        "creator": "test_creator",
        "labels": ["bug"],
        "state": "open",
        "title": "Test Issue",
        "number": "not-a-number",
        "created_date": "2023-01-01T10:00:00Z"
    }
    issue = Issue(issue_data)
    assert issue.number == -1

def test_issue_initialization_invalid_dates():
    """
    Test that an Issue object handles invalid date formats gracefully.
    """
    issue_data = {
        "url": "https://github.com/test/test/issues/1",
        "creator": "test_creator",
        "labels": ["bug"],
        "state": "open",
        "title": "Test Issue",
        "number": "1",
        "created_date": "invalid-date",
        "updated_date": "also-invalid"
    }
    issue = Issue(issue_data)
    assert issue.created_date is None
    assert issue.updated_date is None

def test_state_enum():
    """
    Test the State enum values.
    """
    assert State.open == "open"
    assert State.closed == "closed"
    assert State["open"] == State.open
    assert State["closed"] == State.closed
