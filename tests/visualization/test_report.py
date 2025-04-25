"""
Tests for the report module.
"""

import os
import json
import pytest
from datetime import datetime

from src.visualization.report import Report

def test_report_initialization():
    """
    Test that Report is correctly initialized.
    """
    report = Report("Test Report", "/tmp/results")
    
    assert report.name == "Test Report"
    assert report.results_dir == "/tmp/results"
    assert report.sections == []

def test_add_section():
    """
    Test adding a section to a report.
    """
    report = Report("Test Report", "/tmp/results")
    
    # Add a section
    report.add_section("Section 1", "This is section 1 content.")
    
    # Check that the section was added
    assert len(report.sections) == 1
    assert report.sections[0]["title"] == "Section 1"
    assert report.sections[0]["content"] == "This is section 1 content."
    
    # Add another section
    report.add_section("Section 2", "This is section 2 content.")
    
    # Check that both sections are present
    assert len(report.sections) == 2
    assert report.sections[0]["title"] == "Section 1"
    assert report.sections[1]["title"] == "Section 2"

def test_add_table():
    """
    Test adding a table to a report.
    """
    report = Report("Test Report", "/tmp/results")
    
    # Add a table
    headers = ["Name", "Value", "Description"]
    rows = [
        ["Item 1", "10", "First item"],
        ["Item 2", "20", "Second item"],
        ["Item 3", "30", "Third item"]
    ]
    report.add_table("Test Table", headers, rows)
    
    # Check that the table was added as a section
    assert len(report.sections) == 1
    assert report.sections[0]["title"] == "Test Table"
    
    # Check that the table content is formatted correctly
    content = report.sections[0]["content"]
    assert "Test Table" in content
    
    # The actual implementation formats the table with spaces between columns
    # and aligns the columns based on the widest cell in each column
    assert "Name | Value | Description" in content
    assert "Item 1 | 10    | First item" in content or "Item 1 | 10 | First item" in content
    assert "Item 2 | 20    | Second item" in content or "Item 2 | 20 | Second item" in content
    assert "Item 3 | 30    | Third item" in content or "Item 3 | 30 | Third item" in content

def test_add_key_value_section():
    """
    Test adding a key-value section to a report.
    """
    report = Report("Test Report", "/tmp/results")
    
    # Add a key-value section
    data = {
        "Key 1": "Value 1",
        "Longer Key 2": "Value 2",
        "Key 3": "Value 3"
    }
    report.add_key_value_section("Key-Value Section", data)
    
    # Check that the section was added
    assert len(report.sections) == 1
    assert report.sections[0]["title"] == "Key-Value Section"
    
    # Check that the content is formatted correctly
    content = report.sections[0]["content"]
    assert "Key-Value Section" in content
    
    # The actual implementation aligns the keys based on the longest key
    assert "Key 1       : Value 1" in content or "Key 1: Value 1" in content
    assert "Longer Key 2: Value 2" in content
    assert "Key 3       : Value 3" in content or "Key 3: Value 3" in content

def test_add_list_section():
    """
    Test adding a list section to a report.
    """
    report = Report("Test Report", "/tmp/results")
    
    # Add a list section
    items = ["Item 1", "Item 2", "Item 3"]
    report.add_list_section("List Section", items)
    
    # Check that the section was added
    assert len(report.sections) == 1
    assert report.sections[0]["title"] == "List Section"
    
    # Check that the content is formatted correctly
    content = report.sections[0]["content"]
    assert "List Section" in content
    assert "1. Item 1" in content
    assert "2. Item 2" in content
    assert "3. Item 3" in content

def test_generate_text_report():
    """
    Test generating a text report.
    """
    report = Report("Test Report", "/tmp/results")
    
    # Add sections
    report.add_section("Section 1", "This is section 1 content.")
    report.add_section("Section 2", "This is section 2 content.")
    
    # Generate the report
    text_report = report.generate_text_report()
    
    # Check the report content
    assert "# Test Report" in text_report
    assert "Generated:" in text_report
    assert "## Section 1" in text_report
    assert "This is section 1 content." in text_report
    assert "## Section 2" in text_report
    assert "This is section 2 content." in text_report

def test_save_text_report(tmpdir):
    """
    Test saving a text report to a file.
    """
    # Create a report with a temporary results directory
    results_dir = tmpdir.mkdir("results")
    report = Report("Test Report", str(results_dir))
    
    # Add a section
    report.add_section("Test Section", "This is a test section.")
    
    # Save the report
    filepath = report.save_text_report("test_report")
    
    # Check that the file was created
    assert os.path.exists(filepath)
    
    # Check the file content
    with open(filepath, 'r') as f:
        content = f.read()
    
    assert "# Test Report" in content
    assert "## Test Section" in content
    assert "This is a test section." in content

def test_generate_json_report():
    """
    Test generating a JSON report.
    """
    report = Report("Test Report", "/tmp/results")
    
    # Add sections
    report.add_section("Section 1", "This is section 1 content.")
    report.add_section("Section 2", "This is section 2 content.")
    
    # Generate the report
    json_report = report.generate_json_report()
    
    # Check the report content
    assert json_report["name"] == "Test Report"
    assert "generated" in json_report
    assert len(json_report["sections"]) == 2
    assert json_report["sections"][0]["title"] == "Section 1"
    assert json_report["sections"][0]["content"] == "This is section 1 content."
    assert json_report["sections"][1]["title"] == "Section 2"
    assert json_report["sections"][1]["content"] == "This is section 2 content."

def test_save_json_report(tmpdir):
    """
    Test saving a JSON report to a file.
    """
    # Create a report with a temporary results directory
    results_dir = tmpdir.mkdir("results")
    report = Report("Test Report", str(results_dir))
    
    # Add a section
    report.add_section("Test Section", "This is a test section.")
    
    # Save the report
    filepath = report.save_json_report("test_report")
    
    # Check that the file was created
    assert os.path.exists(filepath)
    
    # Check the file content
    with open(filepath, 'r') as f:
        content = json.load(f)
    
    assert content["name"] == "Test Report"
    assert "generated" in content
    assert len(content["sections"]) == 1
    assert content["sections"][0]["title"] == "Test Section"
    assert content["sections"][0]["content"] == "This is a test section."

def test_generate_markdown_report():
    """
    Test generating a Markdown report.
    """
    report = Report("Test Report", "/tmp/results")
    
    # Add sections
    report.add_section("Section 1", "This is section 1 content.")
    report.add_section("Section 2", "This is section 2 content.")
    
    # Generate the report
    markdown_report = report.generate_markdown_report()
    
    # Check the report content
    assert "# Test Report" in markdown_report
    assert "*Generated:" in markdown_report
    assert "## Section 1" in markdown_report
    assert "This is section 1 content." in markdown_report
    assert "## Section 2" in markdown_report
    assert "This is section 2 content." in markdown_report

def test_save_markdown_report(tmpdir):
    """
    Test saving a Markdown report to a file.
    """
    # Create a report with a temporary results directory
    results_dir = tmpdir.mkdir("results")
    report = Report("Test Report", str(results_dir))
    
    # Add a section
    report.add_section("Test Section", "This is a test section.")
    
    # Save the report
    filepath = report.save_markdown_report("test_report")
    
    # Check that the file was created
    assert os.path.exists(filepath)
    
    # Check the file content
    with open(filepath, 'r') as f:
        content = f.read()
    
    assert "# Test Report" in content
    assert "*Generated:" in content
    assert "## Test Section" in content
    assert "This is a test section." in content
