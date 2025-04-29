"""
Tests for the network analysis module.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from src.analysis.network_analysis import IssueNetworkAnalysis
from src.core.model import Issue, Event, State

def test_network_analysis_initialization():
    """
    Test that IssueNetworkAnalysis is correctly initialized.
    """
    analysis = IssueNetworkAnalysis()
    
    assert analysis.name == "network_analysis"
    assert analysis.results == {}

def test_analyze(sample_issues, monkeypatch):
    """
    Test the analyze method.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Mock the helper methods
    label_graph = nx.Graph()
    label_graph.add_node("bug", count=2)
    label_graph.add_node("documentation", count=1)
    label_graph.add_edge("bug", "documentation", weight=1)
    
    contributor_graph = nx.Graph()
    contributor_graph.add_node("user1", count=2)
    contributor_graph.add_node("user2", count=1)
    contributor_graph.add_edge("user1", "user2", weight=1)
    
    analysis._analyze_label_cooccurrence = MagicMock(return_value={
        'graph': label_graph,
        'label_counts': {"bug": 2, "documentation": 1},
        'cooccurrence_counts': {"bug_documentation": 1}
    })
    
    analysis._analyze_contributor_collaboration = MagicMock(return_value={
        'graph': contributor_graph,
        'contributor_counts': {"user1": 2, "user2": 1},
        'collaboration_counts': {"user1_user2": 1}
    })
    
    # Run the analysis
    results = analysis.analyze(sample_issues)
    
    # Check that the helper methods were called
    analysis._analyze_label_cooccurrence.assert_called_once_with(sample_issues)
    analysis._analyze_contributor_collaboration.assert_called_once_with(sample_issues)
    
    # Check the results
    assert results == {
        'label_network': analysis._analyze_label_cooccurrence.return_value,
        'contributor_network': analysis._analyze_contributor_collaboration.return_value
    }
    
    # Check that the results were stored
    assert analysis.results == results

def test_analyze_label_cooccurrence(sample_issues):
    """
    Test the _analyze_label_cooccurrence method.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Run the analysis
    results = analysis._analyze_label_cooccurrence(sample_issues)
    
    # Check the results structure
    assert 'graph' in results
    assert 'label_counts' in results
    assert 'cooccurrence_counts' in results
    
    # Check that the graph is a networkx Graph
    assert isinstance(results['graph'], nx.Graph)
    
    # Check that label_counts contains the expected labels
    assert 'bug' in results['label_counts']
    assert 'documentation' in results['label_counts']
    assert 'enhancement' in results['label_counts']
    
    # Check that the graph has nodes for each label
    for label in results['label_counts']:
        assert label in results['graph'].nodes
        assert 'count' in results['graph'].nodes[label]
    
    # Check that the graph has edges for co-occurrences
    # In sample_issues, issue 1 has both 'bug' and 'documentation' labels
    if 'bug' in results['graph'].nodes and 'documentation' in results['graph'].nodes:
        assert results['graph'].has_edge('bug', 'documentation') or results['graph'].has_edge('documentation', 'bug')

def test_analyze_contributor_collaboration(sample_issues, monkeypatch):
    """
    Test the _analyze_contributor_collaboration method.
    """
    
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
        
    # Mock get_issue_unique_participants to return specific participants for each issue
    def mock_get_issue_unique_participants(issue):
        if str(issue.number) == '1':
            return ['user1', 'user2', 'user3']  # Issue 1 has 3 participants
        elif str(issue.number) == '2':
            return ['user2', 'user1']  # Issue 2 has 2 participants
        elif str(issue.number) == '4':
            return ['user4', 'user1']  # Issue 4 has 2 participants
        else:
            return [str(issue.creator)]  # Other issues have only the creator
    
    monkeypatch.setattr("src.analysis.network_analysis.get_issue_unique_participants", mock_get_issue_unique_participants)

    # Run the analysis
    results = analysis._analyze_contributor_collaboration(sample_issues)
    
    # Check the results structure
    assert 'graph' in results
    assert 'contributor_counts' in results
    assert 'collaboration_counts' in results
    
    # Check that the graph is a networkx Graph
    assert isinstance(results['graph'], nx.Graph)
    
    # Check that contributor_counts contains at least one expected contributor
    assert any(user in results['contributor_counts'] for user in ['user1', 'user2', 'user3'])
    
    # Check that the graph has nodes for each contributor
    for contributor in results['contributor_counts']:
        assert contributor in results['graph'].nodes
        assert 'count' in results['graph'].nodes[contributor]
    
    # Check that the graph has edges for collaborations
    # Based on our mock, user1 and user2 collaborate on issues 1 and 2
    assert results['graph'].has_edge('user1', 'user2') or results['graph'].has_edge('user2', 'user1')
    # user1 and user3 collaborate on issue 1
    assert results['graph'].has_edge('user1', 'user3') or results['graph'].has_edge('user3', 'user1')
    # user2 and user3 collaborate on issue 1
    assert results['graph'].has_edge('user2', 'user3') or results['graph'].has_edge('user3', 'user2')
    # user1 and user4 collaborate on issue 4
    assert results['graph'].has_edge('user1', 'user4') or results['graph'].has_edge('user4', 'user1')

def test_visualize(sample_issues, monkeypatch):
    """
    Test the visualize method.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results with mock graphs
    label_graph = nx.Graph()
    label_graph.add_node("bug", count=2)
    label_graph.add_node("documentation", count=1)
    label_graph.add_edge("bug", "documentation", weight=1)
    
    contributor_graph = nx.Graph()
    contributor_graph.add_node("user1", count=2)
    contributor_graph.add_node("user2", count=1)
    contributor_graph.add_edge("user1", "user2", weight=1)
    
    analysis.results = {
        'label_network': {
            'graph': label_graph,
            'label_counts': {"bug": 2, "documentation": 1},
            'cooccurrence_counts': {"bug_documentation": 1}
        },
        'contributor_network': {
            'graph': contributor_graph,
            'contributor_counts': {"user1": 2, "user2": 1},
            'collaboration_counts': {"user1_user2": 1}
        }
    }
    
    # Mock the helper methods
    analysis._visualize_label_network = MagicMock()
    analysis._visualize_contributor_network = MagicMock()
    
    # Run the visualization
    analysis.visualize(sample_issues)
    
    # Check that the helper methods were called
    analysis._visualize_label_network.assert_called_once()
    analysis._visualize_contributor_network.assert_called_once()

def test_visualize_label_network(monkeypatch):
    """
    Test the _visualize_label_network method.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results with a mock graph
    label_graph = nx.Graph()
    label_graph.add_node("bug", count=2)
    label_graph.add_node("documentation", count=1)
    label_graph.add_edge("bug", "documentation", weight=1)
    
    analysis.results = {
        'label_network': {
            'graph': label_graph,
            'label_counts': {"bug": 2, "documentation": 1},
            'cooccurrence_counts': {"bug_documentation": 1}
        }
    }
    
    # Mock plt.figure and networkx drawing functions
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("networkx.spring_layout", MagicMock(return_value={}))
    monkeypatch.setattr("networkx.draw_networkx_nodes", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_edges", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_labels", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.axis", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_label_network()
    
    # Check that save_figure was called
    analysis.save_figure.assert_called_once()
    
    # Check the filename
    args, _ = analysis.save_figure.call_args
    assert args[1] == "label_cooccurance_network"

def test_visualize_label_network_with_many_labels(monkeypatch):
    """
    Test the _visualize_label_network method with many labels.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results with a mock graph with many nodes
    label_graph = nx.Graph()
    for i in range(30):  # Create 30 nodes to trigger the top labels visualization
        label_graph.add_node(f"label{i}", count=30-i)
    
    # Add some edges
    for i in range(20):
        label_graph.add_edge(f"label{i}", f"label{i+1}", weight=1)
    
    analysis.results = {
        'label_network': {
            'graph': label_graph,
            'label_counts': {f"label{i}": 30-i for i in range(30)},
            'cooccurrence_counts': {f"label{i}_label{i+1}": 1 for i in range(20)}
        }
    }
    
    # Mock plt.figure and networkx drawing functions
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("networkx.spring_layout", MagicMock(return_value={}))
    monkeypatch.setattr("networkx.draw_networkx_nodes", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_edges", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_labels", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.axis", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_label_network()
    
    # Check that save_figure was called twice (once for full network, once for top labels)
    assert analysis.save_figure.call_count == 2
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "label_cooccurance_network" in filenames
    assert "top_labels_cooccurance_network" in filenames

def test_visualize_contributor_network(monkeypatch):
    """
    Test the _visualize_contributor_network method.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results with a mock graph
    contributor_graph = nx.Graph()
    contributor_graph.add_node("user1", count=2)
    contributor_graph.add_node("user2", count=1)
    contributor_graph.add_edge("user1", "user2", weight=1)
    
    analysis.results = {
        'contributor_network': {
            'graph': contributor_graph,
            'contributor_counts': {"user1": 2, "user2": 1},
            'collaboration_counts': {"user1_user2": 1}
        }
    }
    
    # Mock plt.figure and networkx drawing functions
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("networkx.spring_layout", MagicMock(return_value={}))
    monkeypatch.setattr("networkx.draw_networkx_nodes", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_edges", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_labels", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.axis", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_contributor_network()
    
    # Check that save_figure was called
    analysis.save_figure.assert_called_once()
    
    # Check the filename
    args, _ = analysis.save_figure.call_args
    assert args[1] == "contributor_collaboration_network"

def test_visualize_contributor_network_with_many_contributors(monkeypatch):
    """
    Test the _visualize_contributor_network method with many contributors.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results with a mock graph with many nodes
    contributor_graph = nx.Graph()
    for i in range(30):  # Create 30 nodes to trigger the top contributors visualization
        contributor_graph.add_node(f"user{i}", count=30-i)
    
    # Add some edges
    for i in range(20):
        contributor_graph.add_edge(f"user{i}", f"user{i+1}", weight=1)
    
    analysis.results = {
        'contributor_network': {
            'graph': contributor_graph,
            'contributor_counts': {f"user{i}": 30-i for i in range(30)},
            'collaboration_counts': {f"user{i}_user{i+1}": 1 for i in range(20)}
        }
    }
    
    # Mock plt.figure and networkx drawing functions
    mock_figure = MagicMock()
    monkeypatch.setattr("matplotlib.pyplot.figure", MagicMock(return_value=mock_figure))
    monkeypatch.setattr("networkx.spring_layout", MagicMock(return_value={}))
    monkeypatch.setattr("networkx.draw_networkx_nodes", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_edges", MagicMock())
    monkeypatch.setattr("networkx.draw_networkx_labels", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.title", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.axis", MagicMock())
    monkeypatch.setattr("matplotlib.pyplot.tight_layout", MagicMock())
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualization
    analysis._visualize_contributor_network()
    
    # Check that save_figure was called twice (once for full network, once for top contributors)
    assert analysis.save_figure.call_count == 2
    
    # Check the filenames
    filenames = [args[1] for args, _ in analysis.save_figure.call_args_list]
    assert "contributor_collaboration_network" in filenames
    assert "top_contributors_collaboration_network" in filenames

def test_visualize_empty_networks(monkeypatch):
    """
    Test the visualization methods with empty networks.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results with empty graphs
    label_graph = nx.Graph()
    contributor_graph = nx.Graph()
    
    analysis.results = {
        'label_network': {
            'graph': label_graph,
            'label_counts': {},
            'cooccurrence_counts': {}
        },
        'contributor_network': {
            'graph': contributor_graph,
            'contributor_counts': {},
            'collaboration_counts': {}
        }
    }
    
    # Mock print to check it's called
    mock_print = MagicMock()
    monkeypatch.setattr("builtins.print", mock_print)
    
    # Mock the save_figure method
    analysis.save_figure = MagicMock()
    
    # Run the visualizations
    analysis._visualize_label_network()
    analysis._visualize_contributor_network()
    
    # Check that save_figure was not called
    assert analysis.save_figure.call_count == 0
    
    # Check that print was called with the expected messages
    mock_print.assert_any_call("Label network is empty, skipping visualization")
    mock_print.assert_any_call("Contributor network is empty, skipping visualization")

def test_save_results(monkeypatch, tmpdir):
    """
    Test the save_results method.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results with mock graphs
    label_graph = nx.Graph()
    label_graph.add_node("bug", count=2)
    label_graph.add_node("documentation", count=1)
    label_graph.add_edge("bug", "documentation", weight=1)
    
    contributor_graph = nx.Graph()
    contributor_graph.add_node("user1", count=2)
    contributor_graph.add_node("user2", count=1)
    contributor_graph.add_edge("user1", "user2", weight=1)
    
    analysis.results = {
        'label_network': {
            'graph': label_graph,
            'label_counts': {"bug": 2, "documentation": 1},
            'cooccurrence_counts': {"bug_documentation": 1}
        },
        'contributor_network': {
            'graph': contributor_graph,
            'contributor_counts': {"user1": 2, "user2": 1},
            'collaboration_counts': {"user1_user2": 1}
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
    
    # Check that save_json was called for each network
    assert analysis.save_json.call_count == 2
    
    # Check that _generate_report was called
    analysis._generate_report.assert_called_once()
    
    # Check that the graphs were converted to serializable format
    serializable_label_network = analysis.save_json.call_args_list[0][0][0]
    assert 'nodes' in serializable_label_network
    assert 'edges' in serializable_label_network
    
    serializable_contributor_network = analysis.save_json.call_args_list[1][0][0]
    assert 'nodes' in serializable_contributor_network
    assert 'edges' in serializable_contributor_network

def test_generate_report(monkeypatch):
    """
    Test the _generate_report method.
    """
    # Create an IssueNetworkAnalysis instance
    analysis = IssueNetworkAnalysis()
    
    # Set up the results
    analysis.results = {
        'label_network': {
            'label_counts': {"bug": 5, "documentation": 3, "enhancement": 2},
            'cooccurrence_counts': {"bug_documentation": 2, "bug_enhancement": 1, "documentation_enhancement": 1}
        },
        'contributor_network': {
            'contributor_counts': {"user1": 5, "user2": 3, "user3": 2},
            'collaboration_counts': {"user1_user2": 2, "user1_user3": 1, "user2_user3": 1}
        }
    }
    
    # Mock the Report class
    mock_report = MagicMock()
    mock_report_class = MagicMock(return_value=mock_report)
    monkeypatch.setattr("src.analysis.network_analysis.Report", mock_report_class)
    
    # Run the _generate_report method
    analysis._generate_report()
    
    # Check that Report was initialized correctly
    mock_report_class.assert_called_once_with("Network Analysis Report", analysis.results_dir)
    
    # Check that add_section was called for each section
    assert mock_report.add_section.call_count >= 3  # At least 3 sections
    
    # Check that save_text_report and save_markdown_report were called
    mock_report.save_text_report.assert_called_once_with("result.txt")
    mock_report.save_markdown_report.assert_called_once_with("report.md")
