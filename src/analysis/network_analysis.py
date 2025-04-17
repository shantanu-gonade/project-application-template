"""
Implements an analysis of GitHub issues based on network relationships.
"""

from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx

from src.analysis.base import BaseAnalysis
from src.core.model import Issue, Event, State
from src.utils.helpers import (
    get_issue_unique_participants,
    filter_issues_by_label
)
from src.visualization.report import Report

class IssueNetworkAnalysis(BaseAnalysis):
    """
    Implements an analysis of GitHub issues based on network relationships.
    This analysis focuses on:
    1. Label co-occurrence network
    2. Contributor collaboration network
    3. Issue similarity network
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__("network_analysis")
        self.results = {}
    
    def analyze(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Performs the analysis on the issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of analysis results
        """
        print("=== Network Analysis ===")
        
        # Analyze label co-occurrence
        label_network = self._analyze_label_cooccurrence(issues)
        
        # Analyze contributor collaboration
        contributor_network = self._analyze_contributor_collaboration(issues)
        
        # Store results
        self.results = {
            'label_network': label_network,
            'contributor_network': contributor_network
        }
        
        return self.results
    
    def visualize(self, issues: List[Issue]):
        """
        Creates visualizations of the analysis results.
        
        Args:
            issues: List of Issue objects
        """
        # Visualize label co-occurrence network
        self._visualize_label_network()
        
        # Visualize contributor collaboration network
        self._visualize_contributor_network()
    
    def save_results(self):
        """
        Saves the analysis results to files.
        """
        # Save results as JSON
        for key, value in self.results.items():
            # Convert networkx graphs to serializable format
            if key == 'label_network' or key == 'contributor_network':
                serializable_data = {
                    'nodes': list(value['graph'].nodes()),
                    'edges': list(value['graph'].edges(data=True))
                }
                self.save_json(serializable_data, key)
            else:
                self.save_json(value, key)
        
        # Generate and save report
        self._generate_report()
    
    def _analyze_label_cooccurrence(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes label co-occurrence in issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary with label co-occurrence network
        """
        print("\n=== Label Co-occurrence Network ===")
        
        # Create a graph
        G = nx.Graph()
        
        # Count label occurrences and co-occurrences
        label_counts = defaultdict(int)
        cooccurrence_counts = defaultdict(int)
        
        for issue in issues:
            # Skip issues with less than 2 labels
            if len(issue.labels) < 2:
                continue
            
            # Update label counts
            for label in issue.labels:
                label_counts[label] += 1
            
            # Update co-occurrence counts
            for i, label1 in enumerate(issue.labels):
                for label2 in issue.labels[i+1:]:
                    if label1 < label2:
                        cooccurrence_counts[(label1, label2)] += 1
                    else:
                        cooccurrence_counts[(label2, label1)] += 1
        
        # Add nodes to the graph
        for label, count in label_counts.items():
            G.add_node(label, count=count)
        
        # Add edges to the graph
        for (label1, label2), count in cooccurrence_counts.items():
            G.add_edge(label1, label2, weight=count)
        
        print(f"Created label co-occurrence network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return {
            'graph': G,
            'label_counts': dict(label_counts),
            'cooccurrence_counts': {f"{l1}_{l2}": count for (l1, l2), count in cooccurrence_counts.items()}
        }
    
    def _analyze_contributor_collaboration(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes contributor collaboration on issues.
        
        Args:
            issues: List[Issue]: List of Issue objects
            
        Returns:
            Dictionary with contributor collaboration network
        """
        print("\n=== Contributor Collaboration Network ===")
        
        # Create a graph
        G = nx.Graph()
        
        # Count contributor activity and collaborations
        contributor_counts = defaultdict(int)
        collaboration_counts = defaultdict(int)
        
        for issue in issues:
            # Get unique participants
            participants = get_issue_unique_participants(issue)
            
            # Skip issues with less than 2 participants
            if len(participants) < 2:
                continue
            
            # Update contributor counts
            for participant in participants:
                contributor_counts[participant] += 1
            
            # Update collaboration counts
            for i, participant1 in enumerate(participants):
                for participant2 in participants[i+1:]:
                    if participant1 < participant2:
                        collaboration_counts[(participant1, participant2)] += 1
                    else:
                        collaboration_counts[(participant2, participant1)] += 1
        
        # Add nodes to the graph
        for contributor, count in contributor_counts.items():
            G.add_node(contributor, count=count)
        
        # Add edges to the graph
        for (contributor1, contributor2), count in collaboration_counts.items():
            G.add_edge(contributor1, contributor2, weight=count)
        
        print(f"Created contributor collaboration network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return {
            'graph': G,
            'contributor_counts': dict(contributor_counts),
            'collaboration_counts': {f"{c1}_{c2}": count for (c1, c2), count in collaboration_counts.items()}
        }
    
    def _visualize_label_network(self):
        """
        Creates visualizations of the label co-occurrence network.
        """
        G = self.results['label_network']['graph']
        
        # Skip if the graph is empty
        if G.number_of_nodes() == 0:
            print("Label network is empty, skipping visualization")
            return
        
        # Create figure
        fig = plt.figure(figsize=(12, 12))
        
        # Calculate node sizes based on occurrence count
        node_sizes = [G.nodes[node]['count'] * 20 for node in G.nodes()]
        
        # Calculate edge widths based on co-occurrence count
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        # Calculate node colors based on degree
        node_degrees = dict(G.degree())
        node_colors = [node_degrees[node] for node in G.nodes()]
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, cmap='viridis')
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title('Label Co-occurrence Network')
        plt.axis('off')
        plt.tight_layout()
        
        self.save_figure(fig, "label_cooccurance_network")
        
        # Create a simplified version with top labels only
        if G.number_of_nodes() > 20:
            # Get top 20 labels by degree
            top_labels = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:20]
            top_label_names = [label for label, _ in top_labels]
            
            # Create subgraph
            subgraph = G.subgraph(top_label_names)
            
            # Create figure
            fig = plt.figure(figsize=(12, 12))
            
            # Calculate node sizes based on occurrence count
            node_sizes = [G.nodes[node]['count'] * 30 for node in subgraph.nodes()]
            
            # Calculate edge widths based on co-occurrence count
            edge_widths = [subgraph[u][v]['weight'] * 0.8 for u, v in subgraph.edges()]
            
            # Calculate node colors based on degree
            node_degrees = dict(subgraph.degree())
            node_colors = [node_degrees[node] for node in subgraph.nodes()]
            
            # Use spring layout for node positioning
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Draw the graph
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, cmap='viridis')
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.5, edge_color='gray')
            nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif')
            
            plt.title('Top 20 Labels Co-occurrence Network')
            plt.axis('off')
            plt.tight_layout()
            
            self.save_figure(fig, "top_labels_cooccurance_network")
    
    def _visualize_contributor_network(self):
        """
        Creates visualizations of the contributor collaboration network.
        """
        G = self.results['contributor_network']['graph']
        
        # Skip if the graph is empty
        if G.number_of_nodes() == 0:
            print("Contributor network is empty, skipping visualization")
            return
        
        # Create figure
        fig = plt.figure(figsize=(12, 12))
        
        # Calculate node sizes based on activity count
        node_sizes = [G.nodes[node]['count'] * 10 for node in G.nodes()]
        
        # Calculate edge widths based on collaboration count
        edge_widths = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
        
        # Calculate node colors based on degree
        node_degrees = dict(G.degree())
        node_colors = [node_degrees[node] for node in G.nodes()]
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, seed=42)
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, cmap='viridis')
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        
        plt.title('Contributor Collaboration Network')
        plt.axis('off')
        plt.tight_layout()
        
        self.save_figure(fig, "contributor_collaboration_network")
        
        # Create a simplified version with top contributors only
        if G.number_of_nodes() > 20:
            # Get top 20 contributors by degree
            top_contributors = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:20]
            top_contributor_names = [contributor for contributor, _ in top_contributors]
            
            # Create subgraph
            subgraph = G.subgraph(top_contributor_names)
            
            # Create figure
            fig = plt.figure(figsize=(12, 12))
            
            # Calculate node sizes based on activity count
            node_sizes = [G.nodes[node]['count'] * 20 for node in subgraph.nodes()]
            
            # Calculate edge widths based on collaboration count
            edge_widths = [subgraph[u][v]['weight'] * 0.8 for u, v in subgraph.edges()]
            
            # Calculate node colors based on degree
            node_degrees = dict(subgraph.degree())
            node_colors = [node_degrees[node] for node in subgraph.nodes()]
            
            # Use spring layout for node positioning
            pos = nx.spring_layout(subgraph, seed=42)
            
            # Draw the graph
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, cmap='viridis')
            nx.draw_networkx_edges(subgraph, pos, width=edge_widths, alpha=0.5, edge_color='gray')
            nx.draw_networkx_labels(subgraph, pos, font_size=10, font_family='sans-serif')
            
            plt.title('Top 20 Contributors Collaboration Network')
            plt.axis('off')
            plt.tight_layout()
            
            self.save_figure(fig, "top_contributors_collaboration_network")
    
    def _generate_report(self):
        """
        Generates a report of the analysis results.
        """
        # Create report
        report = Report("Network Analysis Report", self.results_dir)
        
        # Add introduction
        intro = "Analysis of network relationships in GitHub issues"
        report.add_section("Introduction", intro)
        
        # Add label network section
        label_network = self.results['label_network']
        
        label_network_content = "Label co-occurrence network:\n\n"
        label_network_content += f"Number of labels: {len(label_network['label_counts'])}\n"
        label_network_content += f"Number of co-occurrences: {len(label_network['cooccurrence_counts'])}\n\n"
        
        # Add top labels by occurrence
        sorted_labels = sorted(label_network['label_counts'].items(), key=lambda x: x[1], reverse=True)
        
        label_network_content += "Top 10 most common labels:\n\n"
        for i, (label, count) in enumerate(sorted_labels[:10], 1):
            label_network_content += f"{i}. {label}: {count} occurrences\n"
        
        # Add top co-occurrences
        sorted_cooccurrences = sorted(label_network['cooccurrence_counts'].items(), key=lambda x: x[1], reverse=True)
        
        label_network_content += "\nTop 10 most common label co-occurrences:\n\n"
        for i, (label_pair, count) in enumerate(sorted_cooccurrences[:10], 1):
            label1, label2 = label_pair.split('_')
            label_network_content += f"{i}. {label1} + {label2}: {count} co-occurrences\n"
        
        report.add_section("Label Co-occurrence Network", label_network_content)
        
        # Add contributor network section
        contributor_network = self.results['contributor_network']
        
        contributor_network_content = "Contributor collaboration network:\n\n"
        contributor_network_content += f"Number of contributors: {len(contributor_network['contributor_counts'])}\n"
        contributor_network_content += f"Number of collaborations: {len(contributor_network['collaboration_counts'])}\n\n"
        
        # Add top contributors by activity
        sorted_contributors = sorted(contributor_network['contributor_counts'].items(), key=lambda x: x[1], reverse=True)
        
        contributor_network_content += "Top 10 most active contributors:\n\n"
        for i, (contributor, count) in enumerate(sorted_contributors[:10], 1):
            contributor_network_content += f"{i}. {contributor}: {count} issues\n"
        
        # Add top collaborations
        sorted_collaborations = sorted(contributor_network['collaboration_counts'].items(), key=lambda x: x[1], reverse=True)
        
        contributor_network_content += "\nTop 10 most common collaborations:\n\n"
        for i, (contributor_pair, count) in enumerate(sorted_collaborations[:10], 1):
            contributor1, contributor2 = contributor_pair.split('_')
            contributor_network_content += f"{i}. {contributor1} + {contributor2}: {count} collaborations\n"
        
        report.add_section("Contributor Collaboration Network", contributor_network_content)
        
        # Save report
        report.save_text_report("result.txt")
        report.save_markdown_report("report.md")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueNetworkAnalysis().run()
