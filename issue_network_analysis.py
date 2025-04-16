from typing import List, Dict, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
import networkx as nx

from data_loader import DataLoader
from model import Issue, Event
import config

class IssueNetworkAnalysis:
    """
    Analyzes the network of relationships between issues, contributors, and labels:
    1. Contributor collaboration patterns
    2. Issue similarity based on shared contributors
    3. Label co-occurrence patterns
    4. Identification of contributor communities
    """
    
    def __init__(self):
        """
        Constructor
        """
        # No specific parameters needed for this analysis
        pass
    
    def run(self):
        """
        Starting point for this analysis.
        """
        issues: List[Issue] = DataLoader().get_issues()
        
        print(f"\n\nAnalyzing network relationships across {len(issues)} issues\n")
        
        # Build contributor-issue network
        contributor_issue_network = {}
        for issue in issues:
            # Add issue creator
            if issue.creator not in contributor_issue_network:
                contributor_issue_network[issue.creator] = set()
            contributor_issue_network[issue.creator].add(issue.number)
            
            # Add commenters
            for event in issue.events:
                if event.event_type == "commented" and event.author:
                    if event.author not in contributor_issue_network:
                        contributor_issue_network[event.author] = set()
                    contributor_issue_network[event.author].add(issue.number)
        
        print(f"Built contributor-issue network with {len(contributor_issue_network)} contributors")
        
        # Calculate contributor similarity (Jaccard index)
        contributor_similarity = {}
        contributors = list(contributor_issue_network.keys())
        for i in range(len(contributors)):
            for j in range(i+1, len(contributors)):
                c1, c2 = contributors[i], contributors[j]
                issues1 = contributor_issue_network[c1]
                issues2 = contributor_issue_network[c2]
                
                # Jaccard similarity: intersection / union
                intersection = len(issues1.intersection(issues2))
                union = len(issues1.union(issues2))
                
                if union > 0:  # Avoid division by zero
                    similarity = intersection / union
                    if similarity > 0:  # Only store non-zero similarities
                        contributor_similarity[(c1, c2)] = similarity
        
        print(f"Calculated {len(contributor_similarity)} non-zero contributor similarities")
        
        # Analyze label co-occurrence
        label_cooccurrence = {}
        for issue in issues:
            for i in range(len(issue.labels)):
                for j in range(i+1, len(issue.labels)):
                    label1, label2 = issue.labels[i], issue.labels[j]
                    pair = tuple(sorted([label1, label2]))
                    
                    if pair in label_cooccurrence:
                        label_cooccurrence[pair] += 1
                    else:
                        label_cooccurrence[pair] = 1
        
        print(f"Analyzed {len(label_cooccurrence)} label co-occurrences")
        
        # Visualize contributor network
        self._visualize_contributor_network(contributor_issue_network, contributor_similarity)
        
        # Visualize label co-occurrence network
        self._visualize_label_network(issues, label_cooccurrence)
    
    def _visualize_contributor_network(self, contributor_issue_network, contributor_similarity):
        """
        Visualizes the contributor network where:
        - Nodes are contributors
        - Edges represent similarity between contributors
        - Edge weights based on Jaccard similarity
        """
        print("\nVisualizing contributor network...")
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes (contributors)
        for contributor, issues in contributor_issue_network.items():
            G.add_node(contributor, size=len(issues))
        
        # Add edges (similarities)
        for (c1, c2), similarity in contributor_similarity.items():
            # Only add edges with significant similarity
            if similarity > 0.1:  # Threshold to reduce graph complexity
                G.add_edge(c1, c2, weight=similarity)
        
        # Filter to keep only connected nodes
        G = G.subgraph(max(nx.connected_components(G), key=len))
        
        # Limit to top contributors for readability
        if len(G.nodes) > 50:
            top_contributors = sorted(G.nodes, key=lambda x: G.nodes[x]['size'], reverse=True)[:50]
            G = G.subgraph(top_contributors)
        
        print(f"Plotting network with {len(G.nodes)} contributors and {len(G.edges)} connections")
        
        # Set up the plot
        plt.figure(figsize=(14, 10))
        
        # Node sizes based on number of issues
        node_sizes = [G.nodes[node]['size'] * 10 for node in G.nodes]
        
        # Edge weights based on similarity
        edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges]
        
        # Use a layout that spreads nodes apart
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw the network
        nx.draw_networkx(
            G, 
            pos=pos,
            node_size=node_sizes,
            node_color='skyblue',
            edge_color='gray',
            width=edge_weights,
            with_labels=True,
            font_size=8,
            alpha=0.8
        )
        
        plt.title("Contributor Collaboration Network")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def _visualize_label_network(self, issues, label_cooccurrence):
        """
        Visualizes the label co-occurrence network where:
        - Nodes are labels
        - Edges represent co-occurrence
        - Edge weights based on co-occurrence count
        """
        print("\nVisualizing label co-occurrence network...")
        
        # Count label frequencies
        label_counts = defaultdict(int)
        for issue in issues:
            for label in issue.labels:
                label_counts[label] += 1
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes (labels)
        for label, count in label_counts.items():
            G.add_node(label, size=count)
        
        # Add edges (co-occurrences)
        for (label1, label2), count in label_cooccurrence.items():
            # Only add edges with significant co-occurrence
            if count > 1:  # Threshold to reduce graph complexity
                G.add_edge(label1, label2, weight=count)
        
        # Filter to keep only connected nodes
        G = G.subgraph(max(nx.connected_components(G), key=len))
        
        # Limit to top labels for readability
        if len(G.nodes) > 50:
            top_labels = sorted(G.nodes, key=lambda x: G.nodes[x]['size'], reverse=True)[:50]
            G = G.subgraph(top_labels)
        
        print(f"Plotting network with {len(G.nodes)} labels and {len(G.edges)} connections")
        
        # Set up the plot
        plt.figure(figsize=(14, 10))
        
        # Node sizes based on label frequency
        node_sizes = [G.nodes[node]['size'] * 20 for node in G.nodes]
        
        # Edge weights based on co-occurrence count
        edge_weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges]
        
        # Use a layout that spreads nodes apart
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # Draw the network
        nx.draw_networkx(
            G, 
            pos=pos,
            node_size=node_sizes,
            node_color='lightgreen',
            edge_color='gray',
            width=edge_weights,
            with_labels=True,
            font_size=8,
            alpha=0.8
        )
        
        plt.title("Label Co-occurrence Network")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueNetworkAnalysis().run()
