"""
Implements an analysis of GitHub issues based on labels.
"""

from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.analysis.base import BaseAnalysis
from src.core.model import Issue, Event, State
from src.utils.helpers import (
    filter_issues_by_label,
    get_issue_resolution_time,
    get_issue_comment_count,
    calculate_average
)
from src.visualization.plotting import (
    create_bar_chart,
    create_histogram
)
from src.visualization.report import Report

class LabelAnalysis(BaseAnalysis):
    """
    Implements an analysis of GitHub issues based on labels.
    This analysis focuses on:
    1. Label distribution across issues
    2. Activity level of issues with specific labels
    3. Resolution time for issues with specific labels
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__("label_analysis")
        self.results = {}
    
    def analyze(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Performs the analysis on the issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of analysis results
        """
        print("=== Label Analysis ===")
        
        # Analyze label distribution
        label_distribution = self._analyze_label_distribution(issues)
        
        # Analyze label activity (comments per issue)
        label_activity = self._analyze_label_activity(issues)
        
        # Analyze resolution time by label
        label_resolution_times = self._analyze_resolution_time(issues)
        
        # Store results
        self.results = {
            'label_distribution': label_distribution,
            'label_activity': label_activity,
            'label_resolution_times': label_resolution_times
        }
        
        return self.results
    
    def visualize(self, issues: List[Issue]):
        """
        Creates visualizations of the analysis results.
        
        Args:
            issues: List of Issue objects
        """
        # Visualize label distribution
        self._visualize_label_distribution()
        
        # Visualize label activity
        self._visualize_label_activity()
        
        # Visualize resolution time by label
        self._visualize_resolution_time()
    
    def save_results(self):
        """
        Saves the analysis results to files.
        """
        # Save results as JSON
        self.save_json(self.results, "label_analysis_results")
        
        # Generate and save report
        self._generate_report()
    
    def _analyze_label_distribution(self, issues: List[Issue]) -> Dict[str, int]:
        """
        Analyzes the distribution of labels across issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary mapping labels to counts
        """
        print("\n=== Label Distribution ===")
        
        # Count occurrences of each label
        label_counts = {}
        for issue in issues:
            for label in issue.labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
        
        # Sort labels by count (descending)
        sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Print top 10 labels
        print("\nTop 10 most common labels:")
        for i, (label, count) in enumerate(sorted_labels[:10], 1):
            print(f"{i}. {label}: {count} issues")
        
        return label_counts
    
    def _analyze_label_activity(self, issues: List[Issue]) -> Dict[str, float]:
        """
        Analyzes the activity level of issues with different labels.
        Activity is measured by the number of comments per issue.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary mapping labels to average comments per issue
        """
        print("\n=== Label Activity ===")
        
        # Calculate average comments per issue for each label
        label_activity = {}
        label_issue_counts = {}
        
        for issue in issues:
            # Count comments for this issue
            comment_count = get_issue_comment_count(issue)
            
            # Add to totals for each label
            for label in issue.labels:
                if label in label_activity:
                    label_activity[label] += comment_count
                    label_issue_counts[label] += 1
                else:
                    label_activity[label] = comment_count
                    label_issue_counts[label] = 1
        
        # Calculate average comments per issue
        label_avg_activity = {}
        for label in label_activity:
            label_avg_activity[label] = label_activity[label] / label_issue_counts[label]
        
        # Sort by average activity (descending)
        sorted_activity = sorted(label_avg_activity.items(), key=lambda x: x[1], reverse=True)
        
        # Print top 10 most active labels
        print("\nTop 10 most active labels (by average comments per issue):")
        for i, (label, avg_comments) in enumerate(sorted_activity[:10], 1):
            print(f"{i}. {label}: {avg_comments:.2f} comments per issue")
        
        return label_avg_activity
    
    def _analyze_resolution_time(self, issues: List[Issue]) -> Dict[str, float]:
        """
        Analyzes the resolution time for issues with different labels.
        Resolution time is the time between issue creation and closing.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary mapping labels to average resolution time in days
        """
        print("\n=== Resolution Time by Label ===")
        
        # Calculate average resolution time for each label
        label_resolution_times = {}
        label_closed_issue_counts = {}
        
        for issue in issues:
            # Skip open issues
            if issue.state != State.closed:
                continue
            
            # Calculate resolution time in days
            resolution_time = get_issue_resolution_time(issue)
            
            # Skip if no resolution time found
            if resolution_time is None:
                continue
            
            # Add to totals for each label
            for label in issue.labels:
                if label in label_resolution_times:
                    label_resolution_times[label] += resolution_time
                    label_closed_issue_counts[label] += 1
                else:
                    label_resolution_times[label] = resolution_time
                    label_closed_issue_counts[label] = 1
        
        # Calculate average resolution time
        label_avg_resolution = {}
        for label in label_resolution_times:
            # Skip labels with less than 3 closed issues (for statistical significance)
            if label_closed_issue_counts[label] >= 3:
                label_avg_resolution[label] = label_resolution_times[label] / label_closed_issue_counts[label]
        
        # Sort by average resolution time (ascending)
        sorted_resolution = sorted(label_avg_resolution.items(), key=lambda x: x[1])
        
        # Print top 10 fastest and slowest labels
        print("\nTop 10 fastest resolved labels (by average days to close):")
        for i, (label, avg_days) in enumerate(sorted_resolution[:10], 1):
            print(f"{i}. {label}: {avg_days:.2f} days")
        
        print("\nTop 10 slowest resolved labels (by average days to close):")
        for i, (label, avg_days) in enumerate(sorted_resolution[-10:], 1):
            print(f"{i}. {label}: {avg_days:.2f} days")
        
        return label_avg_resolution
    
    def _visualize_label_distribution(self):
        """
        Creates visualizations of the label distribution.
        """
        # Get top 10 labels
        sorted_labels = sorted(self.results['label_distribution'].items(), key=lambda x: x[1], reverse=True)
        top_labels = sorted_labels[:10]
        labels = [label for label, _ in top_labels]
        counts = [count for _, count in top_labels]
        
        # Create bar chart
        fig = create_bar_chart(
            labels=labels,
            values=counts,
            title='Top 10 Most Common Labels',
            xlabel='Label',
            ylabel='Number of Issues',
            color='skyblue',
            rotation=45
        )
        
        # Save figure
        self.save_figure(fig, "most_common_labels")
    
    def _visualize_label_activity(self):
        """
        Creates visualizations of the label activity.
        """
        # Get top 10 active labels
        sorted_activity = sorted(self.results['label_activity'].items(), key=lambda x: x[1], reverse=True)
        top_active_labels = sorted_activity[:10]
        labels = [label for label, _ in top_active_labels]
        avg_comments = [avg for _, avg in top_active_labels]
        
        # Create bar chart
        fig = create_bar_chart(
            labels=labels,
            values=avg_comments,
            title='Top 10 Most Active Labels (by Average Comments per Issue)',
            xlabel='Label',
            ylabel='Average Comments per Issue',
            color='orange',
            rotation=45
        )
        
        # Save figure
        self.save_figure(fig, "most_active_labels")
    
    def _visualize_resolution_time(self):
        """
        Creates visualizations of the resolution time by label.
        """
        # Get top 5 fastest and slowest labels
        sorted_resolution = sorted(self.results['label_resolution_times'].items(), key=lambda x: x[1])
        fastest_labels = sorted_resolution[:5]
        slowest_labels = sorted_resolution[-5:]
        
        # Combine for plotting
        combined_labels = fastest_labels + slowest_labels
        labels = [label for label, _ in combined_labels]
        avg_days = [days for _, days in combined_labels]
        
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        
        # Create bar chart with different colors for fastest and slowest
        bars = plt.bar(labels, avg_days)
        
        # Color the bars (green for fastest, red for slowest)
        for i in range(len(bars)):
            if i < 5:  # First 5 are fastest
                bars[i].set_color('green')
            else:  # Last 5 are slowest
                bars[i].set_color('red')
        
        plt.title('Resolution Time by Label (5 Fastest and 5 Slowest)')
        plt.xlabel('Label')
        plt.ylabel('Average Days to Close')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save figure
        self.save_figure(fig, "resolution_time_label")
    
    def _generate_report(self):
        """
        Generates a report of the analysis results.
        """
        # Create report
        report = Report("Label Analysis Report", self.results_dir)
        
        # Add introduction
        if self.label:
            intro = f"Analysis of issues with label '{self.label}'"
        else:
            intro = "Analysis of label distribution across all issues"
        
        report.add_section("Introduction", intro)
        
        # Add label distribution section
        sorted_labels = sorted(self.results['label_distribution'].items(), key=lambda x: x[1], reverse=True)
        top_labels = sorted_labels[:10]
        
        distribution_content = "Top 10 most common labels:\n\n"
        for i, (label, count) in enumerate(top_labels, 1):
            distribution_content += f"{i}. {label}: {count} issues\n"
        
        report.add_section("Label Distribution", distribution_content)
        
        # Add label activity section
        sorted_activity = sorted(self.results['label_activity'].items(), key=lambda x: x[1], reverse=True)
        top_active_labels = sorted_activity[:10]
        
        activity_content = "Top 10 most active labels (by average comments per issue):\n\n"
        for i, (label, avg_comments) in enumerate(top_active_labels, 1):
            activity_content += f"{i}. {label}: {avg_comments:.2f} comments per issue\n"
        
        report.add_section("Label Activity", activity_content)
        
        # Add resolution time section
        sorted_resolution = sorted(self.results['label_resolution_times'].items(), key=lambda x: x[1])
        fastest_labels = sorted_resolution[:10]
        slowest_labels = sorted_resolution[-10:]
        
        resolution_content = "Top 10 fastest resolved labels (by average days to close):\n\n"
        for i, (label, avg_days) in enumerate(fastest_labels, 1):
            resolution_content += f"{i}. {label}: {avg_days:.2f} days\n"
        
        resolution_content += "\nTop 10 slowest resolved labels (by average days to close):\n\n"
        for i, (label, avg_days) in enumerate(slowest_labels, 1):
            resolution_content += f"{i}. {label}: {avg_days:.2f} days\n"
        
        report.add_section("Resolution Time by Label", resolution_content)
        
        # Save report
        report.save_text_report("result.txt")
        report.save_markdown_report("report.md")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    LabelAnalysis().run()
