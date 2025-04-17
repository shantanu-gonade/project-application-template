from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from data_loader import DataLoader
from model import Issue, Event, State
import config

class LabelAnalysis:
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
        # Parameter is passed in via command line (--label)
        self.label: str = config.get_parameter('label')

    def run(self):
        """
        Starting point for this analysis.
        """
        issues: List[Issue] = DataLoader().get_issues()

        # Filter issues by label if specified
        if self.label:
            filtered_issues = [issue for issue in issues if self.label in issue.labels]
            print(f"\n\nAnalyzing {len(filtered_issues)} issues with label '{self.label}'\n")
        else:
            filtered_issues = issues
            print(f"\n\nAnalyzing label distribution across all {len(issues)} issues\n")

        # Analyze label distribution
        self._analyze_label_distribution(issues)

        # Analyze label activity (comments per issue)
        self._analyze_label_activity(issues)

        # Analyze resolution time by label
        self._analyze_resolution_time(issues)

    def _analyze_label_distribution(self, issues: List[Issue]):
        """
        Analyzes the distribution of labels across issues
        """
        print("=== Label Distribution ===")

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

        # Plot label distribution (top 10)
        plt.figure(figsize=(12, 6))

        # Get top 10 labels
        top_labels = sorted_labels[:10]
        labels = [label for label, _ in top_labels]
        counts = [count for _, count in top_labels]

        # Create bar chart
        plt.bar(labels, counts)
        plt.title('Top 10 Most Common Labels')
        plt.xlabel('Label')
        plt.ylabel('Number of Issues')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Show plot
        plt.show()

    def _analyze_label_activity(self, issues: List[Issue]):
        """
        Analyzes the activity level of issues with different labels
        Activity is measured by the number of comments per issue
        """
        print("\n=== Label Activity ===")

        # Calculate average comments per issue for each label
        label_activity = {}
        label_issue_counts = {}

        for issue in issues:
            # Count comments for this issue
            comment_count = len([event for event in issue.events if event.event_type == "commented"])

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

        # Plot label activity (top 10)
        plt.figure(figsize=(12, 6))

        # Get top 10 active labels
        top_active_labels = sorted_activity[:10]
        labels = [label for label, _ in top_active_labels]
        avg_comments = [avg for _, avg in top_active_labels]

        # Create bar chart
        plt.bar(labels, avg_comments)
        plt.title('Top 10 Most Active Labels (by Average Comments per Issue)')
        plt.xlabel('Label')
        plt.ylabel('Average Comments per Issue')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Show plot
        plt.show()

    def _analyze_resolution_time(self, issues: List[Issue]):
        """
        Analyzes the resolution time for issues with different labels
        Resolution time is the time between issue creation and closing
        """
        print("\n=== Resolution Time by Label ===")

        # Calculate average resolution time for each label
        label_resolution_times = {}
        label_closed_issue_counts = {}

        for issue in issues:
            # Skip open issues
            if issue.state != State.closed:
                continue

            # Find closing event
            closing_event = None
            for event in issue.events:
                if event.event_type == "closed":
                    closing_event = event
                    break

            # Skip if no closing event found
            if not closing_event or not closing_event.event_date:
                continue

            # Calculate resolution time in days
            resolution_time = (closing_event.event_date - issue.created_date).days

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

        # Plot resolution time (top 5 fastest and slowest)
        plt.figure(figsize=(12, 6))

        # Get top 5 fastest and slowest labels
        fastest_labels = sorted_resolution[:5]
        slowest_labels = sorted_resolution[-5:]

        # Combine for plotting
        combined_labels = fastest_labels + slowest_labels
        labels = [label for label, _ in combined_labels]
        avg_days = [days for _, days in combined_labels]

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

        # Show plot
        plt.show()


if __name__ == '__main__':
    # Invoke run method when running this module directly
    LabelAnalysis().run()
