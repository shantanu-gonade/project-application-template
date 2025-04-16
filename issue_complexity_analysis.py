from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from data_loader import DataLoader
from model import Issue, Event, State
import config

class IssueComplexityAnalysis:
    """
    Analyzes the complexity of issues based on various metrics:
    1. Discussion length (number of comments)
    2. Discussion duration (time between first and last comment)
    3. Number of unique participants
    4. Code snippet inclusion (presence of code blocks)
    5. Correlation between complexity and resolution time
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
            print(f"\n\nAnalyzing complexity of {len(filtered_issues)} issues with label '{self.label}'\n")
        else:
            filtered_issues = issues
            print(f"\n\nAnalyzing complexity across all {len(issues)} issues\n")
        
        # Calculate complexity score for each issue
        complexity_scores = []
        for issue in filtered_issues:
            # Count comments
            comment_count = len([e for e in issue.events if e.event_type == "commented"])
            
            # Calculate discussion duration
            comment_events = [e for e in issue.events if e.event_type == "commented" and e.event_date]
            discussion_duration = 0
            if comment_events:
                first_comment = min(comment_events, key=lambda e: e.event_date).event_date
                last_comment = max(comment_events, key=lambda e: e.event_date).event_date
                discussion_duration = (last_comment - first_comment).total_seconds() / 86400  # Convert to days
            
            # Count unique participants
            participants = set([issue.creator])
            for event in issue.events:
                if event.author:
                    participants.add(event.author)
            
            # Check for code blocks in issue body and comments
            has_code = '```' in issue.text if issue.text else False
            for event in issue.events:
                if event.event_type == "commented" and event.comment:
                    if '```' in event.comment:
                        has_code = True
            
            # Calculate complexity score (weighted sum)
            score = (
                comment_count * 0.5 + 
                len(participants) * 1.0 + 
                (5.0 if has_code else 0) +
                min(discussion_duration * 0.1, 5.0)  # Cap duration contribution at 5.0
            )
            
            # Calculate resolution time (if issue is closed)
            resolution_time = None
            if issue.state == State.closed:
                closed_events = [e for e in issue.events if e.event_type == "closed" and e.event_date]
                if closed_events and issue.created_date:
                    closed_date = max(closed_events, key=lambda e: e.event_date).event_date
                    resolution_time = (closed_date - issue.created_date).total_seconds() / 86400  # Convert to days
            
            complexity_scores.append({
                'issue_number': issue.number,
                'title': issue.title,
                'score': score,
                'comment_count': comment_count,
                'discussion_duration': discussion_duration,
                'participant_count': len(participants),
                'has_code': has_code,
                'labels': issue.labels,
                'resolution_time': resolution_time,
                'state': issue.state
            })
        
        print(f"Calculated complexity scores for {len(complexity_scores)} issues")
        
        # Plot distribution of complexity scores
        self._plot_complexity_distribution(complexity_scores)
        
        # Analyze correlation between complexity and resolution time
        self._analyze_complexity_resolution_correlation(complexity_scores)
        
        # Identify most complex issues by label
        self._identify_complex_issues_by_label(complexity_scores)
    
    def _plot_complexity_distribution(self, complexity_scores):
        """
        Plots the distribution of complexity scores.
        """
        print("\nPlotting complexity score distribution...")
        
        # Extract scores
        scores = [score['score'] for score in complexity_scores]
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Complexity Score')
        plt.ylabel('Number of Issues')
        plt.title('Distribution of Issue Complexity Scores')
        plt.grid(axis='y', alpha=0.75)
        
        # Add mean and median lines
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        plt.axvline(mean_score, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_score:.2f}')
        plt.axvline(median_score, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_score:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Create a box plot to show the distribution
        plt.figure(figsize=(10, 6))
        plt.boxplot(scores, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.xlabel('Complexity Score')
        plt.title('Box Plot of Issue Complexity Scores')
        plt.grid(axis='x', alpha=0.75)
        plt.tight_layout()
        plt.show()
    
    def _analyze_complexity_resolution_correlation(self, complexity_scores):
        """
        Analyzes the correlation between complexity and resolution time.
        """
        print("\nAnalyzing correlation between complexity and resolution time...")
        
        # Filter out issues without resolution time
        resolved_issues = [score for score in complexity_scores if score['resolution_time'] is not None]
        
        if len(resolved_issues) < 2:
            print("Not enough resolved issues to analyze correlation.")
            return
        
        # Extract complexity scores and resolution times
        scores = [issue['score'] for issue in resolved_issues]
        resolution_times = [issue['resolution_time'] for issue in resolved_issues]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(scores, resolution_times)[0, 1]
        print(f"Correlation coefficient between complexity and resolution time: {correlation:.4f}")
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(scores, resolution_times, alpha=0.6, edgecolors='w')
        plt.xlabel('Complexity Score')
        plt.ylabel('Resolution Time (days)')
        plt.title(f'Complexity vs. Resolution Time (Correlation: {correlation:.4f})')
        
        # Add trend line
        if len(scores) > 1:
            z = np.polyfit(scores, resolution_times, 1)
            p = np.poly1d(z)
            plt.plot(sorted(scores), p(sorted(scores)), "r--", alpha=0.8)
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Create a grouped bar chart for average resolution time by complexity category
        # Categorize complexity into Low, Medium, High
        df = pd.DataFrame(resolved_issues)
        q1, q3 = np.percentile(df['score'], [25, 75])
        df['complexity_category'] = pd.cut(
            df['score'], 
            bins=[0, q1, q3, float('inf')], 
            labels=['Low', 'Medium', 'High']
        )
        
        # Calculate average resolution time by category
        avg_resolution_by_category = df.groupby('complexity_category')['resolution_time'].mean()
        
        plt.figure(figsize=(10, 6))
        avg_resolution_by_category.plot(kind='bar', color=['green', 'orange', 'red'])
        plt.xlabel('Complexity Category')
        plt.ylabel('Average Resolution Time (days)')
        plt.title('Average Resolution Time by Complexity Category')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _identify_complex_issues_by_label(self, complexity_scores):
        """
        Identifies the most complex issues by label.
        """
        print("\nIdentifying most complex issues by label...")
        
        # Group issues by label
        label_issues = defaultdict(list)
        for score in complexity_scores:
            for label in score['labels']:
                label_issues[label].append(score)
        
        # Calculate average complexity by label (for labels with at least 3 issues)
        label_complexity = {}
        for label, issues in label_issues.items():
            if len(issues) >= 3:
                avg_complexity = sum(issue['score'] for issue in issues) / len(issues)
                label_complexity[label] = {
                    'avg_complexity': avg_complexity,
                    'issue_count': len(issues)
                }
        
        # Sort labels by average complexity
        sorted_labels = sorted(label_complexity.items(), key=lambda x: x[1]['avg_complexity'], reverse=True)
        
        # Display top 10 most complex labels
        top_n = min(10, len(sorted_labels))
        print(f"\nTop {top_n} labels by average complexity:")
        for i, (label, data) in enumerate(sorted_labels[:top_n], 1):
            print(f"{i}. {label}: {data['avg_complexity']:.2f} (from {data['issue_count']} issues)")
        
        # Plot top labels by complexity
        plt.figure(figsize=(12, 8))
        labels = [label for label, _ in sorted_labels[:top_n]]
        avg_complexities = [data['avg_complexity'] for _, data in sorted_labels[:top_n]]
        issue_counts = [data['issue_count'] for _, data in sorted_labels[:top_n]]
        
        # Create horizontal bar chart
        bars = plt.barh(labels, avg_complexities, color='skyblue')
        
        # Add issue count annotations
        for i, (bar, count) in enumerate(zip(bars, issue_counts)):
            plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'n={count}', va='center')
        
        plt.xlabel('Average Complexity Score')
        plt.title('Average Issue Complexity by Label')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Identify the most complex individual issues
        top_issues = sorted(complexity_scores, key=lambda x: x['score'], reverse=True)[:10]
        print("\nTop 10 most complex individual issues:")
        for i, issue in enumerate(top_issues, 1):
            print(f"{i}. Issue #{issue['issue_number']} (Score: {issue['score']:.2f}): {issue['title']}")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueComplexityAnalysis().run()
