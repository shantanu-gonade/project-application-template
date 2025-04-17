"""
Implements an analysis of GitHub issues based on complexity.
"""

from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from src.analysis.base import BaseAnalysis
from src.core.model import Issue, Event, State
from src.utils.helpers import (
    get_issue_resolution_time,
    get_issue_comment_count,
    get_issue_unique_participants,
    has_code_blocks,
    calculate_average,
    calculate_median
)
from src.visualization.plotting import (
    create_histogram,
    create_scatter_plot,
    create_bar_chart,
    create_box_plot
)
from src.visualization.report import Report

class IssueComplexityAnalysis(BaseAnalysis):
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
        super().__init__("complexity_analysis")
        self.results = {}
    
    def analyze(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Performs the analysis on the issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of analysis results
        """
        print("=== Issue Complexity Analysis ===")
        
        # Calculate complexity scores
        complexity_scores = self._calculate_complexity_scores(issues)
        
        # Analyze correlation with resolution time
        correlation_results = self._analyze_complexity_resolution_correlation(complexity_scores)
        
        # Analyze complexity by label
        label_complexity = self._analyze_complexity_by_label(complexity_scores)
        
        # Store results
        self.results = {
            'complexity_scores': complexity_scores,
            'correlation_results': correlation_results,
            'label_complexity': label_complexity
        }
        
        return self.results
    
    def visualize(self, issues: List[Issue]):
        """
        Creates visualizations of the analysis results.
        
        Args:
            issues: List of Issue objects
        """
        # Visualize complexity distribution
        self._visualize_complexity_distribution()
        
        # Visualize correlation with resolution time
        self._visualize_complexity_resolution_correlation()
        
        # Visualize complexity by label
        self._visualize_complexity_by_label()
    
    def save_results(self):
        """
        Saves the analysis results to files.
        """
        # Save results as JSON
        # Convert complexity scores to serializable format
        serializable_scores = []
        for score in self.results['complexity_scores']:
            serializable_score = {k: v for k, v in score.items()}
            # Convert state enum to string
            if 'state' in serializable_score:
                serializable_score['state'] = str(serializable_score['state'])
            serializable_scores.append(serializable_score)
        
        self.save_json(serializable_scores, "complexity_scores")
        self.save_json(self.results['correlation_results'], "correlation_results")
        self.save_json(self.results['label_complexity'], "label_complexity")
        
        # Generate and save report
        self._generate_report()
    
    def _calculate_complexity_scores(self, issues: List[Issue]) -> List[Dict[str, Any]]:
        """
        Calculates complexity scores for each issue.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            List of dictionaries with complexity scores
        """
        print("\n=== Calculating Complexity Scores ===")
        
        complexity_scores = []
        
        for issue in issues:
            # Count comments
            comment_count = get_issue_comment_count(issue)
            
            # Calculate discussion duration
            comment_events = [e for e in issue.events if e.event_type == "commented" and e.event_date]
            discussion_duration = 0
            if comment_events and len(comment_events) > 1:
                first_comment = min(comment_events, key=lambda e: e.event_date).event_date
                last_comment = max(comment_events, key=lambda e: e.event_date).event_date
                discussion_duration = (last_comment - first_comment).total_seconds() / 86400  # Convert to days
            
            # Count unique participants
            participants = get_issue_unique_participants(issue)
            
            # Check for code blocks in issue body and comments
            has_code = has_code_blocks(issue.text)
            for event in issue.events:
                if event.event_type == "commented" and event.comment:
                    if has_code_blocks(event.comment):
                        has_code = True
                        break
            
            # Calculate complexity score (weighted sum)
            score = (
                comment_count * 0.5 + 
                len(participants) * 1.0 + 
                (5.0 if has_code else 0) +
                min(discussion_duration * 0.1, 5.0)  # Cap duration contribution at 5.0
            )
            
            # Calculate resolution time (if issue is closed)
            resolution_time = get_issue_resolution_time(issue)
            
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
        
        return complexity_scores
    
    def _analyze_complexity_resolution_correlation(self, complexity_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes the correlation between complexity and resolution time.
        
        Args:
            complexity_scores: List of dictionaries with complexity scores
            
        Returns:
            Dictionary with correlation results
        """
        print("\n=== Analyzing Correlation Between Complexity and Resolution Time ===")
        
        # Filter out issues without resolution time
        resolved_issues = [score for score in complexity_scores if score['resolution_time'] is not None]
        
        if len(resolved_issues) < 2:
            print("Not enough resolved issues to analyze correlation")
            return {
                'correlation': None,
                'resolved_issues_count': len(resolved_issues)
            }
        
        # Extract complexity scores and resolution times
        scores = [issue['score'] for issue in resolved_issues]
        resolution_times = [issue['resolution_time'] for issue in resolved_issues]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(scores, resolution_times)[0, 1]
        print(f"Correlation coefficient between complexity and resolution time: {correlation:.4f}")
        
        # Categorize complexity into Low, Medium, High
        q1, q3 = np.percentile(scores, [25, 75])
        
        categories = []
        for score in scores:
            if score <= q1:
                categories.append('Low')
            elif score <= q3:
                categories.append('Medium')
            else:
                categories.append('High')
        
        # Calculate average resolution time by category
        category_times = defaultdict(list)
        for category, time in zip(categories, resolution_times):
            category_times[category].append(time)
        
        avg_resolution_by_category = {
            category: sum(times) / len(times) 
            for category, times in category_times.items()
        }
        
        print("Average resolution time by complexity category:")
        for category, avg_time in avg_resolution_by_category.items():
            print(f"  {category}: {avg_time:.2f} days")
        
        return {
            'correlation': correlation,
            'resolved_issues_count': len(resolved_issues),
            'scores': scores,
            'resolution_times': resolution_times,
            'categories': categories,
            'avg_resolution_by_category': avg_resolution_by_category
        }
    
    def _analyze_complexity_by_label(self, complexity_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes complexity by label.
        
        Args:
            complexity_scores: List of dictionaries with complexity scores
            
        Returns:
            Dictionary with label complexity results
        """
        print("\n=== Analyzing Complexity by Label ===")
        
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
                    'issue_count': len(issues),
                    'issues': issues
                }
        
        # Sort labels by average complexity
        sorted_labels = sorted(label_complexity.items(), key=lambda x: x[1]['avg_complexity'], reverse=True)
        
        # Display top 10 most complex labels
        top_n = min(10, len(sorted_labels))
        print(f"\nTop {top_n} labels by average complexity:")
        for i, (label, data) in enumerate(sorted_labels[:top_n], 1):
            print(f"{i}. {label}: {data['avg_complexity']:.2f} (from {data['issue_count']} issues)")
        
        return {
            'label_complexity': {label: {
                'avg_complexity': data['avg_complexity'],
                'issue_count': data['issue_count']
            } for label, data in label_complexity.items()},
            'sorted_labels': [(label, data['avg_complexity'], data['issue_count']) 
                             for label, data in sorted_labels]
        }
    
    def _visualize_complexity_distribution(self):
        """
        Creates visualizations of the complexity score distribution.
        """
        # Extract scores
        scores = [score['score'] for score in self.results['complexity_scores']]
        
        # Create histogram
        fig = create_histogram(
            data=scores,
            title='Distribution of Issue Complexity Scores',
            xlabel='Complexity Score',
            ylabel='Number of Issues',
            bins=20,
            show_mean=True,
            show_median=True
        )
        
        self.save_figure(fig, "issue_complexity_scores_histogram")
        
        # Create box plot
        fig = create_box_plot(
            data=scores,
            title='Box Plot of Issue Complexity Scores',
            xlabel='Complexity Score',
            vertical=False
        )
        
        self.save_figure(fig, "issue_complexity_scores_box_plot")
        
        # Create visualizations for complexity components
        comment_counts = [score['comment_count'] for score in self.results['complexity_scores']]
        participant_counts = [score['participant_count'] for score in self.results['complexity_scores']]
        discussion_durations = [score['discussion_duration'] for score in self.results['complexity_scores']]
        
        # Create histograms for each component
        fig = plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Complexity Scores')
        plt.xlabel('Score')
        plt.ylabel('Number of Issues')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(comment_counts, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Comment Counts')
        plt.xlabel('Number of Comments')
        plt.ylabel('Number of Issues')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(participant_counts, bins=max(participant_counts), alpha=0.7, color='green', edgecolor='black')
        plt.title('Participant Counts')
        plt.xlabel('Number of Participants')
        plt.ylabel('Number of Issues')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.hist([d for d in discussion_durations if d > 0], bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.title('Discussion Durations')
        plt.xlabel('Duration (days)')
        plt.ylabel('Number of Issues')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        self.save_figure(fig, "complexity_components")
    
    def _visualize_complexity_resolution_correlation(self):
        """
        Creates visualizations of the correlation between complexity and resolution time.
        """
        correlation_results = self.results['correlation_results']
        
        # Skip if not enough data
        if correlation_results['correlation'] is None:
            print("Not enough data to visualize complexity-resolution correlation")
            return
        
        # Create scatter plot
        fig = create_scatter_plot(
            x=correlation_results['scores'],
            y=correlation_results['resolution_times'],
            title=f'Complexity vs. Resolution Time (Correlation: {correlation_results["correlation"]:.4f})',
            xlabel='Complexity Score',
            ylabel='Resolution Time (days)',
            add_trendline=True
        )
        
        self.save_figure(fig, "complexity_vs_resolution_time")
        
        # Create bar chart for average resolution time by category
        categories = list(correlation_results['avg_resolution_by_category'].keys())
        avg_times = list(correlation_results['avg_resolution_by_category'].values())
        
        fig = create_bar_chart(
            labels=categories,
            values=avg_times,
            title='Average Resolution Time by Complexity Category',
            xlabel='Complexity Category',
            ylabel='Average Resolution Time (days)',
            color=['green', 'orange', 'red']
        )
        
        self.save_figure(fig, "avg_resolution_time_vs_complexity_category")
    
    def _visualize_complexity_by_label(self):
        """
        Creates visualizations of the complexity by label.
        """
        label_complexity = self.results['label_complexity']
        
        # Skip if not enough data
        if not label_complexity['sorted_labels']:
            print("Not enough data to visualize complexity by label")
            return
        
        # Get top 10 labels by complexity
        top_labels = label_complexity['sorted_labels'][:10]
        
        # Create horizontal bar chart
        labels = [label for label, _, _ in top_labels]
        complexities = [complexity for _, complexity, _ in top_labels]
        counts = [count for _, _, count in top_labels]
        
        fig = plt.figure(figsize=(12, 8))
        
        bars = plt.barh(labels, complexities, color='skyblue')
        
        # Add issue count annotations
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                    f'n={count}', va='center')
        
        plt.xlabel('Average Complexity Score')
        plt.title('Average Issue Complexity by Label')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(fig, "avg_issue_complexity_vs_label")
    
    def _generate_report(self):
        """
        Generates a report of the analysis results.
        """
        # Create report
        report = Report("Issue Complexity Analysis Report", self.results_dir)
        
        # Add introduction
        intro = "Analysis of issue complexity based on various metrics"
        report.add_section("Introduction", intro)
        
        # Add complexity distribution section
        scores = [score['score'] for score in self.results['complexity_scores']]
        
        distribution_content = "Complexity score distribution:\n\n"
        distribution_content += f"Number of issues analyzed: {len(scores)}\n"
        distribution_content += f"Average complexity score: {np.mean(scores):.2f}\n"
        distribution_content += f"Median complexity score: {np.median(scores):.2f}\n"
        distribution_content += f"Minimum complexity score: {min(scores):.2f}\n"
        distribution_content += f"Maximum complexity score: {max(scores):.2f}\n"
        
        report.add_section("Complexity Score Distribution", distribution_content)
        
        # Add correlation section
        correlation_results = self.results['correlation_results']
        
        correlation_content = "Correlation between complexity and resolution time:\n\n"
        
        if correlation_results['correlation'] is not None:
            correlation_content += f"Correlation coefficient: {correlation_results['correlation']:.4f}\n"
            correlation_content += f"Number of resolved issues analyzed: {correlation_results['resolved_issues_count']}\n\n"
            
            correlation_content += "Average resolution time by complexity category:\n\n"
            for category, avg_time in correlation_results['avg_resolution_by_category'].items():
                correlation_content += f"{category} complexity: {avg_time:.2f} days\n"
        else:
            correlation_content += "Not enough resolved issues to analyze correlation"
        
        report.add_section("Complexity-Resolution Correlation", correlation_content)
        
        # Add label complexity section
        label_complexity = self.results['label_complexity']
        
        label_content = "Complexity by label:\n\n"
        
        if label_complexity['sorted_labels']:
            label_content += "Top 10 labels by average complexity:\n\n"
            for i, (label, complexity, count) in enumerate(label_complexity['sorted_labels'][:10], 1):
                label_content += f"{i}. {label}: {complexity:.2f} (from {count} issues)\n"
            
            label_content += "\nBottom 10 labels by average complexity:\n\n"
            for i, (label, complexity, count) in enumerate(label_complexity['sorted_labels'][-10:], 1):
                label_content += f"{i}. {label}: {complexity:.2f} (from {count} issues)\n"
        else:
            label_content += "Not enough labeled issues to analyze complexity by label"
        
        report.add_section("Complexity by Label", label_content)
        
        # Add most complex issues section
        top_issues = sorted(self.results['complexity_scores'], key=lambda x: x['score'], reverse=True)[:10]
        
        issues_content = "Top 10 most complex issues:\n\n"
        for i, issue in enumerate(top_issues, 1):
            issues_content += f"{i}. Issue #{issue['issue_number']} (Score: {issue['score']:.2f}): {issue['title']}\n"
            issues_content += f"   Comments: {issue['comment_count']}, Participants: {issue['participant_count']}, "
            issues_content += f"Duration: {issue['discussion_duration']:.1f} days, Has Code: {'Yes' if issue['has_code'] else 'No'}\n"
        
        report.add_section("Most Complex Issues", issues_content)
        
        # Save report
        report.save_text_report("results.txt")
        report.save_markdown_report("report.md")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueComplexityAnalysis().run()
