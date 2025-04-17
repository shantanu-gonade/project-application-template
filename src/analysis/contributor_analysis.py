"""
Implements an analysis of GitHub issues based on contributors.
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
    filter_issues_by_user,
    get_issue_resolution_time,
    get_issue_comment_count,
    get_issue_first_response_time,
    get_issue_unique_participants,
    calculate_average
)
from src.visualization.plotting import (
    create_bar_chart,
    create_histogram,
    create_pie_chart
)
from src.visualization.report import Report

class ContributorAnalysis(BaseAnalysis):
    """
    Implements an analysis of GitHub issues based on contributors.
    This analysis focuses on:
    1. Top contributors by different metrics
    2. Contributor label focus
    3. Contributor activity patterns
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__("contributor_analysis")
        self.results = {}
    
    def analyze(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Performs the analysis on the issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of analysis results
        """
        print("=== Contributor Analysis ===")
        
        # Identify top contributors
        top_contributors = self._identify_top_contributors(issues)
        
        # Analyze contributor label focus
        contributor_label_focus = self._analyze_contributor_label_focus(issues)
        
        # Analyze contributor activity patterns
        activity_patterns = self._analyze_activity_patterns(issues)
        
        # Analyze response times
        response_times = self._analyze_response_times(issues)
        
        # Store results
        self.results = {
            'top_contributors': top_contributors,
            'contributor_label_focus': contributor_label_focus,
            'activity_patterns': activity_patterns,
            'response_times': response_times
        }
        
        return self.results
    
    def visualize(self, issues: List[Issue]):
        """
        Creates visualizations of the analysis results.
        
        Args:
            issues: List of Issue objects
        """
        # Visualize top contributors
        self._visualize_top_contributors()
        
        # Visualize contributor label focus
        self._visualize_contributor_label_focus()
        
        # Visualize activity patterns
        self._visualize_activity_patterns()
        
        # Visualize response times
        self._visualize_response_times()
    
    def save_results(self):
        """
        Saves the analysis results to files.
        """
        # Save results as JSON
        for key, value in self.results.items():
            self.save_json(value, key)
        
        # Generate and save report
        self._generate_report()
    
    def _identify_top_contributors(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Identifies top contributors by different metrics.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of top contributors by different metrics
        """
        print("\n=== Top Contributors ===")
        
        # Count issue creation by user
        issue_creators = {}
        for issue in issues:
            if issue.creator in issue_creators:
                issue_creators[issue.creator] += 1
            else:
                issue_creators[issue.creator] = 1
        
        # Count comments by user
        commenters = {}
        for issue in issues:
            for event in issue.events:
                if event.event_type == "commented" and event.author:
                    if event.author in commenters:
                        commenters[event.author] += 1
                    else:
                        commenters[event.author] = 1
        
        # Count issue closures by user
        issue_closers = {}
        for issue in issues:
            if issue.state == State.closed:
                for event in issue.events:
                    if event.event_type == "closed" and event.author:
                        if event.author in issue_closers:
                            issue_closers[event.author] += 1
                        else:
                            issue_closers[event.author] = 1
        
        # Sort by count (descending)
        sorted_creators = sorted(issue_creators.items(), key=lambda x: x[1], reverse=True)
        sorted_commenters = sorted(commenters.items(), key=lambda x: x[1], reverse=True)
        sorted_closers = sorted(issue_closers.items(), key=lambda x: x[1], reverse=True)
        
        # Print top 10 contributors
        print("\nTop 10 issue creators:")
        for i, (user, count) in enumerate(sorted_creators[:10], 1):
            print(f"{i}. {user}: {count} issues")
        
        print("\nTop 10 commenters:")
        for i, (user, count) in enumerate(sorted_commenters[:10], 1):
            print(f"{i}. {user}: {count} comments")
        
        print("\nTop 10 issue closers:")
        for i, (user, count) in enumerate(sorted_closers[:10], 1):
            print(f"{i}. {user}: {count} issues closed")
        
        return {
            'top_issue_creators': dict(sorted_creators[:10]),
            'top_commenters': dict(sorted_commenters[:10]),
            'top_issue_closers': dict(sorted_closers[:10])
        }
    
    def _analyze_contributor_label_focus(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes which labels different contributors focus on.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary mapping contributors to their label focus
        """
        print("\n=== Contributor Label Focus ===")
        
        # Get top contributors (by issue creation)
        issue_creators = {}
        for issue in issues:
            if issue.creator in issue_creators:
                issue_creators[issue.creator] += 1
            else:
                issue_creators[issue.creator] = 1
        
        sorted_creators = sorted(issue_creators.items(), key=lambda x: x[1], reverse=True)
        top_creators = [user for user, _ in sorted_creators[:5]]
        
        # Analyze label focus for top contributors
        contributor_label_focus = {}
        
        for user in top_creators:
            # Get issues created by this user
            user_issues = filter_issues_by_user(issues, user)
            
            # Count labels
            label_counts = {}
            for issue in user_issues:
                for label in issue.labels:
                    if label in label_counts:
                        label_counts[label] += 1
                    else:
                        label_counts[label] = 1
            
            # Sort by count (descending)
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Store top 5 labels
            contributor_label_focus[user] = dict(sorted_labels[:5])
            
            # Print results
            print(f"\nTop 5 labels for {user}:")
            for i, (label, count) in enumerate(sorted_labels[:5], 1):
                print(f"{i}. {label}: {count} issues")
        
        return contributor_label_focus
    
    def _analyze_activity_patterns(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes contributor activity patterns.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of activity patterns
        """
        print("\n=== Activity Patterns ===")
        
        # Create timeline of issue creation
        timeline = []
        for issue in issues:
            if issue.created_date:
                timeline.append({
                    'date': issue.created_date,
                    'user': issue.creator,
                    'action': 'created'
                })
        
        # Add comments to timeline
        for issue in issues:
            for event in issue.events:
                if event.event_type == "commented" and event.author and event.event_date:
                    timeline.append({
                        'date': event.event_date,
                        'user': event.author,
                        'action': 'commented'
                    })
        
        # Add closures to timeline
        for issue in issues:
            for event in issue.events:
                if event.event_type == "closed" and event.author and event.event_date:
                    timeline.append({
                        'date': event.event_date,
                        'user': event.author,
                        'action': 'closed'
                    })
        
        # Sort timeline by date
        timeline.sort(key=lambda x: x['date'])
        
        # Group by month
        activity_by_month = defaultdict(lambda: defaultdict(int))
        for event in timeline:
            month = event['date'].strftime('%Y-%m')
            activity_by_month[month][event['user']] += 1
        
        # Get top contributors overall
        user_activity = defaultdict(int)
        for event in timeline:
            user_activity[event['user']] += 1
        
        sorted_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)
        top_users = [user for user, _ in sorted_users[:5]]
        
        # Extract activity for top users by month
        activity_timeline = {}
        for user in top_users:
            activity_timeline[user] = []
            for month in sorted(activity_by_month.keys()):
                activity_timeline[user].append({
                    'month': month,
                    'count': activity_by_month[month][user]
                })
        
        print(f"\nAnalyzed activity timeline for top {len(top_users)} contributors")
        
        return {
            'activity_timeline': activity_timeline,
            'top_users': top_users
        }
    
    def _analyze_response_times(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes response times for different contributors.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of response times
        """
        print("\n=== Response Times ===")
        
        # Get top contributors (by issue creation)
        issue_creators = {}
        for issue in issues:
            if issue.creator in issue_creators:
                issue_creators[issue.creator] += 1
            else:
                issue_creators[issue.creator] = 1
        
        sorted_creators = sorted(issue_creators.items(), key=lambda x: x[1], reverse=True)
        top_creators = [user for user, _ in sorted_creators[:5]]
        
        # Calculate average time to first response for each top creator
        response_times = {}
        for user in top_creators:
            # Get issues created by this user
            user_issues = filter_issues_by_user(issues, user)
            
            # Calculate time to first response
            times = []
            for issue in user_issues:
                time = get_issue_first_response_time(issue)
                if time is not None:
                    times.append(time)
            
            # Calculate average
            if times:
                avg_time = sum(times) / len(times)
                response_times[user] = {
                    'avg_time': avg_time,
                    'issue_count': len(times)
                }
                print(f"Average time to first response for {user}: {avg_time:.2f} days ({len(times)} issues)")
            else:
                response_times[user] = {
                    'avg_time': None,
                    'issue_count': 0
                }
                print(f"No response time data for {user}")
        
        return response_times
    
    def _visualize_top_contributors(self):
        """
        Creates visualizations of the top contributors.
        """
        # Visualize top issue creators
        top_creators = self.results['top_contributors']['top_issue_creators']
        fig = create_bar_chart(
            labels=list(top_creators.keys()),
            values=list(top_creators.values()),
            title='Top Issue Creators',
            xlabel='User',
            ylabel='Number of Issues Created',
            color='skyblue',
            rotation=45
        )
        self.save_figure(fig, "top_issue_creators")
        
        # Visualize top commenters
        top_commenters = self.results['top_contributors']['top_commenters']
        fig = create_bar_chart(
            labels=list(top_commenters.keys()),
            values=list(top_commenters.values()),
            title='Top Commenters',
            xlabel='User',
            ylabel='Number of Comments',
            color='orange',
            rotation=45
        )
        self.save_figure(fig, "top_commenters")
        
        # Visualize top issue closers
        top_closers = self.results['top_contributors']['top_issue_closers']
        fig = create_bar_chart(
            labels=list(top_closers.keys()),
            values=list(top_closers.values()),
            title='Top Issue Closers',
            xlabel='User',
            ylabel='Number of Issues Closed',
            color='green',
            rotation=45
        )
        self.save_figure(fig, "top_issue_closers")
    
    def _visualize_contributor_label_focus(self):
        """
        Creates visualizations of the contributor label focus.
        """
        # Create a heatmap-like visualization for each top contributor
        for user, labels in self.results['contributor_label_focus'].items():
            fig = create_bar_chart(
                labels=list(labels.keys()),
                values=list(labels.values()),
                title=f'Label Focus for {user}',
                xlabel='Label',
                ylabel='Number of Issues',
                color='purple',
                rotation=45
            )
            self.save_figure(fig, f"{user.lower().replace(' ', '_')}_label_focus")
        
        # Create a combined visualization
        fig = plt.figure(figsize=(12, 8))
        
        # Get all users and their label focus
        users = list(self.results['contributor_label_focus'].keys())
        
        # Create a heatmap-like visualization
        for i, user in enumerate(users):
            labels = self.results['contributor_label_focus'][user]
            plt.bar(
                [f"{label} ({user})" for label in labels.keys()],
                list(labels.values()),
                label=user
            )
        
        plt.title('Label Focus by Contributor')
        plt.xlabel('Label (User)')
        plt.ylabel('Number of Issues')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        self.save_figure(fig, "contributor_label_heatmap")
    
    def _visualize_activity_patterns(self):
        """
        Creates visualizations of the activity patterns.
        """
        # Create a line chart for each top user
        activity_timeline = self.results['activity_patterns']['activity_timeline']
        top_users = self.results['activity_patterns']['top_users']
        
        fig = plt.figure(figsize=(12, 6))
        
        for user in top_users:
            timeline = activity_timeline[user]
            months = [item['month'] for item in timeline]
            counts = [item['count'] for item in timeline]
            
            plt.plot(months, counts, marker='o', label=user)
        
        plt.title('Activity Timeline by Contributor')
        plt.xlabel('Month')
        plt.ylabel('Number of Actions')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(fig, "activity_timeline")
    
    def _visualize_response_times(self):
        """
        Creates visualizations of the response times.
        """
        # Create a bar chart of average response times
        response_times = self.results['response_times']
        
        # Filter out users with no response time data
        users_with_data = {user: data for user, data in response_times.items() 
                          if data['avg_time'] is not None}
        
        if users_with_data:
            fig = plt.figure(figsize=(10, 6))
            
            users = list(users_with_data.keys())
            avg_times = [data['avg_time'] for data in users_with_data.values()]
            issue_counts = [data['issue_count'] for data in users_with_data.values()]
            
            # Create bar chart
            bars = plt.bar(users, avg_times, color='skyblue')
            
            # Add issue count annotations
            for i, (bar, count) in enumerate(zip(bars, issue_counts)):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f'n={count}',
                    ha='center',
                    va='bottom'
                )
            
            plt.title('Average Time to First Response by Contributor')
            plt.xlabel('User')
            plt.ylabel('Average Time (days)')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            self.save_figure(fig, "response_times")
            
            # If the user parameter is specified, create a more detailed visualization
            if self.user and self.user in response_times:
                # Get issues created by this user
                user_issues = filter_issues_by_user(self.load_data(), self.user)
                
                # Calculate time to first response for each issue
                response_times_list = []
                for issue in user_issues:
                    time = get_issue_first_response_time(issue)
                    if time is not None:
                        response_times_list.append(time)
                
                if response_times_list:
                    fig = create_histogram(
                        data=response_times_list,
                        title=f'Response Times for Issues Created by {self.user}',
                        xlabel='Time to First Response (days)',
                        ylabel='Number of Issues',
                        bins=20,
                        show_mean=True,
                        show_median=True
                    )
                    
                    self.save_figure(fig, f"{self.user.lower().replace(' ', '_')}_response_times")
    
    def _generate_report(self):
        """
        Generates a report of the analysis results.
        """
        # Create report
        report = Report("Contributor Analysis Report", self.results_dir)
        
        # Add introduction
        if self.user:
            intro = f"Analysis of issues created by '{self.user}'"
        else:
            intro = "Analysis of contributor activity across all issues"
        
        report.add_section("Introduction", intro)
        
        # Add top contributors section
        top_creators = self.results['top_contributors']['top_issue_creators']
        top_commenters = self.results['top_contributors']['top_commenters']
        top_closers = self.results['top_contributors']['top_issue_closers']
        
        contributors_content = "Top 10 issue creators:\n\n"
        for i, (user, count) in enumerate(top_creators.items(), 1):
            contributors_content += f"{i}. {user}: {count} issues\n"
        
        contributors_content += "\nTop 10 commenters:\n\n"
        for i, (user, count) in enumerate(top_commenters.items(), 1):
            contributors_content += f"{i}. {user}: {count} comments\n"
        
        contributors_content += "\nTop 10 issue closers:\n\n"
        for i, (user, count) in enumerate(top_closers.items(), 1):
            contributors_content += f"{i}. {user}: {count} issues closed\n"
        
        report.add_section("Top Contributors", contributors_content)
        
        # Add contributor label focus section
        label_focus_content = "Label focus for top contributors:\n\n"
        for user, labels in self.results['contributor_label_focus'].items():
            label_focus_content += f"Top 5 labels for {user}:\n\n"
            for i, (label, count) in enumerate(labels.items(), 1):
                label_focus_content += f"{i}. {label}: {count} issues\n"
            label_focus_content += "\n"
        
        report.add_section("Contributor Label Focus", label_focus_content)
        
        # Add response times section
        response_times = self.results['response_times']
        response_times_content = "Average time to first response by contributor:\n\n"
        
        for user, data in response_times.items():
            if data['avg_time'] is not None:
                response_times_content += f"{user}: {data['avg_time']:.2f} days ({data['issue_count']} issues)\n"
            else:
                response_times_content += f"{user}: No response time data\n"
        
        report.add_section("Response Times", response_times_content)
        
        # Add activity patterns section
        activity_content = "Activity patterns for top contributors:\n\n"
        activity_content += "See the activity_timeline.png visualization for a graphical representation of activity over time.\n"
        
        report.add_section("Activity Patterns", activity_content)
        
        # Save report
        report.save_text_report("overall_analysis_report.txt")
        
        # If user parameter is specified, generate a specific report for that user
        if self.user and self.user in self.results['contributor_label_focus']:
            user_report = Report(f"Analysis Report for {self.user}", self.results_dir)
            
            user_report.add_section("Introduction", f"Detailed analysis of issues created by {self.user}")
            
            # Add label focus section
            labels = self.results['contributor_label_focus'][self.user]
            label_content = f"Top 5 labels for {self.user}:\n\n"
            for i, (label, count) in enumerate(labels.items(), 1):
                label_content += f"{i}. {label}: {count} issues\n"
            
            user_report.add_section("Label Focus", label_content)
            
            # Add response time section
            if self.user in response_times and response_times[self.user]['avg_time'] is not None:
                data = response_times[self.user]
                response_content = f"Average time to first response: {data['avg_time']:.2f} days ({data['issue_count']} issues)\n"
            else:
                response_content = "No response time data available"
            
            user_report.add_section("Response Times", response_content)
            
            # Save user-specific report
            user_report.save_text_report(f"{self.user.lower().replace(' ', '_')}_analysis_report.txt")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    ContributorAnalysis().run()
