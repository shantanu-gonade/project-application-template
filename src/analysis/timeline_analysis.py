"""
Implements an analysis of GitHub issues based on timeline.
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
    get_issue_first_response_time,
    group_by_time_period,
    extract_date_components
)
from src.visualization.plotting import (
    create_line_chart,
    create_bar_chart,
    create_histogram
)
from src.visualization.report import Report

class TimelineAnalysis(BaseAnalysis):
    """
    Implements an analysis of GitHub issues based on timeline.
    This analysis focuses on:
    1. Issue creation and closing patterns over time
    2. Issue lifecycle (time to first response, time to resolution)
    3. Activity trends and patterns (by day of week, hour of day)
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__("timeline_analysis")
        self.results = {}
    
    def analyze(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Performs the analysis on the issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of analysis results
        """
        print("=== Timeline Analysis ===")
        
        # Analyze issue creation and closing patterns
        creation_closing_patterns = self._analyze_creation_closing_patterns(issues)
        
        # Analyze issue lifecycle
        issue_lifecycle = self._analyze_issue_lifecycle(issues)
        
        # Analyze activity trends
        activity_trends = self._analyze_activity_trends(issues)
        
        # Store results
        self.results = {
            'creation_closing_patterns': creation_closing_patterns,
            'issue_lifecycle': issue_lifecycle,
            'activity_trends': activity_trends
        }
        
        return self.results
    
    def visualize(self, issues: List[Issue]):
        """
        Creates visualizations of the analysis results.
        
        Args:
            issues: List of Issue objects
        """
        # Visualize issue creation and closing patterns
        self._visualize_creation_closing_patterns()
        
        # Visualize issue lifecycle
        self._visualize_issue_lifecycle()
        
        # Visualize activity trends
        self._visualize_activity_trends()
    
    def save_results(self):
        """
        Saves the analysis results to files.
        """
        # Save results as JSON
        for key, value in self.results.items():
            self.save_json(value, key)
        
        # Generate and save report
        self._generate_report()
    
    def _analyze_creation_closing_patterns(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes issue creation and closing patterns over time.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of creation and closing patterns
        """
        print("\n=== Issue Creation and Closing Patterns ===")
        
        # Extract creation dates
        creation_dates = [issue.created_date for issue in issues if issue.created_date]
        
        # Extract closing dates
        closing_dates = []
        for issue in issues:
            if issue.state == State.closed:
                for event in issue.events:
                    if event.event_type == "closed" and event.event_date:
                        closing_dates.append(event.event_date)
                        break
        
        # Group by month
        creation_by_month = group_by_time_period(creation_dates, 'month')
        closing_by_month = group_by_time_period(closing_dates, 'month')
        
        # Ensure all months are in both dictionaries
        all_months = sorted(set(list(creation_by_month.keys()) + list(closing_by_month.keys())))
        for month in all_months:
            if month not in creation_by_month:
                creation_by_month[month] = 0
            if month not in closing_by_month:
                closing_by_month[month] = 0
        
        # Sort by month
        creation_by_month = {month: creation_by_month[month] for month in sorted(creation_by_month.keys())}
        closing_by_month = {month: closing_by_month[month] for month in sorted(closing_by_month.keys())}
        
        # Calculate cumulative counts
        cumulative_creation = {}
        cumulative_closing = {}
        
        creation_count = 0
        closing_count = 0
        
        for month in all_months:
            creation_count += creation_by_month[month]
            closing_count += closing_by_month[month]
            
            cumulative_creation[month] = creation_count
            cumulative_closing[month] = closing_count
        
        # Calculate open issues over time
        open_issues = {}
        for month in all_months:
            open_issues[month] = cumulative_creation[month] - cumulative_closing[month]
        
        print(f"Analyzed issue creation and closing patterns across {len(all_months)} months")
        
        return {
            'creation_by_month': creation_by_month,
            'closing_by_month': closing_by_month,
            'cumulative_creation': cumulative_creation,
            'cumulative_closing': cumulative_closing,
            'open_issues': open_issues
        }
    
    def _analyze_issue_lifecycle(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes issue lifecycle (time to first response, time to resolution).
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of issue lifecycle metrics
        """
        print("\n=== Issue Lifecycle ===")
        
        # Calculate time to first response
        first_response_times = []
        for issue in issues:
            time = get_issue_first_response_time(issue)
            if time is not None:
                first_response_times.append(time)
        
        # Calculate time to resolution
        resolution_times = []
        for issue in issues:
            time = get_issue_resolution_time(issue)
            if time is not None:
                resolution_times.append(time)
        
        # Calculate statistics
        if first_response_times:
            avg_first_response = sum(first_response_times) / len(first_response_times)
            median_first_response = sorted(first_response_times)[len(first_response_times) // 2]
            print(f"Average time to first response: {avg_first_response:.2f} days")
            print(f"Median time to first response: {median_first_response:.2f} days")
        else:
            avg_first_response = None
            median_first_response = None
            print("No first response time data available")
        
        if resolution_times:
            avg_resolution = sum(resolution_times) / len(resolution_times)
            median_resolution = sorted(resolution_times)[len(resolution_times) // 2]
            print(f"Average time to resolution: {avg_resolution:.2f} days")
            print(f"Median time to resolution: {median_resolution:.2f} days")
        else:
            avg_resolution = None
            median_resolution = None
            print("No resolution time data available")
        
        return {
            'first_response_times': first_response_times,
            'resolution_times': resolution_times,
            'avg_first_response': avg_first_response,
            'median_first_response': median_first_response,
            'avg_resolution': avg_resolution,
            'median_resolution': median_resolution
        }
    
    def _analyze_activity_trends(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Analyzes activity trends and patterns (by day of week, hour of day).
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of activity trends
        """
        print("\n=== Activity Trends ===")
        
        # Collect all events
        all_events = []
        
        # Add issue creation events
        for issue in issues:
            if issue.created_date:
                all_events.append({
                    'date': issue.created_date,
                    'type': 'created'
                })
        
        # Add comment events
        for issue in issues:
            for event in issue.events:
                if event.event_type == "commented" and event.event_date:
                    all_events.append({
                        'date': event.event_date,
                        'type': 'commented'
                    })
        
        # Add closing events
        for issue in issues:
            for event in issue.events:
                if event.event_type == "closed" and event.event_date:
                    all_events.append({
                        'date': event.event_date,
                        'type': 'closed'
                    })
        
        # Extract day of week and hour of day
        day_of_week_counts = defaultdict(int)
        hour_of_day_counts = defaultdict(int)
        
        for event in all_events:
            date = event['date']
            day_of_week = date.weekday()  # 0 = Monday, 6 = Sunday
            hour_of_day = date.hour
            
            day_of_week_counts[day_of_week] += 1
            hour_of_day_counts[hour_of_day] += 1
        
        # Convert to lists for easier plotting
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = [day_of_week_counts[i] for i in range(7)]
        
        hours = list(range(24))
        hour_counts = [hour_of_day_counts[i] for i in range(24)]
        
        print(f"Analyzed {len(all_events)} events for activity trends")
        
        return {
            'day_of_week': {
                'labels': days,
                'counts': day_counts
            },
            'hour_of_day': {
                'labels': hours,
                'counts': hour_counts
            }
        }
    
    def _visualize_creation_closing_patterns(self):
        """
        Creates visualizations of the issue creation and closing patterns.
        """
        patterns = self.results['creation_closing_patterns']
        
        # Create line chart of issue creation and closing by month
        fig = plt.figure(figsize=(12, 6))
        
        months = list(patterns['creation_by_month'].keys())
        creation_counts = list(patterns['creation_by_month'].values())
        closing_counts = list(patterns['closing_by_month'].values())
        
        plt.plot(months, creation_counts, marker='o', label='Created', color='blue')
        plt.plot(months, closing_counts, marker='x', label='Closed', color='red')
        
        plt.title('Issue Creation and Closure Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Issues')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(fig, "Issue creation and closure over time")
        
        # Create line chart of open issues over time
        fig = plt.figure(figsize=(12, 6))
        
        months = list(patterns['open_issues'].keys())
        open_counts = list(patterns['open_issues'].values())
        
        plt.plot(months, open_counts, marker='o', color='green')
        
        plt.title('Open Issues Over Time')
        plt.xlabel('Month')
        plt.ylabel('Number of Open Issues')
        plt.xticks(rotation=45, ha='right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(fig, "Open issues over time")
    
    def _visualize_issue_lifecycle(self):
        """
        Creates visualizations of the issue lifecycle.
        """
        lifecycle = self.results['issue_lifecycle']
        
        # Create histogram of time to first response
        if lifecycle['first_response_times']:
            fig = create_histogram(
                data=lifecycle['first_response_times'],
                title='Time to First Response',
                xlabel='Days',
                ylabel='Number of Issues',
                bins=20,
                show_mean=True,
                show_median=True
            )
            
            self.save_figure(fig, "Time to first response")
        
        # Create histogram of time to resolution
        if lifecycle['resolution_times']:
            fig = create_histogram(
                data=lifecycle['resolution_times'],
                title='Time to Resolution',
                xlabel='Days',
                ylabel='Number of Issues',
                bins=20,
                show_mean=True,
                show_median=True
            )
            
            self.save_figure(fig, "Time to resolution")
        
        # Create combined visualization
        fig = plt.figure(figsize=(12, 6))
        
        # Add lifecycle metrics
        metrics = []
        values = []
        
        if lifecycle['avg_first_response'] is not None:
            metrics.append('Avg. Time to First Response')
            values.append(lifecycle['avg_first_response'])
        
        if lifecycle['median_first_response'] is not None:
            metrics.append('Median Time to First Response')
            values.append(lifecycle['median_first_response'])
        
        if lifecycle['avg_resolution'] is not None:
            metrics.append('Avg. Time to Resolution')
            values.append(lifecycle['avg_resolution'])
        
        if lifecycle['median_resolution'] is not None:
            metrics.append('Median Time to Resolution')
            values.append(lifecycle['median_resolution'])
        
        if metrics and values:
            plt.bar(metrics, values, color=['blue', 'lightblue', 'red', 'lightcoral'])
            
            plt.title('Issue Lifecycle Metrics')
            plt.ylabel('Days')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            self.save_figure(fig, "Issue lifecycle")
    
    def _visualize_activity_trends(self):
        """
        Creates visualizations of the activity trends.
        """
        trends = self.results['activity_trends']
        
        # Create bar chart of activity by day of week
        fig = create_bar_chart(
            labels=trends['day_of_week']['labels'],
            values=trends['day_of_week']['counts'],
            title='Activity by Day of Week',
            xlabel='Day',
            ylabel='Number of Events',
            color='purple'
        )
        
        self.save_figure(fig, "Activity by day of week")
        
        # Create bar chart of activity by hour of day
        fig = create_bar_chart(
            labels=[str(h) for h in trends['hour_of_day']['labels']],
            values=trends['hour_of_day']['counts'],
            title='Activity by Hour of Day',
            xlabel='Hour',
            ylabel='Number of Events',
            color='orange'
        )
        
        self.save_figure(fig, "Activity by hour of day")
        
        # Create combined visualization
        fig = plt.figure(figsize=(12, 10))
        
        # Activity by day of week
        plt.subplot(2, 1, 1)
        plt.bar(trends['day_of_week']['labels'], trends['day_of_week']['counts'], color='purple')
        plt.title('Activity by Day of Week')
        plt.ylabel('Number of Events')
        plt.grid(axis='y', alpha=0.3)
        
        # Activity by hour of day
        plt.subplot(2, 1, 2)
        plt.bar([str(h) for h in trends['hour_of_day']['labels']], trends['hour_of_day']['counts'], color='orange')
        plt.title('Activity by Hour of Day')
        plt.xlabel('Hour')
        plt.ylabel('Number of Events')
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        self.save_figure(fig, "Activity trends")
    
    def _generate_report(self):
        """
        Generates a report of the analysis results.
        """
        # Create report
        report = Report("Timeline Analysis Report", self.results_dir)
        
        # Add introduction
        intro = "Analysis of issue activity over time"
        report.add_section("Introduction", intro)
        
        # Add creation and closing patterns section
        patterns = self.results['creation_closing_patterns']
        
        patterns_content = "Issue creation and closing patterns:\n\n"
        patterns_content += "See the 'Issue creation and closure over time.png' visualization for a graphical representation.\n\n"
        
        # Add some statistics
        total_created = sum(patterns['creation_by_month'].values())
        total_closed = sum(patterns['closing_by_month'].values())
        
        patterns_content += f"Total issues created: {total_created}\n"
        patterns_content += f"Total issues closed: {total_closed}\n"
        
        # Find month with most activity
        max_creation_month = max(patterns['creation_by_month'].items(), key=lambda x: x[1])
        max_closing_month = max(patterns['closing_by_month'].items(), key=lambda x: x[1])
        
        patterns_content += f"\nMonth with most issues created: {max_creation_month[0]} ({max_creation_month[1]} issues)\n"
        patterns_content += f"Month with most issues closed: {max_closing_month[0]} ({max_closing_month[1]} issues)\n"
        
        report.add_section("Issue Creation and Closing Patterns", patterns_content)
        
        # Add issue lifecycle section
        lifecycle = self.results['issue_lifecycle']
        
        lifecycle_content = "Issue lifecycle metrics:\n\n"
        
        if lifecycle['avg_first_response'] is not None:
            lifecycle_content += f"Average time to first response: {lifecycle['avg_first_response']:.2f} days\n"
        
        if lifecycle['median_first_response'] is not None:
            lifecycle_content += f"Median time to first response: {lifecycle['median_first_response']:.2f} days\n"
        
        if lifecycle['avg_resolution'] is not None:
            lifecycle_content += f"Average time to resolution: {lifecycle['avg_resolution']:.2f} days\n"
        
        if lifecycle['median_resolution'] is not None:
            lifecycle_content += f"Median time to resolution: {lifecycle['median_resolution']:.2f} days\n"
        
        report.add_section("Issue Lifecycle", lifecycle_content)
        
        # Add activity trends section
        trends = self.results['activity_trends']
        
        trends_content = "Activity trends:\n\n"
        
        # Day of week trends
        trends_content += "Activity by day of week:\n\n"
        for day, count in zip(trends['day_of_week']['labels'], trends['day_of_week']['counts']):
            trends_content += f"{day}: {count} events\n"
        
        # Hour of day trends
        trends_content += "\nActivity by hour of day:\n\n"
        for hour, count in zip(trends['hour_of_day']['labels'], trends['hour_of_day']['counts']):
            trends_content += f"{hour}:00: {count} events\n"
        
        report.add_section("Activity Trends", trends_content)
        
        # Save report
        report.save_text_report("result.txt")
        report.save_markdown_report("report.md")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    TimelineAnalysis().run()
