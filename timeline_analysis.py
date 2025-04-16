from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from data_loader import DataLoader
from model import Issue, Event, State
import config

class TimelineAnalysis:
    """
    Implements an analysis of GitHub issues over time.
    This analysis focuses on:
    1. Issue creation and closing patterns over time
    2. Issue lifecycle (time to first response, time to resolution)
    3. Activity trends over time
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
        
        print(f"\n\nAnalyzing issue timeline patterns across {len(issues)} issues\n")
        
        # Analyze issue creation and closing over time
        self._analyze_issue_creation_closing(issues)
        
        # Analyze issue lifecycle
        self._analyze_issue_lifecycle(issues)
        
        # Analyze activity trends
        self._analyze_activity_trends(issues)
    
    def _analyze_issue_creation_closing(self, issues: List[Issue]):
        """
        Analyzes the pattern of issue creation and closing over time.
        """
        # Collect issue creation dates
        creation_dates = [issue.created_date for issue in issues]
        
        # Collect issue closing dates (based on events of type 'closed')
        closing_dates = [
            event.event_date for issue in issues for event in issue.events if event.event_type == 'closed'
        ]
        
        # Convert to datetime objects for easier handling if they are strings
        creation_dates = [
            datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ') if isinstance(date, str) else date
            for date in creation_dates
        ]
        closing_dates = [
            datetime.strptime(date, '%Y-%m-%dT%H:%M:%SZ') if isinstance(date, str) else date
            for date in closing_dates
        ]

        # Plot creation and closing times
        plt.figure(figsize=(10, 6))
        
        # Plot creation dates
        plt.hist(creation_dates, bins=30, color='blue', alpha=0.7, label="Issue Creation")
        
        # Plot closing dates
        if closing_dates:
            plt.hist(closing_dates, bins=30, color='red', alpha=0.7, label="Issue Closure")
        
        plt.xlabel('Date')
        plt.ylabel('Number of Issues')
        plt.title('Issue Creation and Closure Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def _analyze_issue_lifecycle(self, issues: List[Issue]):
        """
        Analyzes the lifecycle of issues, including time to first response and time to resolution.
        """
        time_to_first_response = []
        time_to_resolution = []
        
        for issue in issues:
            # Find the closure event and extract the closure date
            closing_event = next((event for event in issue.events if event.event_type == 'closed'), None)
            if closing_event:
                closure_time = closing_event.event_date  # Assuming event_date is already a datetime object
                
                # Find the first comment event (this can be your first response)
                first_response_time = None
                for event in issue.events:
                    if event.event_type == 'commented' and not first_response_time:
                        first_response_time = event.event_date  # Assuming event_date is already a datetime object
                        break  # We only need the first comment as the first response
                
                # If we found the first response time, calculate the time to first response and resolution time
                if first_response_time:
                    # Check if created_date is a string or datetime and handle accordingly
                    if isinstance(issue.created_date, str):
                        created_time = datetime.strptime(issue.created_date, '%Y-%m-%dT%H:%M:%SZ')  # Convert from string
                    else:
                        created_time = issue.created_date  # If it's already a datetime object, use it directly
                    
                    time_to_first_response.append((first_response_time - created_time).total_seconds())
                    time_to_resolution.append((closure_time - created_time).total_seconds())
        
        # Plot time to first response
        plt.figure(figsize=(10, 6))
        plt.hist(time_to_first_response, bins=30, color='green', alpha=0.7, label="Time to First Response (s)")
        plt.hist(time_to_resolution, bins=30, color='orange', alpha=0.7, label="Time to Resolution (s)")
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Number of Issues')
        plt.title('Issue Lifecycle: Time to First Response vs. Time to Resolution')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _analyze_activity_trends(self, issues: List[Issue]):
        """
        Analyzes trends in activity over time based on events in the issues.
        """
        events_by_date = defaultdict(int)
        
        for issue in issues:
            for event in issue.events:
                # Use event_date instead of created_date
                event_date = event.event_date if isinstance(event.event_date, datetime) else datetime.strptime(event.event_date, '%Y-%m-%dT%H:%M:%SZ')
                events_by_date[event_date.date()] += 1
        
        # Convert to pandas DataFrame for easier plotting
        event_dates = list(events_by_date.keys())
        event_counts = list(events_by_date.values())
        
        df = pd.DataFrame({'Date': event_dates, 'Event Count': event_counts})
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Plot activity trends over time
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['Event Count'], marker='o', linestyle='-', color='purple')
        
        plt.xlabel('Date')
        plt.ylabel('Number of Events')
        plt.title('Activity Trends Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # Invoke run method when running this module directly
    TimelineAnalysis().run()
