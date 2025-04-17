"""
Common utility functions for the application.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import re

from src.core.model import Issue, Event, State

def filter_issues_by_label(issues: List[Issue], label: str) -> List[Issue]:
    """
    Filters issues by label.
    
    Args:
        issues: List of Issue objects
        label: Label to filter by
        
    Returns:
        Filtered list of Issue objects
    """
    return [issue for issue in issues if label in issue.labels]

def filter_issues_by_user(issues: List[Issue], user: str) -> List[Issue]:
    """
    Filters issues by creator.
    
    Args:
        issues: List of Issue objects
        user: User to filter by
        
    Returns:
        Filtered list of Issue objects
    """
    return [issue for issue in issues if issue.creator == user]

def filter_issues_by_state(issues: List[Issue], state: State) -> List[Issue]:
    """
    Filters issues by state.
    
    Args:
        issues: List of Issue objects
        state: State to filter by
        
    Returns:
        Filtered list of Issue objects
    """
    return [issue for issue in issues if issue.state == state]

def filter_issues_by_date_range(
    issues: List[Issue], 
    start_date: Optional[datetime] = None, 
    end_date: Optional[datetime] = None
) -> List[Issue]:
    """
    Filters issues by creation date range.
    
    Args:
        issues: List of Issue objects
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        
    Returns:
        Filtered list of Issue objects
    """
    filtered_issues = issues
    
    if start_date:
        filtered_issues = [issue for issue in filtered_issues 
                          if issue.created_date and issue.created_date >= start_date]
    
    if end_date:
        filtered_issues = [issue for issue in filtered_issues 
                          if issue.created_date and issue.created_date <= end_date]
    
    return filtered_issues

def get_issue_resolution_time(issue: Issue) -> Optional[float]:
    """
    Calculates the resolution time of an issue in days.
    
    Args:
        issue: Issue object
        
    Returns:
        Resolution time in days, or None if the issue is not closed
    """
    if issue.state != State.closed or not issue.created_date:
        return None
    
    # Find closing event
    closing_event = None
    for event in issue.events:
        if event.event_type == "closed" and event.event_date:
            closing_event = event
            break
    
    if not closing_event or not closing_event.event_date:
        return None
    
    # Calculate resolution time in days
    return (closing_event.event_date - issue.created_date).total_seconds() / 86400

def get_issue_first_response_time(issue: Issue) -> Optional[float]:
    """
    Calculates the time to first response for an issue in days.
    
    Args:
        issue: Issue object
        
    Returns:
        Time to first response in days, or None if there are no responses
    """
    if not issue.created_date:
        return None
    
    # Find first comment event
    comment_events = [event for event in issue.events 
                     if event.event_type == "commented" and event.event_date]
    
    if not comment_events:
        return None
    
    first_comment = min(comment_events, key=lambda e: e.event_date)
    
    # Calculate time to first response in days
    return (first_comment.event_date - issue.created_date).total_seconds() / 86400

def get_issue_comment_count(issue: Issue) -> int:
    """
    Counts the number of comments on an issue.
    
    Args:
        issue: Issue object
        
    Returns:
        Number of comments
    """
    return len([event for event in issue.events if event.event_type == "commented"])

def get_issue_unique_participants(issue: Issue) -> List[str]:
    """
    Gets the list of unique participants in an issue.
    
    Args:
        issue: Issue object
        
    Returns:
        List of unique participant usernames
    """
    participants = set([issue.creator])
    for event in issue.events:
        if event.author:
            participants.add(event.author)
    
    return list(participants)

def has_code_blocks(text: Optional[str]) -> bool:
    """
    Checks if a text contains code blocks.
    
    Args:
        text: Text to check
        
    Returns:
        True if the text contains code blocks, False otherwise
    """
    if not text:
        return False
    
    return '```' in text

def group_by_attribute(items: List[Any], key_func: Callable[[Any], Any]) -> Dict[Any, List[Any]]:
    """
    Groups items by a key function.
    
    Args:
        items: List of items to group
        key_func: Function that extracts the key from an item
        
    Returns:
        Dictionary mapping keys to lists of items
    """
    result = {}
    for item in items:
        key = key_func(item)
        if key in result:
            result[key].append(item)
        else:
            result[key] = [item]
    
    return result

def calculate_average(values: List[float]) -> float:
    """
    Calculates the average of a list of values.
    
    Args:
        values: List of values
        
    Returns:
        Average value
    """
    if not values:
        return 0.0
    
    return sum(values) / len(values)

def calculate_median(values: List[float]) -> float:
    """
    Calculates the median of a list of values.
    
    Args:
        values: List of values
        
    Returns:
        Median value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n % 2 == 0:
        return (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
    else:
        return sorted_values[n // 2]

def extract_date_components(date: datetime) -> Dict[str, int]:
    """
    Extracts components from a datetime object.
    
    Args:
        date: Datetime object
        
    Returns:
        Dictionary with year, month, day, weekday, hour components
    """
    return {
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'weekday': date.weekday(),
        'hour': date.hour
    }

def group_by_time_period(
    dates: List[datetime], 
    period: str = 'month'
) -> Dict[str, int]:
    """
    Groups dates by a time period and counts occurrences.
    
    Args:
        dates: List of datetime objects
        period: Time period to group by ('day', 'week', 'month', 'year')
        
    Returns:
        Dictionary mapping period labels to counts
    """
    result = {}
    
    for date in dates:
        if period == 'day':
            key = date.strftime('%Y-%m-%d')
        elif period == 'week':
            # Get the Monday of the week
            monday = date - timedelta(days=date.weekday())
            key = monday.strftime('%Y-%m-%d')
        elif period == 'month':
            key = date.strftime('%Y-%m')
        elif period == 'year':
            key = str(date.year)
        else:
            raise ValueError(f"Invalid period: {period}")
        
        if key in result:
            result[key] += 1
        else:
            result[key] = 1
    
    return result
