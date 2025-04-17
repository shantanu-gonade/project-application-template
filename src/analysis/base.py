"""
Base class for all analyses.
Provides common functionality for loading data, saving results, etc.
"""

from typing import List, Dict, Any, Optional
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

from src.core.data_loader import DataLoader
from src.core.model import Issue, Event, State
import src.core.config as config

class BaseAnalysis:
    """
    Base class for all analyses.
    Provides common functionality for loading data, saving results, etc.
    """
    
    def __init__(self, name: str):
        """
        Constructor
        
        Args:
            name: Name of the analysis, used for saving results
        """
        self.name = name
        self.label: Optional[str] = config.get_parameter('label')
        self.user: Optional[str] = config.get_parameter('user')
        self.results_dir = os.path.join('results', self.name)
        
        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Create images directory if it doesn't exist
        self.images_dir = os.path.join(self.results_dir, 'images')
        os.makedirs(self.images_dir, exist_ok=True)
    
    def run(self):
        """
        Starting point for this analysis.
        Should be overridden by subclasses.
        """
        issues = self.load_data()
        self.analyze(issues)
        self.visualize(issues)
        self.save_results()
    
    def load_data(self) -> List[Issue]:
        """
        Loads the issues data.
        
        Returns:
            List of Issue objects
        """
        issues: List[Issue] = DataLoader().get_issues()
        
        # Filter issues by label if specified
        if self.label:
            filtered_issues = [issue for issue in issues if self.label in issue.labels]
            print(f"\n\nAnalyzing {len(filtered_issues)} issues with label '{self.label}'\n")
            return filtered_issues
        
        # Filter issues by user if specified
        if self.user:
            filtered_issues = [issue for issue in issues if issue.creator == self.user]
            print(f"\n\nAnalyzing {len(filtered_issues)} issues created by '{self.user}'\n")
            return filtered_issues
        
        print(f"\n\nAnalyzing all {len(issues)} issues\n")
        return issues
    
    def analyze(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Performs the analysis on the issues.
        Should be overridden by subclasses.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of analysis results
        """
        raise NotImplementedError("Subclasses must implement analyze()")
    
    def visualize(self, issues: List[Issue]):
        """
        Creates visualizations of the analysis results.
        Should be overridden by subclasses.
        
        Args:
            issues: List of Issue objects
        """
        raise NotImplementedError("Subclasses must implement visualize()")
    
    def save_results(self):
        """
        Saves the analysis results to a file.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement save_results()")
    
    def save_figure(self, fig, filename: str):
        """
        Saves a matplotlib figure to the images directory.
        
        Args:
            fig: Matplotlib figure
            filename: Name of the file (without extension)
        """
        filepath = os.path.join(self.images_dir, f"{filename}.png")
        fig.savefig(filepath)
        print(f"Saved figure to {filepath}")
    
    def save_json(self, data: Dict[str, Any], filename: str):
        """
        Saves data to a JSON file in the results directory.
        
        Args:
            data: Data to save
            filename: Name of the file (without extension)
        """
        filepath = os.path.join(self.results_dir, f"{filename}.json")
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Saved data to {filepath}")
    
    def save_text(self, text: str, filename: str):
        """
        Saves text to a file in the results directory.
        
        Args:
            text: Text to save
            filename: Name of the file (without extension)
        """
        filepath = os.path.join(self.results_dir, f"{filename}.txt")
        with open(filepath, 'w') as f:
            f.write(text)
        print(f"Saved text to {filepath}")
