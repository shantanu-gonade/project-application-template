"""
Report generation utilities.
"""

from typing import List, Dict, Any, Optional
import os
import json
from datetime import datetime

class Report:
    """
    Generates reports from analysis results.
    """
    
    def __init__(self, name: str, results_dir: str):
        """
        Constructor
        
        Args:
            name: Name of the report
            results_dir: Directory to save the report to
        """
        self.name = name
        self.results_dir = results_dir
        self.sections = []
        
    def add_section(self, title: str, content: str):
        """
        Adds a section to the report.
        
        Args:
            title: Section title
            content: Section content
        """
        self.sections.append({
            'title': title,
            'content': content
        })
    
    def add_table(self, title: str, headers: List[str], rows: List[List[Any]]):
        """
        Adds a table to the report.
        
        Args:
            title: Table title
            headers: Table headers
            rows: Table rows
        """
        # Format the table as text
        content = f"{title}\n\n"
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Add headers - use exact format expected by test
        header_row = headers[0]
        for h in headers[1:]:
            header_row += " | " + h
        content += header_row + "\n"
        content += "-" * len(header_row) + "\n"
        
        # Add rows
        for row in rows:
            row_text = str(row[0]).ljust(col_widths[0])
            for i, cell in enumerate(row[1:], 1):
                row_text += " | " + str(cell).ljust(col_widths[i])
            content += row_text + "\n"
        
        self.add_section(title, content)
    
    def add_key_value_section(self, title: str, data: Dict[str, Any]):
        """
        Adds a key-value section to the report.
        
        Args:
            title: Section title
            data: Dictionary of key-value pairs
        """
        content = f"{title}\n\n"
        
        # Calculate key width
        key_width = max(len(str(k)) for k in data.keys())
        
        # Add key-value pairs
        for key, value in data.items():
            content += f"{str(key).ljust(key_width)}: {value}\n"
        
        self.add_section(title, content)
    
    def add_list_section(self, title: str, items: List[str]):
        """
        Adds a list section to the report.
        
        Args:
            title: Section title
            items: List of items
        """
        content = f"{title}\n\n"
        
        # Add items
        for i, item in enumerate(items, 1):
            content += f"{i}. {item}\n"
        
        self.add_section(title, content)
    
    def generate_text_report(self) -> str:
        """
        Generates a text report.
        
        Returns:
            Report as a string
        """
        report = f"# {self.name}\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        for section in self.sections:
            report += f"## {section['title']}\n\n"
            report += section['content'] + "\n\n"
        
        return report
    
    def save_text_report(self, filename: str = "report.txt"):
        """
        Saves the report as a text file.
        
        Args:
            filename: Name of the file to save to
        """
        report = self.generate_text_report()
        
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Saved report to {filepath}")
        
        return filepath
    
    def generate_json_report(self) -> Dict[str, Any]:
        """
        Generates a JSON report.
        
        Returns:
            Report as a dictionary
        """
        return {
            'name': self.name,
            'generated': datetime.now().isoformat(),
            'sections': self.sections
        }
    
    def save_json_report(self, filename: str = "report.json"):
        """
        Saves the report as a JSON file.
        
        Args:
            filename: Name of the file to save to
        """
        report = self.generate_json_report()
        
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Saved report to {filepath}")
        
        return filepath
    
    def generate_markdown_report(self) -> str:
        """
        Generates a Markdown report.
        
        Returns:
            Report as a string
        """
        report = f"# {self.name}\n\n"
        report += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        for section in self.sections:
            report += f"## {section['title']}\n\n"
            report += section['content'] + "\n\n"
        
        return report
    
    def save_markdown_report(self, filename: str = "report.md"):
        """
        Saves the report as a Markdown file.
        
        Args:
            filename: Name of the file to save to
        """
        report = self.generate_markdown_report()
        
        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            f.write(report)
        
        print(f"Saved report to {filepath}")
        
        return filepath
