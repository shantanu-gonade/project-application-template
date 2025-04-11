import os
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime
import json
from data_loader import DataLoader
from model import Issue, Event
import config

class ContributorAnalysis:
    """
    Implements an analysis of GitHub contributor activity.
    This analysis focuses on:
    1. Top contributors by different metrics
    2. Contributor activity patterns
    3. Contributor focus areas (labels)
    4. Multiple analysis types
    5. Image and text file storage
    6. Comprehensive statistics
    """
    
    def __init__(self):
        """
        Constructor
        """
        self.user: str = config.get_parameter('user')
        self.output_dir = "contributor_analysis_results"
        self._setup_output_directory()
        if self.user:
            self.contributor_report_file = open(os.path.join(self.output_dir, f"{self.user}_analysis_report.txt"), "w", encoding="utf-8")
        else:
            self.overall_report_file = open(os.path.join(self.output_dir, "overall_analysis_report.txt"), "w", encoding="utf-8")
        
    def _setup_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
    
    def run(self):
        """
        Starting point for this analysis.
        """
        try:
            issues: List[Issue] = DataLoader().get_issues()
            
            if self.user:
                self.contributor_report_file.write(f"Contributor Analysis Report\n")
                self.contributor_report_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.contributor_report_file.write(f"\nAnalyzing activity for contributor: {self.user}\n")
                self._analyze_specific_contributor(issues, self.user)
            else:
                self.overall_report_file.write(f"Contributor Analysis Report\n")
                self.overall_report_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.overall_report_file.write(f"Total issues analyzed: {len(issues)}\n\n")
                self.overall_report_file.write("\nAnalyzing all contributors\n")
                self._analyze_all_contributors(issues)
                
        finally:
            if self.user:
                self.contributor_report_file.close()
            else:
                self.overall_report_file.close()
    
    def _analyze_all_contributors(self, issues: List[Issue]):
        """Comprehensive analysis of all contributors"""
        analyses = [
            ("Issue Creators", self._analyze_top_issue_creators),
            ("Commenters", self._analyze_top_commenters),
            ("Issue Closers", self._analyze_top_issue_closers),
            ("Label Focus", self._analyze_contributor_label_focus),
            ("Activity Timeline", self._analyze_activity_timeline),
            ("Response Times", self._analyze_response_times),
            ("Collaboration Network", self._analyze_collaboration_network),
            ("Issue Lifecycle", self._analyze_issue_lifecycle)
        ]
        
        for name, analysis_func in analyses:
            self.overall_report_file.write(f"\n=== {name} Analysis ===\n")
            analysis_func(issues)

    def _save_analysis_data(self, filename: str, data: dict):
        """Save analysis data as JSON"""
        with open(os.path.join(self.output_dir, f"{filename}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_plot(self, filename: str):
        """Save current matplotlib plot to file"""
        plt.savefig(os.path.join(self.output_dir, "images", f"{filename}.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()

    def _write_to_report(self, text: str):
        """Write text to both console and report file"""
        print(text)
        if self.user:
            self.contributor_report_file.write(text + "\n")
        else:
            self.overall_report_file.write(text + "\n")

    def _analyze_top_issue_creators(self, issues: List[Issue]):
        """Top issue creator analysis"""
        creator_counts = Counter(issue.creator for issue in issues if issue.creator)
        top_creators = creator_counts.most_common(15)
        
        data = {
            "top_creators": [{"name": name, "count": count} for name, count in top_creators],
            "total_issues": len(issues),
            "unique_creators": len(creator_counts)
        }
        self._save_analysis_data("top_issue_creators", data)
        
        self._write_to_report("\nTop 15 Issue Creators:")
        for i, (creator, count) in enumerate(top_creators, 1):
            self._write_to_report(f"{i}. {creator}: {count} issues ({count/len(issues):.1%})")
        
        # Visualization
        self._plot_horizontal_bar(
            [creator for creator, _ in top_creators],
            [count for _, count in top_creators],
            "Top Issue Creators",
            "Number of Issues Created"
        )
        self._save_plot("top_issue_creators")

    def _analyze_top_commenters(self, issues: List[Issue]):
        """Top ommenter analysis with comment length"""
        commenter_counts = Counter()
        comment_lengths = defaultdict(list)
        
        for issue in issues:
            for event in issue.events:
                if event.event_type == "commented":
                    commenter_counts[event.author] += 1
                    if hasattr(event, 'text'):
                        comment_lengths[event.author].append(len(event.text))
        
        top_commenters = commenter_counts.most_common(15)
        
        # Calculate average comment length
        avg_lengths = {author: np.mean(lengths) for author, lengths in comment_lengths.items()}
        
        data = {
            "top_commenters": [{
                "name": name, 
                "count": count,
                "avg_comment_length": avg_lengths.get(name, 0)
            } for name, count in top_commenters],
            "total_comments": sum(commenter_counts.values())
        }
        self._save_analysis_data("top_commenters", data)
        
        self._write_to_report("\nTop 15 Commenters:")
        for i, (commenter, count) in enumerate(top_commenters, 1):
            avg_len = avg_lengths.get(commenter, 0)
            self._write_to_report(f"{i}. {commenter}: {count} comments (avg length: {avg_len:.0f} chars)")
        
        # Visualization
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Bar plot for comment counts
        names = [name for name, _ in top_commenters]
        counts = [count for _, count in top_commenters]
        ax1.barh(names, counts, color='skyblue')
        ax1.set_xlabel('Number of Comments')
        ax1.set_title('Top Commenters with Average Comment Length')
        
        # Line plot for average comment length
        ax2 = ax1.twiny()
        lengths = [avg_lengths.get(name, 0) for name in names]
        ax2.plot(lengths, names, 'ro-', label='Avg Length')
        ax2.set_xlabel('Average Comment Length (chars)')
        
        plt.tight_layout()
        self._save_plot("top_commenters")

    def _analyze_top_issue_closers(self, issues: List[Issue]):
        """Top Closer analysis with time to close"""
        closer_counts = Counter()
        close_times = defaultdict(list)
        
        for issue in issues:
            if issue.state == "closed":
                created = self._parse_datetime(issue.created_date) 
                updated = self._parse_datetime(issue.updated_date)
                time_to_close = (updated - created).total_seconds() / 3600  # in hours
                
                for event in issue.events:
                    if event.event_type == "closed":
                        closer_counts[event.author] += 1
                        close_times[event.author].append(time_to_close)
        
        top_closers = closer_counts.most_common(15)
        avg_close_times = {author: np.mean(times) for author, times in close_times.items()}
        
        data = {
            "top_closers": [{
                "name": name,
                "count": count,
                "avg_hours_to_close": avg_close_times.get(name, 0)
            } for name, count in top_closers],
            "total_closed_issues": sum(closer_counts.values())
        }
        self._save_analysis_data("top_issue_closers", data)
        
        self._write_to_report("\nTop 15 Issue Closers:")
        for i, (closer, count) in enumerate(top_closers, 1):
            avg_time = avg_close_times.get(closer, 0)
            self._write_to_report(f"{i}. {closer}: {count} issues closed (avg time: {avg_time:.1f} hours)")
        
        # Visualization
        self._plot_horizontal_bar(
            [name for name, _ in top_closers],
            [count for _, count in top_closers],
            "Top Issue Closers",
            "Number of Issues Closed"
        )
        self._save_plot("top_issue_closers")

    def _analyze_contributor_label_focus(self, issues: List[Issue]):
        """Label focus analysis with heatmap"""
        contributor_labels = defaultdict(Counter)
        
        for issue in issues:
            contributors = {issue.creator} if issue.creator else set()
            contributors.update(
                event.author for event in issue.events 
                if event.event_type in ["commented", "closed"]
            )
            
            for contributor in contributors:
                for label in issue.labels:
                    contributor_labels[contributor][label] += 1
        
        # Get top 10 contributors and top 10 labels
        top_contributors = sorted(
            contributor_labels.items(),
            key=lambda x: sum(x[1].values()),
            reverse=True
        )[:10]
        
        all_labels = set()
        for _, labels in contributor_labels.items():
            all_labels.update(labels.keys())
        top_labels = Counter()
        for labels in contributor_labels.values():
            top_labels.update(labels)
        top_labels = [label for label, _ in top_labels.most_common(10)]
        
        # Prepare data for heatmap
        heatmap_data = []
        for contributor, labels in top_contributors:
            row = [labels.get(label, 0) for label in top_labels]
            heatmap_data.append(row)
        
        data = {
            "contributor_label_matrix": {
                "contributors": [name for name, _ in top_contributors],
                "labels": top_labels,
                "data": heatmap_data
            }
        }
        self._save_analysis_data("contributor_label_focus", data)
        
        self._write_to_report("\nContributor Label Focus (Top 10 Contributors x Top 10 Labels):")
        df = pd.DataFrame(heatmap_data, index=[name for name, _ in top_contributors], columns=top_labels)
        self._write_to_report("\n" + str(df))
        
        # Heatmap visualization
        plt.figure(figsize=(12, 8))
        plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Interaction Count')
        plt.xticks(np.arange(len(top_labels)), top_labels, rotation=45, ha='right')
        plt.yticks(np.arange(len(top_contributors)), [name for name, _ in top_contributors])
        plt.title("Contributor Label Focus Heatmap")
        plt.tight_layout()
        self._save_plot("contributor_label_heatmap")

    def _analyze_activity_timeline(self, issues: List[Issue]):
        """Analyze activity patterns over time"""
        daily_activity = defaultdict(Counter)
        
        for issue in issues:
            # Issue creation
            created_date = self._parse_datetime(issue.created_date).date()
            daily_activity[created_date]["created"] += 1
            
            # Events
            for event in issue.events:
                event_date = self._parse_datetime(event.event_date).date()
                daily_activity[event_date][event.event_type] += 1
        
        # Convert to DataFrame
        dates = sorted(daily_activity.keys())
        event_types = set()
        for activities in daily_activity.values():
            event_types.update(activities.keys())
        
        timeline_data = []
        for date in dates:
            row = {"date": date.strftime("%Y-%m-%d")}
            row.update(daily_activity[date])
            timeline_data.append(row)
        
        self._save_analysis_data("activity_timeline", {"timeline": timeline_data})
        
        # Visualization
        df = pd.DataFrame(timeline_data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        plt.figure(figsize=(14, 8))
        for event_type in ['created', 'commented', 'closed']:
            if event_type in df.columns:
                df[event_type].plot(label=event_type.capitalize())
        
        plt.title("Activity Timeline")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        self._save_plot("activity_timeline")

    def _analyze_response_times(self, issues: List[Issue]):
        """Analyze how quickly contributors respond to issues"""
        response_times = defaultdict(list)
        
        for issue in issues:
            if not issue.events:
                continue
                
            created = self._parse_datetime(issue.created_date)
            first_comment = None
            
            for event in issue.events:
                if event.event_type == "commented":
                    event_time = self._parse_datetime(event.event_date)
                    response_time = (event_time - created).total_seconds() / 3600  # hours
                    response_times[event.author].append(response_time)
                    break
        
        # Filter contributors with at least 5 responses
        filtered_times = {k: v for k, v in response_times.items() if len(v) >= 5}
        avg_response = {k: np.mean(v) for k, v in filtered_times.items()}
        
        data = {
            "average_response_times": [
                {"contributor": k, "avg_hours": v, "response_count": len(response_times[k])}
                for k, v in sorted(avg_response.items(), key=lambda x: x[1])
            ]
        }
        self._save_analysis_data("response_times", data)
        
        self._write_to_report("\nAverage Response Times (contributors with ≥5 responses):")
        for i, (contributor, time) in enumerate(sorted(avg_response.items(), key=lambda x: x[1]), 1):
            self._write_to_report(f"{i}. {contributor}: {time:.1f} hours (based on {len(response_times[contributor])} responses)")
        
        # Visualization
        if avg_response:
            names, times = zip(*sorted(avg_response.items(), key=lambda x: x[1]))
            self._plot_horizontal_bar(
                names,
                times,
                "Average Response Times (hours)",
                "Response Time (hours)"
            )
            self._save_plot("response_times")

    def _analyze_collaboration_network(self, issues: List[Issue]):
        """Analyze collaboration patterns between contributors"""
        collaborations = defaultdict(Counter)
        
        for issue in issues:
            participants = set()
            if issue.creator:
                participants.add(issue.creator)
            
            participants.update(
                event.author for event in issue.events
                if event.event_type in ["commented", "closed"]
            )
            
            # Record all pairwise collaborations
            participants = list(participants)
            for i in range(len(participants)):
                for j in range(i+1, len(participants)):
                    pair = [p for p in [participants[i], participants[j]] if p is not None]
                    if len(pair) == 2:  # Only proceed if we have two valid participants
                        a, b = sorted(pair)
                        collaborations[a][b] += 1
        
        # Prepare data for visualization
        edges = []
        for a, partners in collaborations.items():
            for b, count in partners.items():
                if count >= 3:  # Only show significant collaborations
                    edges.append((a, b, count))
        
        data = {
            "collaboration_network": {
                "edges": [{"source": a, "target": b, "weight": c} for a, b, c in edges],
                "nodes": list(set([a for a, _, _ in edges] + [b for _, b, _ in edges]))
            }
        }
        self._save_analysis_data("collaboration_network", data)
        
        self._write_to_report("\nTop Collaborations (≥3 joint issues):")
        for a, b, count in sorted(edges, key=lambda x: -x[2])[:20]:
            self._write_to_report(f"- {a} ↔ {b}: {count} collaborations")

    def _analyze_issue_lifecycle(self, issues: List[Issue]):
        """Analyze the lifecycle of issues from creation to resolution"""
        lifecycle_data = []
        
        for issue in issues:
            if issue.state != "closed":
                continue
                
            created = self._parse_datetime(issue.created_date)
            closed = self._parse_datetime(issue.updated_date)
            duration = (closed - created).total_seconds() / 86400  # in days
            
            comments = sum(1 for e in issue.events if e.event_type == "commented")
            closers = [e.author for e in issue.events if e.event_type == "closed"]
            
            lifecycle_data.append({
                "duration_days": duration,
                "comment_count": comments,
                "labels": issue.labels,
                "closers": closers
            })
        
        self._save_analysis_data("issue_lifecycle", {"issues": lifecycle_data})
        
        if lifecycle_data:
            durations = [x["duration_days"] for x in lifecycle_data]
            comments = [x["comment_count"] for x in lifecycle_data]
            
            self._write_to_report("\nIssue Lifecycle Statistics:")
            self._write_to_report(f"- Average time to close: {np.mean(durations):.1f} days")
            self._write_to_report(f"- Median time to close: {np.median(durations):.1f} days")
            self._write_to_report(f"- Average comments per issue: {np.mean(comments):.1f}")
            
            # Duration distribution plot
            plt.figure(figsize=(12, 6))
            plt.hist(durations, bins=30, edgecolor='black')
            plt.title("Distribution of Issue Resolution Times")
            plt.xlabel("Days to Close")
            plt.ylabel("Number of Issues")
            plt.axvline(np.median(durations), color='r', linestyle='dashed', linewidth=1)
            plt.text(np.median(durations) + 0.5, plt.ylim()[1]*0.9, 
                    f'Median: {np.median(durations):.1f} days', color='r')
            plt.tight_layout()
            self._save_plot("issue_resolution_times")

    def _analyze_specific_contributor(self, issues: List[Issue], user: str):
        """Analysis for a specific contributor"""
        # Basic stats
        created_issues = [issue for issue in issues if issue.creator == user]
        comments = sum(
            1 for issue in issues 
            for event in issue.events 
            if event.event_type == "commented" and event.author == user
        )
        closed_issues = sum(
            1 for issue in issues 
            for event in issue.events 
            if event.event_type == "closed" and event.author == user
        )
        
        # Label focus
        label_counts = Counter()
        for issue in issues:
            if user in [issue.creator] + [e.author for e in issue.events if e.event_type == "commented"]:
                for label in issue.labels:
                    label_counts[label] += 1
        
        # Response times (as commenter)
        response_times = []
        for issue in issues:
            if issue.creator == user:  # Skip issues they created
                continue
                
            created = self._parse_datetime(issue.created_date)
            for event in issue.events:
                if event.event_type == "commented" and event.author == user:
                    event_time = self._parse_datetime(event.event_date)
                    response_times.append((event_time - created).total_seconds() / 3600)  # hours
                    break
        
        # Prepare report
        self._write_to_report(f"\n=== Comprehensive Analysis for {user} ===")
        self._write_to_report(f"\nBasic Statistics:")
        self._write_to_report(f"- Issues created: {len(created_issues)}")
        self._write_to_report(f"- Comments made: {comments}")
        self._write_to_report(f"- Issues closed: {closed_issues}")
        
        if response_times:
            self._write_to_report(f"\nResponse Times (as commenter):")
            self._write_to_report(f"- Average: {np.mean(response_times):.1f} hours")
            self._write_to_report(f"- Median: {np.median(response_times):.1f} hours")
            self._write_to_report(f"- Fastest: {np.min(response_times):.1f} hours")
            self._write_to_report(f"- Slowest: {np.max(response_times):.1f} hours")
            
            # Response time visualization
            plt.figure(figsize=(10, 6))
            plt.hist(response_times, bins=20, edgecolor='black')
            plt.title(f"{user}'s Response Time Distribution")
            plt.xlabel("Hours to First Comment")
            plt.ylabel("Count")
            plt.tight_layout()
            self._save_plot(f"{user}_response_times")
        
        if label_counts:
            self._write_to_report(f"\nLabel Focus (Top 5):")
            for label, count in label_counts.most_common(5):
                self._write_to_report(f"- {label}: {count} interactions")
            
            # Label focus visualization
            top_labels = label_counts.most_common(5)
            self._plot_horizontal_bar(
                [label for label, _ in top_labels],
                [count for _, count in top_labels],
                f"{user}'s Label Focus",
                "Interactions"
            )
            self._save_plot(f"{user}_label_focus")

    def _plot_horizontal_bar(self, labels, values, title, xlabel):
        """Helper for horizontal bar plots"""
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(labels))
        plt.barh(y_pos, values, align='center')
        plt.yticks(y_pos, labels)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.tight_layout()

    def _parse_datetime(self, dt):
        """Safely parse datetime from either string or datetime object"""
        if isinstance(dt, str):
            return datetime.strptime(dt, "%Y-%m-%dT%H:%M:%SZ")
        elif isinstance(dt, datetime):
            return dt
        else:
            raise ValueError(f"Invalid datetime format: {type(dt)}")

if __name__ == '__main__':
    ContributorAnalysis().run()