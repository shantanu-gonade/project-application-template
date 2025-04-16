from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from data_loader import DataLoader
from model import Issue, Event, State
import config

class IssueResolutionPrediction:
    """
    Builds a predictive model for issue resolution:
    1. Predicts whether an issue will be resolved quickly or slowly
    2. Identifies factors that contribute to faster resolution
    3. Analyzes the impact of different events on resolution time
    4. Provides recommendations for improving issue resolution
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
        
        print(f"\n\nBuilding predictive model for issue resolution using {len(issues)} issues\n")
        
        # Prepare features for each issue
        issue_features = []
        for issue in issues:
            # Skip open issues
            if issue.state != State.closed:
                continue
                
            # Find closing event
            closing_event = None
            for event in issue.events:
                if event.event_type == "closed" and event.event_date:
                    closing_event = event
                    break
                    
            # Skip if no closing event found
            if not closing_event or not issue.created_date:
                continue
                
            # Calculate resolution time in days
            resolution_time = (closing_event.event_date - issue.created_date).days
            
            # Skip negative resolution times (data errors)
            if resolution_time < 0:
                continue
                
            # Extract features
            features = {
                'issue_number': issue.number,
                'title_length': len(issue.title) if issue.title else 0,
                'description_length': len(issue.text) if issue.text else 0,
                'label_count': len(issue.labels),
                'has_bug_label': any('bug' in label.lower() for label in issue.labels),
                'has_feature_label': any('feature' in label.lower() for label in issue.labels),
                'has_enhancement_label': any('enhancement' in label.lower() for label in issue.labels),
                'comment_count': len([e for e in issue.events if e.event_type == "commented"]),
                'participant_count': len(set([issue.creator] + [e.author for e in issue.events if e.author])),
                'has_code': '```' in issue.text if issue.text else False,
                'has_comments_with_code': any('```' in e.comment for e in issue.events if e.event_type == "commented" and e.comment),
                'resolution_time': resolution_time,
                'is_quick_resolution': resolution_time <= 7  # Binary classification: resolved within a week
            }
            
            issue_features.append(features)
        
        print(f"Prepared features for {len(issue_features)} closed issues")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(issue_features)
        
        if len(df) < 10:
            print("Not enough data to build a predictive model. Need at least 10 closed issues.")
            return
        
        # Analyze resolution time distribution
        self._analyze_resolution_time_distribution(df)
        
        # Analyze feature correlations
        self._analyze_feature_correlations(df)
        
        # Build predictive model
        self._build_predictive_model(df)
        
        # Provide recommendations
        self._provide_recommendations(df)
    
    def _analyze_resolution_time_distribution(self, df):
        """
        Analyzes the distribution of resolution times.
        """
        print("\nAnalyzing resolution time distribution...")
        
        # Create histogram of resolution times
        plt.figure(figsize=(10, 6))
        plt.hist(df['resolution_time'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Resolution Time (days)')
        plt.ylabel('Number of Issues')
        plt.title('Distribution of Issue Resolution Times')
        plt.grid(axis='y', alpha=0.75)
        
        # Add median line
        median_time = df['resolution_time'].median()
        plt.axvline(median_time, color='red', linestyle='dashed', linewidth=1, 
                   label=f'Median: {median_time:.1f} days')
        
        # Add mean line
        mean_time = df['resolution_time'].mean()
        plt.axvline(mean_time, color='green', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {mean_time:.1f} days')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Show quick vs. slow resolution percentage
        quick_pct = df['is_quick_resolution'].mean() * 100
        print(f"Percentage of issues resolved within a week: {quick_pct:.1f}%")
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        plt.pie([quick_pct, 100 - quick_pct], 
                labels=['Quick Resolution (≤7 days)', 'Slow Resolution (>7 days)'],
                autopct='%1.1f%%',
                colors=['#66b3ff', '#ff9999'],
                startangle=90,
                explode=(0.1, 0))
        plt.title('Quick vs. Slow Issue Resolution')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def _analyze_feature_correlations(self, df):
        """
        Analyzes correlations between features and resolution time.
        """
        print("\nAnalyzing feature correlations with resolution time...")
        
        # Select numeric features for correlation analysis
        numeric_features = [
            'title_length', 'description_length', 'label_count', 
            'comment_count', 'participant_count', 'resolution_time'
        ]
        
        # Calculate correlations
        corr = df[numeric_features].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.show()
        
        # Plot individual feature correlations with resolution time
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        features_to_plot = [
            'comment_count', 'participant_count', 'label_count', 'description_length'
        ]
        
        for i, feature in enumerate(features_to_plot):
            sns.regplot(x=feature, y='resolution_time', data=df, ax=axes[i], scatter_kws={'alpha': 0.5})
            axes[i].set_title(f'{feature.replace("_", " ").title()} vs. Resolution Time')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Analyze categorical features
        categorical_features = ['has_bug_label', 'has_feature_label', 'has_enhancement_label', 'has_code', 'has_comments_with_code']
        
        fig, axes = plt.subplots(len(categorical_features), 1, figsize=(10, 4 * len(categorical_features)))
        
        for i, feature in enumerate(categorical_features):
            # Calculate average resolution time for each category
            avg_times = df.groupby(feature)['resolution_time'].mean()
            
            # Create bar chart
            avg_times.plot(kind='bar', ax=axes[i], color=['#ff9999', '#66b3ff'])
            axes[i].set_title(f'Average Resolution Time by {feature.replace("_", " ").title()}')
            axes[i].set_ylabel('Avg. Resolution Time (days)')
            axes[i].set_xticklabels(['No', 'Yes'])
            axes[i].grid(axis='y', alpha=0.3)
            
            # Add count annotations
            counts = df.groupby(feature).size()
            for j, count in enumerate(counts):
                axes[i].annotate(f'n={count}', 
                               xy=(j, avg_times.iloc[j]), 
                               xytext=(0, 5),
                               textcoords='offset points',
                               ha='center')
        
        plt.tight_layout()
        plt.show()
    
    def _build_predictive_model(self, df):
        """
        Builds a predictive model for issue resolution time.
        """
        print("\nBuilding predictive model for quick vs. slow resolution...")
        
        # Select features and target
        feature_cols = [
            'title_length', 'description_length', 'label_count', 
            'has_bug_label', 'has_feature_label', 'has_enhancement_label',
            'comment_count', 'participant_count', 'has_code', 'has_comments_with_code'
        ]
        
        X = df[feature_cols]
        y = df['is_quick_resolution']
        
        # Split data into training and testing sets
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = clf.predict(X_test_scaled)
            
            # Evaluate model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Model accuracy: {accuracy:.2f}")
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['Slow Resolution', 'Quick Resolution']))
            
            # Plot confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Slow Resolution', 'Quick Resolution'],
                       yticklabels=['Slow Resolution', 'Quick Resolution'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.show()
            
            # Plot feature importance
            feature_importance = clf.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), [feature_cols[i].replace('_', ' ').title() for i in sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance for Predicting Quick Resolution')
            plt.tight_layout()
            plt.show()
            
            return clf, feature_cols, feature_importance
            
        except Exception as e:
            print(f"Error building predictive model: {e}")
            return None, None, None
    
    def _provide_recommendations(self, df):
        """
        Provides recommendations for improving issue resolution based on analysis.
        """
        print("\nRecommendations for improving issue resolution:")
        
        # Analyze what factors are associated with quick resolution
        quick_resolution = df[df['is_quick_resolution'] == True]
        slow_resolution = df[df['is_quick_resolution'] == False]
        
        # Compare average values for key metrics
        metrics = ['comment_count', 'participant_count', 'label_count', 'description_length']
        quick_avg = quick_resolution[metrics].mean()
        slow_avg = slow_resolution[metrics].mean()
        
        # Calculate percentage differences
        pct_diff = (quick_avg - slow_avg) / slow_avg * 100
        
        # Identify significant factors (absolute percentage difference > 20%)
        significant_factors = pct_diff[abs(pct_diff) > 20].sort_values()
        
        if not significant_factors.empty:
            print("\nSignificant factors affecting resolution time:")
            for factor, diff in significant_factors.items():
                direction = "lower" if diff < 0 else "higher"
                print(f"- {factor.replace('_', ' ').title()} is {abs(diff):.1f}% {direction} in quickly resolved issues")
        
        # Check categorical factors
        categorical_factors = ['has_bug_label', 'has_feature_label', 'has_enhancement_label', 'has_code', 'has_comments_with_code']
        for factor in categorical_factors:
            quick_pct = quick_resolution[factor].mean() * 100
            slow_pct = slow_resolution[factor].mean() * 100
            diff = quick_pct - slow_pct
            
            if abs(diff) > 10:  # Only show if difference is more than 10 percentage points
                more_less = "more" if diff > 0 else "less"
                print(f"- {factor.replace('_', ' ').replace('has ', '').title()} is {abs(diff):.1f} percentage points {more_less} common in quickly resolved issues")
        
        # General recommendations
        print("\nGeneral recommendations:")
        print("1. Encourage clear and concise issue descriptions")
        print("2. Ensure issues have appropriate labels for better categorization")
        print("3. Promote active participation and discussion on issues")
        print("4. Include code examples when relevant to the issue")
        print("5. Regularly review and prioritize older unresolved issues")
        print("6. Consider implementing templates for bug reports and feature requests")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueResolutionPrediction().run()
