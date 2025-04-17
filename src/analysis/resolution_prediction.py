"""
Implements a predictive model for issue resolution time.
"""

from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.analysis.base import BaseAnalysis
from src.core.model import Issue, Event, State
from src.utils.helpers import (
    get_issue_resolution_time,
    get_issue_comment_count,
    get_issue_unique_participants,
    has_code_blocks
)
from src.visualization.plotting import (
    create_histogram,
    create_bar_chart,
    create_pie_chart
)
from src.visualization.report import Report

class IssueResolutionPrediction(BaseAnalysis):
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
        super().__init__("resolution_prediction")
        self.results = {}
    
    def analyze(self, issues: List[Issue]) -> Dict[str, Any]:
        """
        Performs the analysis on the issues.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            Dictionary of analysis results
        """
        print("=== Issue Resolution Prediction ===")
        
        # Prepare features
        issue_features = self._prepare_features(issues)
        
        # Analyze resolution time distribution
        resolution_distribution = self._analyze_resolution_time_distribution(issue_features)
        
        # Analyze feature correlations
        feature_correlations = self._analyze_feature_correlations(issue_features)
        
        # Build predictive model
        model_results = self._build_predictive_model(issue_features)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issue_features, model_results)
        
        # Store results
        self.results = {
            'issue_features': issue_features,
            'resolution_distribution': resolution_distribution,
            'feature_correlations': feature_correlations,
            'model_results': model_results,
            'recommendations': recommendations
        }
        
        return self.results
    
    def visualize(self, issues: List[Issue]):
        """
        Creates visualizations of the analysis results.
        
        Args:
            issues: List of Issue objects
        """
        # Visualize resolution time distribution
        self._visualize_resolution_time_distribution()
        
        # Visualize feature correlations
        self._visualize_feature_correlations()
        
        # Visualize model results
        self._visualize_model_results()
        
        # Visualize recommendations
        self._visualize_recommendations()
    
    def save_results(self):
        """
        Saves the analysis results to files.
        """
        # Save results as JSON
        # Convert DataFrame to serializable format
        serializable_features = self.results['issue_features'].to_dict(orient='records')
        
        self.save_json(serializable_features, "issue_features")
        self.save_json(self.results['resolution_distribution'], "resolution_distribution")
        self.save_json(self.results['feature_correlations'], "feature_correlations")
        
        # Convert model results to serializable format
        model_results = {k: v for k, v in self.results['model_results'].items() 
                        if k not in ['model', 'X_train', 'X_test', 'y_train', 'y_test']}
        
        self.save_json(model_results, "model_results")
        self.save_json(self.results['recommendations'], "recommendations")
        
        # Generate and save report
        self._generate_report()
    
    def _prepare_features(self, issues: List[Issue]) -> pd.DataFrame:
        """
        Prepares features for each issue.
        
        Args:
            issues: List of Issue objects
            
        Returns:
            DataFrame with issue features
        """
        print("\n=== Preparing Features ===")
        
        issue_features = []
        
        for issue in issues:
            # Skip open issues
            if issue.state != State.closed:
                continue
            
            # Calculate resolution time
            resolution_time = get_issue_resolution_time(issue)
            
            # Skip if no resolution time found
            if resolution_time is None:
                continue
            
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
                'comment_count': get_issue_comment_count(issue),
                'participant_count': len(get_issue_unique_participants(issue)),
                'has_code': has_code_blocks(issue.text) or any(has_code_blocks(e.comment) 
                                                             for e in issue.events 
                                                             if e.event_type == "commented" and e.comment),
                'resolution_time': resolution_time,
                'is_quick_resolution': resolution_time <= 7  # Binary classification: resolved within a week
            }
            
            issue_features.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(issue_features)
        
        print(f"Prepared features for {len(df)} closed issues")
        
        return df
    
    def _analyze_resolution_time_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the distribution of resolution times.
        
        Args:
            df: DataFrame with issue features
            
        Returns:
            Dictionary with resolution time distribution statistics
        """
        print("\n=== Analyzing Resolution Time Distribution ===")
        
        # Calculate statistics
        resolution_times = df['resolution_time'].tolist()
        
        mean_time = np.mean(resolution_times)
        median_time = np.median(resolution_times)
        min_time = np.min(resolution_times)
        max_time = np.max(resolution_times)
        
        print(f"Mean resolution time: {mean_time:.2f} days")
        print(f"Median resolution time: {median_time:.2f} days")
        print(f"Min resolution time: {min_time:.2f} days")
        print(f"Max resolution time: {max_time:.2f} days")
        
        # Calculate quick vs. slow resolution percentage
        quick_pct = df['is_quick_resolution'].mean() * 100
        print(f"Percentage of issues resolved within a week: {quick_pct:.1f}%")
        
        return {
            'resolution_times': resolution_times,
            'mean_time': mean_time,
            'median_time': median_time,
            'min_time': min_time,
            'max_time': max_time,
            'quick_resolution_pct': quick_pct
        }
    
    def _analyze_feature_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes correlations between features and resolution time.
        
        Args:
            df: DataFrame with issue features
            
        Returns:
            Dictionary with feature correlation results
        """
        print("\n=== Analyzing Feature Correlations ===")
        
        # Select numeric features for correlation analysis
        numeric_features = [
            'title_length', 'description_length', 'label_count', 
            'comment_count', 'participant_count', 'resolution_time'
        ]
        
        # Calculate correlations
        corr = df[numeric_features].corr()
        
        # Extract correlations with resolution time
        resolution_correlations = corr['resolution_time'].drop('resolution_time').to_dict()
        
        # Sort by absolute correlation
        sorted_correlations = sorted(resolution_correlations.items(), 
                                    key=lambda x: abs(x[1]), 
                                    reverse=True)
        
        print("Feature correlations with resolution time:")
        for feature, correlation in sorted_correlations:
            print(f"  {feature}: {correlation:.4f}")
        
        # Analyze categorical features
        categorical_features = ['has_bug_label', 'has_feature_label', 'has_enhancement_label', 'has_code']
        
        categorical_impact = {}
        for feature in categorical_features:
            # Calculate average resolution time for each category
            avg_times = df.groupby(feature)['resolution_time'].mean().to_dict()
            
            # Calculate difference
            if True in avg_times and False in avg_times:
                diff = avg_times[True] - avg_times[False]
                diff_pct = (diff / avg_times[False]) * 100 if avg_times[False] > 0 else 0
                
                categorical_impact[feature] = {
                    'avg_time_true': avg_times.get(True),
                    'avg_time_false': avg_times.get(False),
                    'difference': diff,
                    'difference_pct': diff_pct
                }
                
                print(f"  {feature}: {diff:.2f} days difference ({diff_pct:.1f}%)")
        
        return {
            'correlation_matrix': corr.to_dict(),
            'resolution_correlations': resolution_correlations,
            'sorted_correlations': [(feature, corr) for feature, corr in sorted_correlations],
            'categorical_impact': categorical_impact
        }
    
    def _build_predictive_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Builds a predictive model for issue resolution time.
        
        Args:
            df: DataFrame with issue features
            
        Returns:
            Dictionary with model results
        """
        print("\n=== Building Predictive Model ===")
        
        if len(df) < 10:
            print("Not enough data to build a predictive model. Need at least 10 closed issues.")
            return {
                'success': False,
                'message': "Not enough data to build a predictive model"
            }
        
        # Select features and target
        feature_cols = [
            'title_length', 'description_length', 'label_count', 
            'has_bug_label', 'has_feature_label', 'has_enhancement_label',
            'comment_count', 'participant_count', 'has_code'
        ]
        
        X = df[feature_cols]
        y = df['is_quick_resolution']
        
        try:
            # Split data into training and testing sets
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
            
            # Get classification report
            report = classification_report(y_test, y_pred, 
                                          target_names=['Slow Resolution', 'Quick Resolution'],
                                          output_dict=True)
            
            # Get confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Get feature importance
            feature_importance = clf.feature_importances_
            
            # Create feature importance dictionary
            importance_dict = {feature: importance for feature, importance 
                              in zip(feature_cols, feature_importance)}
            
            # Sort by importance
            sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            
            print("Top 3 most important features:")
            for feature, importance in sorted_importance[:3]:
                print(f"  {feature}: {importance:.4f}")
            
            return {
                'success': True,
                'model': clf,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm.tolist(),
                'feature_importance': importance_dict,
                'sorted_importance': [(feature, importance) for feature, importance in sorted_importance]
            }
            
        except Exception as e:
            print(f"Error building predictive model: {e}")
            return {
                'success': False,
                'message': f"Error building predictive model: {e}"
            }
    
    def _generate_recommendations(self, df: pd.DataFrame, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates recommendations for improving issue resolution.
        
        Args:
            df: DataFrame with issue features
            model_results: Dictionary with model results
            
        Returns:
            Dictionary with recommendations
        """
        print("\n=== Generating Recommendations ===")
        
        recommendations = []
        
        # Check if model was successfully built
        if model_results.get('success', False):
            # Use feature importance to generate recommendations
            for feature, importance in model_results['sorted_importance'][:5]:
                if importance > 0.1:  # Only consider important features
                    if feature == 'comment_count':
                        recommendations.append({
                            'feature': 'comment_count',
                            'recommendation': "Encourage active discussion on issues to clarify requirements and solutions",
                            'importance': importance
                        })
                    elif feature == 'description_length':
                        recommendations.append({
                            'feature': 'description_length',
                            'recommendation': "Provide detailed issue descriptions with clear steps to reproduce",
                            'importance': importance
                        })
                    elif feature == 'has_code':
                        recommendations.append({
                            'feature': 'has_code',
                            'recommendation': "Include code examples or stack traces when reporting issues",
                            'importance': importance
                        })
                    elif feature == 'participant_count':
                        recommendations.append({
                            'feature': 'participant_count',
                            'recommendation': "Involve relevant team members in issue discussions early",
                            'importance': importance
                        })
                    elif feature == 'label_count':
                        recommendations.append({
                            'feature': 'label_count',
                            'recommendation': "Use appropriate labels to categorize issues properly",
                            'importance': importance
                        })
        
        # Add general recommendations
        recommendations.extend([
            {
                'feature': 'general',
                'recommendation': "Implement issue templates to ensure consistent information",
                'importance': 0.8
            },
            {
                'feature': 'general',
                'recommendation': "Regularly review and prioritize older unresolved issues",
                'importance': 0.7
            },
            {
                'feature': 'general',
                'recommendation': "Assign issues to specific team members for accountability",
                'importance': 0.6
            }
        ])
        
        # Compare quick vs. slow resolution issues
        quick_resolution = df[df['is_quick_resolution'] == True]
        slow_resolution = df[df['is_quick_resolution'] == False]
        
        # Compare average values for key metrics
        metrics = ['comment_count', 'participant_count', 'label_count', 'description_length']
        quick_avg = quick_resolution[metrics].mean()
        slow_avg = slow_resolution[metrics].mean()
        
        # Calculate percentage differences
        pct_diff = (quick_avg - slow_avg) / slow_avg * 100
        
        # Identify significant factors (absolute percentage difference > 20%)
        significant_factors = []
        for factor, diff in pct_diff.items():
            if abs(diff) > 20:
                direction = "lower" if diff < 0 else "higher"
                significant_factors.append({
                    'factor': factor,
                    'difference': diff,
                    'direction': direction
                })
                print(f"  {factor} is {abs(diff):.1f}% {direction} in quickly resolved issues")
        
        # Check categorical factors
        categorical_factors = []
        for factor in ['has_bug_label', 'has_feature_label', 'has_enhancement_label', 'has_code']:
            quick_pct = quick_resolution[factor].mean() * 100
            slow_pct = slow_resolution[factor].mean() * 100
            diff = quick_pct - slow_pct
            
            if abs(diff) > 10:  # Only show if difference is more than 10 percentage points
                more_less = "more" if diff > 0 else "less"
                categorical_factors.append({
                    'factor': factor,
                    'difference': diff,
                    'direction': more_less
                })
                print(f"  {factor} is {abs(diff):.1f} percentage points {more_less} common in quickly resolved issues")
        
        return {
            'recommendations': recommendations,
            'significant_factors': significant_factors,
            'categorical_factors': categorical_factors
        }
    
    def _visualize_resolution_time_distribution(self):
        """
        Creates visualizations of the resolution time distribution.
        """
        distribution = self.results['resolution_distribution']
        
        # Create histogram of resolution times
        fig = create_histogram(
            data=distribution['resolution_times'],
            title='Distribution of Issue Resolution Times',
            xlabel='Resolution Time (days)',
            ylabel='Number of Issues',
            bins=30,
            show_mean=True,
            show_median=True
        )
        
        self.save_figure(fig, "resolution_time")
        
        # Create pie chart of quick vs. slow resolution
        quick_pct = distribution['quick_resolution_pct']
        slow_pct = 100 - quick_pct
        
        fig = create_pie_chart(
            values=[quick_pct, slow_pct],
            labels=['Quick Resolution (≤7 days)', 'Slow Resolution (>7 days)'],
            title='Quick vs. Slow Issue Resolution',
            colors=['#66b3ff', '#ff9999'],
            explode=[0.1, 0]
        )
        
        self.save_figure(fig, "quick_vs_slow_issue_resolution")
    
    def _visualize_feature_correlations(self):
        """
        Creates visualizations of the feature correlations.
        """
        correlations = self.results['feature_correlations']
        df = pd.DataFrame(self.results['issue_features'])
        
        # Create correlation heatmap
        corr_matrix = pd.DataFrame(correlations['correlation_matrix'])
        
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
        plt.title('Feature Correlations')
        plt.tight_layout()
        
        self.save_figure(fig, "issue_feature_correlations")
        
        # Create bar chart of correlations with resolution time
        sorted_correlations = correlations['sorted_correlations']
        
        features = [feature.replace('_', ' ').title() for feature, _ in sorted_correlations]
        correlation_values = [corr for _, corr in sorted_correlations]
        
        fig = plt.figure(figsize=(10, 6))
        bars = plt.barh(features, correlation_values)
        
        # Color bars based on correlation direction
        for i, bar in enumerate(bars):
            bar.set_color('#66b3ff' if correlation_values[i] > 0 else '#ff9999')
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('Correlation with Resolution Time')
        plt.title('Feature Correlations with Resolution Time')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(fig, "resolution_time_features")
        
        # Create individual feature plots
        fig = plt.figure(figsize=(15, 10))
        
        # Select top 4 numeric features by correlation
        numeric_features = [f for f, _ in sorted_correlations if f in ['comment_count', 'participant_count', 'label_count', 'description_length']][:4]
        
        for i, feature in enumerate(numeric_features):
            plt.subplot(2, 2, i+1)
            plt.scatter(df[feature], df['resolution_time'], alpha=0.5)
            
            # Add trend line
            z = np.polyfit(df[feature], df['resolution_time'], 1)
            p = np.poly1d(z)
            plt.plot(sorted(df[feature]), p(sorted(df[feature])), "r--", alpha=0.8)
            
            plt.title(f'{feature.replace("_", " ").title()} vs. Resolution Time')
            plt.xlabel(feature.replace('_', ' ').title())
            plt.ylabel('Resolution Time (days)')
            plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        self.save_figure(fig, "issue_resolution_times")
    
    def _visualize_model_results(self):
        """
        Creates visualizations of the model results.
        """
        model_results = self.results['model_results']
        
        # Skip if model wasn't successfully built
        if not model_results.get('success', False):
            print("Model wasn't successfully built, skipping visualizations")
            return
        
        # Create confusion matrix visualization
        cm = np.array(model_results['confusion_matrix'])
        
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Slow Resolution', 'Quick Resolution'],
                   yticklabels=['Slow Resolution', 'Quick Resolution'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        self.save_figure(fig, "confusion_matrix")
        
        # Create feature importance visualization
        sorted_importance = model_results['sorted_importance']
        
        features = [feature.replace('_', ' ').title() for feature, _ in sorted_importance]
        importance_values = [importance for _, importance in sorted_importance]
        
        fig = plt.figure(figsize=(10, 8))
        plt.barh(features, importance_values, color='skyblue')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance for Predicting Quick Resolution')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(fig, "feature_importance")
    
    def _visualize_recommendations(self):
        """
        Creates visualizations of the recommendations.
        """
        recommendations = self.results['recommendations']
        
        # Create bar chart of recommendation importance
        sorted_recommendations = sorted(recommendations['recommendations'], 
                                       key=lambda x: x['importance'], 
                                       reverse=True)
        
        rec_texts = [rec['recommendation'][:50] + '...' if len(rec['recommendation']) > 50 
                    else rec['recommendation'] for rec in sorted_recommendations]
        importance_values = [rec['importance'] for rec in sorted_recommendations]
        
        fig = plt.figure(figsize=(12, 8))
        plt.barh(rec_texts, importance_values, color='lightgreen')
        plt.xlabel('Importance')
        plt.title('Recommendations for Improving Issue Resolution')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        self.save_figure(fig, "recommendations")
    
    def _generate_report(self):
        """
        Generates a report of the analysis results.
        """
        # Create report
        report = Report("Issue Resolution Prediction Report", self.results_dir)
        
        # Add introduction
        intro = "Analysis of issue resolution time and predictive modeling"
        report.add_section("Introduction", intro)
        
        # Add resolution time distribution section
        distribution = self.results['resolution_distribution']
        
        distribution_content = "Resolution time distribution:\n\n"
        distribution_content += f"Mean resolution time: {distribution['mean_time']:.2f} days\n"
        distribution_content += f"Median resolution time: {distribution['median_time']:.2f} days\n"
        distribution_content += f"Minimum resolution time: {distribution['min_time']:.2f} days\n"
        distribution_content += f"Maximum resolution time: {distribution['max_time']:.2f} days\n\n"
        distribution_content += f"Percentage of issues resolved within a week: {distribution['quick_resolution_pct']:.1f}%\n"
        
        report.add_section("Resolution Time Distribution", distribution_content)
        
        # Add feature correlations section
        correlations = self.results['feature_correlations']
        
        correlation_content = "Feature correlations with resolution time:\n\n"
        for feature, correlation in correlations['sorted_correlations']:
            correlation_content += f"{feature.replace('_', ' ').title()}: {correlation:.4f}\n"
        
        correlation_content += "\nImpact of categorical features:\n\n"
        for feature, impact in correlations['categorical_impact'].items():
            feature_name = feature.replace('has_', '').replace('_', ' ').title()
            diff = impact['difference']
            diff_pct = impact['difference_pct']
            direction = "increases" if diff > 0 else "decreases"
            correlation_content += f"{feature_name}: {direction} resolution time by {abs(diff):.2f} days ({abs(diff_pct):.1f}%)\n"
        
        report.add_section("Feature Correlations", correlation_content)
        
        # Add model results section
        model_results = self.results['model_results']
        
        model_content = "Predictive model results:\n\n"
        
        if model_results.get('success', False):
            model_content += f"Model accuracy: {model_results['accuracy']:.2f}\n\n"
            
            model_content += "Classification report:\n\n"
            report_dict = model_results['classification_report']
            for class_name, metrics in report_dict.items():
                if isinstance(metrics, dict):
                    model_content += f"{class_name}:\n"
                    model_content += f"  Precision: {metrics['precision']:.2f}\n"
                    model_content += f"  Recall: {metrics['recall']:.2f}\n"
                    model_content += f"  F1-score: {metrics['f1-score']:.2f}\n"
                    model_content += f"  Support: {metrics['support']}\n\n"
            
            model_content += "Feature importance:\n\n"
            for feature, importance in model_results['sorted_importance']:
                model_content += f"{feature.replace('_', ' ').title()}: {importance:.4f}\n"
        else:
            model_content += model_results.get('message', "Model building failed")
        
        report.add_section("Predictive Model", model_content)
        
        # Add recommendations section
        recommendations = self.results['recommendations']
        
        rec_content = "Recommendations for improving issue resolution:\n\n"
        
        for rec in sorted(recommendations['recommendations'], key=lambda x: x['importance'], reverse=True):
            rec_content += f"- {rec['recommendation']}\n"
        
        rec_content += "\nSignificant factors affecting resolution time:\n\n"
        
        for factor in recommendations['significant_factors']:
            factor_name = factor['factor'].replace('_', ' ').title()
            direction = factor['direction']
            diff = abs(factor['difference'])
            rec_content += f"- {factor_name} is {diff:.1f}% {direction} in quickly resolved issues\n"
        
        for factor in recommendations['categorical_factors']:
            factor_name = factor['factor'].replace('has_', '').replace('_', ' ').title()
            direction = factor['direction']
            diff = abs(factor['difference'])
            rec_content += f"- {factor_name} is {diff:.1f} percentage points {direction} common in quickly resolved issues\n"
        
        report.add_section("Recommendations", rec_content)
        
        # Save report
        report.save_text_report("result.txt")
        report.save_markdown_report("report.md")


if __name__ == '__main__':
    # Invoke run method when running this module directly
    IssueResolutionPrediction().run()
