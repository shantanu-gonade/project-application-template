#!/usr/bin/env python3
"""
Starting point of the application. This module is invoked from
the command line to run the analyses.
"""

import argparse
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.core import config
from src.analysis.label_analysis import LabelAnalysis
from src.analysis.contributor_analysis import ContributorAnalysis
from src.analysis.timeline_analysis import TimelineAnalysis
from src.analysis.network_analysis import IssueNetworkAnalysis
from src.analysis.complexity_analysis import IssueComplexityAnalysis
from src.analysis.resolution_prediction import IssueResolutionPrediction


def parse_args():
    """
    Parses the command line arguments that were provided along
    with the python command. The --feature flag must be provided as
    that determines what analysis to run. Optionally, you can pass in
    a user and/or a label to run analysis focusing on specific issues.

    You can also add more command line arguments following the pattern
    below.
    """
    ap = argparse.ArgumentParser("run.py")

    # Required parameter specifying what analysis to run
    ap.add_argument('--feature', '-f', type=int, required=True,
                    help='Which of the three features to run')

    # Optional parameter for analyses focusing on a specific user (i.e., contributor)
    ap.add_argument('--user', '-u', type=str, required=False,
                    help='Optional parameter for analyses focusing on a specific user')

    # Optional parameter for analyses focusing on a specific label
    ap.add_argument('--label', '-l', type=str, required=False,
                    help='Optional parameter for analyses focusing on a specific label')

    return ap.parse_args()


def main():
    """
    Main entry point for the application.
    """
    # Parse feature to call from command line arguments
    args = parse_args()
    
    # Add arguments to config so that they can be accessed in other parts of the application
    config.overwrite_from_args(args)

    # Run the feature specified in the --feature flag
    if args.feature == 1:
        LabelAnalysis().run()
    elif args.feature == 2:
        ContributorAnalysis().run()
    elif args.feature == 3:
        TimelineAnalysis().run()
    elif args.feature == 4:
        IssueNetworkAnalysis().run()
    elif args.feature == 5:
        IssueComplexityAnalysis().run()
    elif args.feature == 6:
        IssueResolutionPrediction().run()
    else:
        print('Need to specify which feature to run with --feature flag.')
        print('Available features:')
        print('  1: Label Analysis')
        print('  2: Contributor Analysis')
        print('  3: Timeline Analysis')
        print('  4: Issue Network Analysis')
        print('  5: Issue Complexity Analysis')
        print('  6: Issue Resolution Prediction')


if __name__ == '__main__':
    main()
