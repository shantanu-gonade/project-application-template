# GitHub Issue Analysis for Python-Poetry

This application analyzes GitHub issues from the [poetry](https://github.com/python-poetry/poetry) Open Source project and generates interesting insights through various analyses and visualizations.

## Features

The application implements several different analyses:

1. **Label Analysis** (Feature 1): Analyzes issues based on labels

   - Shows label distribution across all issues
   - Identifies the most active labels (by average comments per issue)
   - Analyzes resolution time for issues with different labels
   - Can focus on a specific label using the `--label` parameter

2. **Contributor Analysis** (Feature 2): Analyzes contributor activity

   - Identifies top contributors by different metrics (issue creation, comments, closing)
   - Analyzes which labels different contributors focus on
   - Shows contributor activity patterns
   - Can focus on a specific contributor using the `--user` parameter

3. **Timeline Analysis** (Feature 3): Analyzes issue activity over time

   - Shows issue creation and closing patterns over time
   - Analyzes issue lifecycle (time to first response, time to resolution)
   - Identifies activity trends and patterns (by day of week, hour of day)

4. **Network Analysis** (Feature 4): Analyzes network relationships

   - Creates label co-occurrence network
   - Creates contributor collaboration network
   - Visualizes network relationships using graph visualizations

5. **Complexity Analysis** (Feature 5): Analyzes issue complexity

   - Calculates complexity scores based on various metrics
   - Analyzes correlation between complexity and resolution time
   - Identifies most complex issues and labels

6. **Resolution Prediction** (Feature 6): Predicts issue resolution time
   - Builds a machine learning model to predict quick vs. slow resolution
   - Identifies factors that contribute to faster resolution
   - Provides recommendations for improving issue resolution

## Project Structure

The application is organized into a modular structure:

```
github-issue-analysis/
├── src/
│   ├── core/
│   │   ├── model.py          # Data models (Issue, Event, State)
│   │   ├── data_loader.py    # Data loading functionality
│   │   └── config.py         # Configuration handling
│   ├── analysis/
│   │   ├── base.py           # Base analysis class with common functionality
│   │   ├── label_analysis.py
│   │   ├── contributor_analysis.py
│   │   ├── timeline_analysis.py
│   │   ├── network_analysis.py
│   │   ├── complexity_analysis.py
│   │   └── resolution_prediction.py
│   ├── visualization/
│   │   ├── plotting.py       # Common plotting functions
│   │   └── report.py         # Report generation utilities
│   └── utils/
│       └── helpers.py        # Common utility functions
├── results/                  # Directory for analysis results
├── run.py                    # Main entry point
├── config.json               # Configuration file
└── requirements.txt          # Dependencies
```

## Setup

### Install dependencies

In the root directory of the application, create a virtual environment, activate that environment, and install the dependencies:

```
pip install -r requirements.txt
```

### Configure the data file

The application uses a JSON file containing GitHub issues data. Update the `config.json` file with the path to your data file:

```json
{
  "ENPM611_PROJECT_DATA_PATH": "path/to/poetry_data.json"
}
```

Alternatively, you can set an environment variable:

```
export ENPM611_PROJECT_DATA_PATH=path/to/poetry_data.json
```

## Running the Analyses

### Label Analysis

To run the label analysis on all issues:

```
python run.py --feature 1
```

To focus on issues with a specific label:

```
python run.py --feature 1 --label "bug"
```

### Contributor Analysis

To run the contributor analysis for all contributors:

```
python run.py --feature 2
```

To focus on a specific contributor:

```
python run.py --feature 2 --user "username"
```

### Timeline Analysis

To run the timeline analysis:

```
python run.py --feature 3
```

### Network Analysis

To run the network analysis:

```
python run.py --feature 4
```

### Complexity Analysis

To run the complexity analysis:

```
python run.py --feature 5
```

### Resolution Prediction

To run the resolution prediction:

```
python run.py --feature 6
```

## Analysis Results

Each analysis generates both textual output to the console and visual charts using matplotlib. The results are saved in the `results/` directory, organized by analysis type:

- Text reports (`.txt` and `.md` files)
- JSON data files
- Visualizations (`.png` files)

## Extending the Application

To add a new analysis:

1. Create a new analysis class in the `src/analysis/` directory, inheriting from `BaseAnalysis`
2. Implement the required methods: `analyze()`, `visualize()`, and `save_results()`
3. Add the new analysis to the `run.py` file
4. Update the README.md with information about the new analysis
