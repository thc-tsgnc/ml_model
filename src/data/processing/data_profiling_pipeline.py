# src/data_processing/data_profiling_pipeline.py

import pandas as pd
from ydata_profiling import ProfileReport
from pathlib import Path
import json
from datetime import datetime
import os

def profile_dataset(df: pd.DataFrame, report_title: str = "Pandas Profiling Report", minimal: bool = False) -> ProfileReport:
    """
    Generate a comprehensive profile of the dataset using pandas-profiling.
    
    :param df: Input DataFrame
    :param report_title: Title of the profiling report
    :param minimal: If True, generates a minimal report (faster but less detailed)
    :return: ProfileReport object
    """
    profile = ProfileReport(df, title=report_title, minimal=minimal)
    return profile

def save_profile_results(profile: ProfileReport, output_filename: str, output_path: str, save_html: bool = False) -> dict:
    """
    Save the profiling results to a specified location.
    
    :param profile: ProfileReport object
    :param output_filename: Name of the output file (without extension)
    :param output_path: Path to save the results
    :param save_html: If True, saves an HTML report in addition to JSON and CSV
    :return: Dictionary containing paths to saved files and metadata
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save HTML report if requested
    if save_html:
        html_path = output_dir / f"{output_filename}.html"
        profile.to_file(html_path)
        saved_files['html_report'] = str(html_path)

    # Save JSON report
    json_path = output_dir / f"{output_filename}.json"
    profile.to_file(json_path)
    saved_files['json_report'] = str(json_path)

    # Save basic statistics as CSV
    csv_path = output_dir / f"{output_filename}_basic_stats.csv"
    profile.description_set['table'].to_csv(csv_path)
    saved_files['csv_report'] = str(csv_path)

    # Create metadata
    metadata = {
        "dataset_shape": profile.df.shape,
        "number_of_variables": len(profile.df.columns),
        "number_of_observations": len(profile.df),
        "report_title": profile.title,
        "minimal_report": profile.minimal,
        "saved_files": saved_files
    }
    
    # Save metadata
    metadata_path = output_dir / f"{output_filename}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    return metadata

def run_profiling(df: pd.DataFrame, output_path: str = "results/profiling", data_playtype: str = None, minimal: bool = False, save_html: bool = False) -> dict:
    """
    Run the entire profiling process, including generating the profile and saving the results.
    
    :param df: Input DataFrame
    :param output_path: Path to save the results
    :param data_playtype: Type of data play (used in filename and report title if provided)
    :param minimal: If True, generates a minimal report (faster but less detailed)
    :param save_html: If True, saves an HTML report in addition to JSON and CSV
    :return: Dictionary containing paths to saved files and metadata
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"profile_report_{timestamp}"
    report_title = f"Pandas Profiling Report {timestamp}"
    
    if data_playtype:
        base_name = f"{data_playtype}_{base_name}"
        report_title = f"{data_playtype} {report_title}"
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    profile = profile_dataset(df, report_title=report_title, minimal=minimal)
    
    metadata = save_profile_results(
        profile,
        output_filename=base_name,
        output_path=output_path,
        save_html=save_html
    )
    
    return metadata
