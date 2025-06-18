import os
import joblib
import optuna
import plotly.io as pio
from pathlib import Path

def find_study_files(search_dir, suffix='.pkl'):
    """Recursively find all Optuna study pickle files in a directory."""
    study_files = []
    for root, _, files in os.walk(search_dir):
        for file in files:
            if file.endswith(suffix):
                study_files.append(os.path.join(root, file))
    return study_files

def safe_plot_and_save(plot_func, study, out_path, **kwargs):
    try:
        fig = plot_func(study, **kwargs)
        pio.write_html(fig, out_path)
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Failed to plot {plot_func.__name__} for {out_path}: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot all Optuna studies in a directory.")
    parser.add_argument('--search_dir', type=str, default='../evaluation_results/optuna_studies', help='Directory to search for .pkl Optuna studies')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Base directory to save plots')
    args = parser.parse_args()

    study_files = find_study_files(args.search_dir)
    if not study_files:
        print(f"No study files found in {args.search_dir}")
        return

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for study_path in study_files:
        study_name = Path(study_path).stem.replace('.pkl','')
        out_dir = Path(args.output_dir) / study_name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            study = joblib.load(study_path)
        except Exception as e:
            print(f"Could not load {study_path}: {e}")
            continue
        print(f"Loaded study: {study_name} ({study_path})")
        # Standard plots
        safe_plot_and_save(optuna.visualization.plot_optimization_history, study, out_dir/'optimization_history.html')
        safe_plot_and_save(optuna.visualization.plot_param_importances, study, out_dir/'param_importances.html')
        safe_plot_and_save(optuna.visualization.plot_slice, study, out_dir/'slice.html')
        safe_plot_and_save(optuna.visualization.plot_contour, study, out_dir/'contour.html')
        safe_plot_and_save(optuna.visualization.plot_parallel_coordinate, study, out_dir/'parallel_coordinate.html')
        safe_plot_and_save(optuna.visualization.plot_evaluations, study, out_dir/'evaluations.html')
        safe_plot_and_save(optuna.visualization.plot_edf, study, out_dir/'edf.html')
        # Optionally add more plots as needed

if __name__ == '__main__':
    main()
