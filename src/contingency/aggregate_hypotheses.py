#!/usr/bin/env python3
"""
Contingency Plan: Aggregate Hypotheses from All Session States

This script parses all session_state.json files in runtime/runs directory,
extracts hypotheses, and saves them to a compiled file for manual feature engineering.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
import argparse

def load_session_state(session_path: Path) -> Dict[str, Any]:
    """Load session state from JSON file."""
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {session_path}: {e}")
        return {}

def extract_hypotheses_from_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Extract all hypotheses from all run directories."""
    all_hypotheses = []
    run_metadata = []
    
    print(f"Scanning runs directory: {runs_dir}")
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        session_file = run_dir / "session_state.json"
        if not session_file.exists():
            print(f"No session_state.json in {run_dir.name}")
            continue
            
        print(f"Processing {run_dir.name}...")
        session_data = load_session_state(session_file)
        
        if not session_data:
            continue
            
        # Extract hypotheses
        hypotheses = session_data.get("hypotheses", [])
        if hypotheses:
            print(f"  Found {len(hypotheses)} hypotheses")
            for hypothesis in hypotheses:
                # Add metadata about which run this came from
                hypothesis_with_meta = {
                    "run_id": run_dir.name,
                    **hypothesis
                }
                all_hypotheses.append(hypothesis_with_meta)
        else:
            print(f"  No hypotheses found")
            
        # Track run metadata
        run_info = {
            "run_id": run_dir.name,
            "hypotheses_count": len(hypotheses),
            "insights_count": len(session_data.get("insights", [])),
            "candidate_features_count": len(session_data.get("candidate_features", [])),
            "has_data": bool(session_data)
        }
        run_metadata.append(run_info)
    
    return all_hypotheses, run_metadata

def save_compiled_hypotheses(hypotheses: List[Dict[str, Any]], 
                           metadata: List[Dict[str, Any]], 
                           output_file: Path):
    """Save compiled hypotheses to JSON file."""
    compiled_data = {
        "total_hypotheses": len(hypotheses),
        "total_runs_processed": len(metadata),
        "runs_with_hypotheses": len([r for r in metadata if r["hypotheses_count"] > 0]),
        "compilation_timestamp": "2025-06-17T11:30:00+02:00",
        "run_metadata": metadata,
        "hypotheses": hypotheses
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(compiled_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompiled hypotheses saved to: {output_file}")
    print(f"Total hypotheses: {len(hypotheses)}")
    print(f"Runs processed: {len(metadata)}")
    print(f"Runs with hypotheses: {len([r for r in metadata if r['hypotheses_count'] > 0])}")

def print_hypothesis_summary(hypotheses: List[Dict[str, Any]]):
    """Print a summary of the hypotheses found."""
    if not hypotheses:
        print("No hypotheses found across all runs.")
        return
        
    print(f"\n=== HYPOTHESIS SUMMARY ===")
    for i, hyp in enumerate(hypotheses, 1):
        print(f"\n{i}. Run: {hyp.get('run_id', 'unknown')}")
        print(f"   ID: {hyp.get('id', 'no-id')}")
        print(f"   Summary: {hyp.get('summary', 'No summary')}")
        print(f"   Rationale: {hyp.get('rationale', 'No rationale')[:100]}...")

def main():
    parser = argparse.ArgumentParser(description="Aggregate hypotheses from all VULCAN runs")
    parser.add_argument("--runs_dir", type=str, default="/root/fuegoRecommender/runtime/runs",
                       help="Path to runs directory")
    parser.add_argument("--output", type=str, default="/root/fuegoRecommender/src/contingency/compiled_hypotheses.json",
                       help="Output file for compiled hypotheses")
    parser.add_argument("--summary", action="store_true", help="Print detailed summary of hypotheses")
    
    args = parser.parse_args()
    
    runs_dir = Path(args.runs_dir)
    output_file = Path(args.output)
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        return 1
        
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract hypotheses
    hypotheses, metadata = extract_hypotheses_from_runs(runs_dir)
    
    # Save compiled results
    save_compiled_hypotheses(hypotheses, metadata, output_file)
    
    # Print summary if requested
    if args.summary:
        print_hypothesis_summary(hypotheses)
    
    return 0

if __name__ == "__main__":
    exit(main())
