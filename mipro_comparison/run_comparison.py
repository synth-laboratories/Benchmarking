#!/usr/bin/env python3
"""
Run MIPRO Comparison: Synth vs DSPy MIPROv2

This script orchestrates running both MIPRO implementations on Banking77
and compares the results.

Usage:
    python run_comparison.py --all
    python run_comparison.py --synth-only
    python run_comparison.py --dspy-only
"""

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path


def run_synth_mipro(args) -> dict:
    """Run Synth MIPRO benchmark."""
    print("\n" + "="*70)
    print("RUNNING SYNTH MIPRO")
    print("="*70)
    
    cmd = [
        sys.executable, 
        str(Path(__file__).parent / "run_synth_mipro_banking77.py"),
        "--model", args.model,
        "--rollouts", str(args.rollouts),
        "--train-size", str(args.train_size),
        "--val-size", str(args.val_size),
        "--mode", "offline",
    ]
    
    if args.local:
        cmd.append("--local")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            return {"status": "failed", "error": f"Exit code: {result.returncode}"}
        
        # Find the most recent result file
        results_dir = Path(__file__).parent / "results"
        result_files = sorted(results_dir.glob("banking77_synth_mipro_*.json"), reverse=True)
        if result_files:
            with open(result_files[0]) as f:
                return json.load(f)
        
        return {"status": "completed", "note": "No result file found"}
        
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "Timeout"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def run_dspy_mipro(args) -> dict:
    """Run DSPy MIPROv2 benchmark."""
    print("\n" + "="*70)
    print("RUNNING DSPY MIPROv2")
    print("="*70)
    
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_dspy_mipro_banking77.py"),
        "--model", args.dspy_model,
        "--trials", str(args.trials),
        "--train-size", str(args.train_size),
        "--val-size", str(args.val_size),
        "--auto", args.auto,
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            return {"status": "failed", "error": f"Exit code: {result.returncode}"}
        
        # Find the most recent result file
        results_dir = Path(__file__).parent / "results"
        result_files = sorted(results_dir.glob("banking77_dspy_mipro_*.json"), reverse=True)
        if result_files:
            with open(result_files[0]) as f:
                return json.load(f)
        
        return {"status": "completed", "note": "No result file found"}
        
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": "Timeout"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


def print_comparison(synth_result: dict, dspy_result: dict):
    """Print comparison of results."""
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Synth MIPRO':<20} {'DSPy MIPROv2':<20}")
    print("-"*70)
    
    # Status
    synth_status = synth_result.get("status", "unknown")
    dspy_status = dspy_result.get("status", "unknown")
    print(f"{'Status':<30} {synth_status:<20} {dspy_status:<20}")
    
    # Elapsed time
    synth_time = synth_result.get("elapsed_seconds", 0)
    dspy_time = dspy_result.get("elapsed_seconds", 0)
    print(f"{'Elapsed Time (s)':<30} {synth_time:<20.1f} {dspy_time:<20.1f}")
    
    # Baseline accuracy
    synth_baseline = synth_result.get("results", {}).get("baseline_accuracy")
    dspy_baseline = dspy_result.get("results", {}).get("baseline_accuracy")
    if synth_baseline is not None or dspy_baseline is not None:
        synth_bl_str = f"{synth_baseline:.1%}" if synth_baseline else "N/A"
        dspy_bl_str = f"{dspy_baseline:.1%}" if dspy_baseline else "N/A"
        print(f"{'Baseline Accuracy':<30} {synth_bl_str:<20} {dspy_bl_str:<20}")
    
    # Optimized accuracy / best score
    synth_best = synth_result.get("results", {}).get("best_score")
    dspy_opt = dspy_result.get("results", {}).get("optimized_accuracy")
    if synth_best is not None or dspy_opt is not None:
        synth_opt_str = f"{synth_best:.1%}" if synth_best else "N/A"
        dspy_opt_str = f"{dspy_opt:.1%}" if dspy_opt else "N/A"
        print(f"{'Optimized Accuracy':<30} {synth_opt_str:<20} {dspy_opt_str:<20}")
    
    # Improvement
    dspy_improvement = dspy_result.get("results", {}).get("improvement")
    if dspy_improvement is not None:
        synth_imp_str = "N/A"  # Synth doesn't have explicit baseline comparison in same run
        dspy_imp_str = f"{dspy_improvement:+.1%}"
        print(f"{'Improvement':<30} {synth_imp_str:<20} {dspy_imp_str:<20}")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Run MIPRO comparison benchmark")
    parser.add_argument("--all", action="store_true", help="Run both benchmarks")
    parser.add_argument("--synth-only", action="store_true", help="Run only Synth MIPRO")
    parser.add_argument("--dspy-only", action="store_true", help="Run only DSPy MIPROv2")
    
    # Synth MIPRO args
    parser.add_argument("--local", action="store_true", help="Use localhost backend for Synth")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="Model for Synth MIPRO")
    parser.add_argument("--rollouts", type=int, default=50, help="Rollouts for Synth MIPRO")
    
    # DSPy MIPROv2 args
    parser.add_argument("--dspy-model", type=str, default="gpt-4o-mini", help="Model for DSPy MIPROv2")
    parser.add_argument("--trials", type=int, default=20, help="Trials for DSPy MIPROv2")
    parser.add_argument("--auto", type=str, default="light", choices=["light", "medium", "heavy"],
                        help="MIPROv2 auto preset")
    
    # Common args
    parser.add_argument("--train-size", type=int, default=100, help="Training set size")
    parser.add_argument("--val-size", type=int, default=50, help="Validation set size")
    
    args = parser.parse_args()
    
    # Default to --all if no specific option is given
    if not (args.all or args.synth_only or args.dspy_only):
        args.all = True
    
    print("="*70)
    print("MIPRO COMPARISON BENCHMARK: Banking77")
    print("="*70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train size: {args.train_size}")
    print(f"Val size: {args.val_size}")
    
    synth_result = {}
    dspy_result = {}
    
    # Run Synth MIPRO
    if args.all or args.synth_only:
        synth_result = run_synth_mipro(args)
    
    # Run DSPy MIPROv2
    if args.all or args.dspy_only:
        dspy_result = run_dspy_mipro(args)
    
    # Print comparison if we ran both
    if args.all:
        print_comparison(synth_result, dspy_result)
    
    # Save combined results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    combined_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "train_size": args.train_size,
            "val_size": args.val_size,
        },
        "synth_mipro": synth_result if synth_result else None,
        "dspy_miprov2": dspy_result if dspy_result else None,
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"mipro_comparison_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nCombined results saved to {output_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
