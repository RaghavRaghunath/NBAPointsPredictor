#!/usr/bin/env python3
"""
Script to train models for all supported statistics.

Usage:
    python train_all_models.py [--optimize] [--trials N] [--stats STAT1,STAT2,...]

Example:
    python train_all_models.py --optimize --trials 10 --stats PTS,AST,REB
"""
import argparse
import importlib
import sys
import os
import time
from typing import List, Dict, Any

# Add the src directory to the path so we can import the training script
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src', 'models'))

# Default stats to train
DEFAULT_STATS = ['PTS', 'AST', 'REB', 'STL', 'BLK']

def train_models(stats: List[str], optimize: bool = False, n_trials: int = 10) -> Dict[str, Any]:
    """
    Train models for the specified statistics.
    
    Args:
        stats: List of statistics to train models for (e.g., ['PTS', 'AST'])
        optimize: Whether to run hyperparameter optimization
        n_trials: Number of trials for hyperparameter optimization
        
    Returns:
        Dictionary with training results for each stat
    """
    results = {}
    
    # Import the training module
    try:
        train_module = importlib.import_module('train_rnn_model')
    except ImportError as e:
        print(f"❌ Error importing training module: {e}")
        return {}
    
    for stat in stats:
        print(f"\n{'='*80}")
        print(f"TRAINING MODEL FOR {stat}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # Call the main function from the training script
            result = train_module.main(
                optimize_hyperparams=optimize,
                n_trials=n_trials,
                target_stat=stat
            )
            
            # Store the results
            results[stat] = {
                'success': True,
                'time_seconds': time.time() - start_time,
                'model_path': result.get('model_path', ''),
                'metrics': result.get('metrics', {})
            }
            
            print(f"✅ Successfully trained {stat} model in {results[stat]['time_seconds']:.1f} seconds")
            print(f"   Model saved to: {results[stat]['model_path']}")
            
        except Exception as e:
            print(f"❌ Error training {stat} model: {e}")
            results[stat] = {
                'success': False,
                'error': str(e),
                'time_seconds': time.time() - start_time
            }
    
    return results

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Train models for NBA player statistics prediction')
    parser.add_argument('--optimize', action='store_true', help='Run hyperparameter optimization')
    parser.add_argument('--trials', type=int, default=10, help='Number of hyperparameter optimization trials')
    parser.add_argument('--stats', type=str, default=','.join(DEFAULT_STATS),
                       help=f'Comma-separated list of stats to train (default: {DEFAULT_STATS})')
    
    args = parser.parse_args()
    
    # Parse stats
    stats = [s.strip().upper() for s in args.stats.split(',') if s.strip()]
    
    # Validate stats
    for stat in stats:
        if stat not in DEFAULT_STATS:
            print(f"⚠️  Warning: Unsupported stat '{stat}'. Supported stats are: {', '.join(DEFAULT_STATS)}")
    
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING FOR STATS: {', '.join(stats)}")
    print(f"Hyperparameter optimization: {'ENABLED' if args.optimize else 'DISABLED'}")
    if args.optimize:
        print(f"Number of optimization trials: {args.trials}")
    print(f"{'='*80}\n")
    
    # Train the models
    results = train_models(
        stats=stats,
        optimize=args.optimize,
        n_trials=args.trials
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results.values() if r.get('success', False))
    print(f"\nCompleted training {success_count}/{len(stats)} models successfully\n")
    
    for stat, result in results.items():
        status = "✅ SUCCESS" if result.get('success') else "❌ FAILED"
        print(f"{stat}: {status} ({result.get('time_seconds', 0):.1f}s)")
        
        if not result.get('success'):
            print(f"   Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"   Model: {result.get('model_path', 'Unknown')}")
            print(f"   Metrics: {result.get('metrics', {})}")
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
