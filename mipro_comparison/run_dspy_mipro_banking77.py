#!/usr/bin/env python3
"""
Run Banking77 MIPROv2 Comparison using DSPy

This script runs MIPROv2 optimization on Banking77 using the DSPy library.

Usage:
    python run_dspy_mipro_banking77.py
    python run_dspy_mipro_banking77.py --trials 20 --model gpt-4o-mini
    python run_dspy_mipro_banking77.py --auto light
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import dspy
    from datasets import load_dataset
except ImportError as e:
    print(f"Error: Missing dependencies. {e}")
    print("Install with: pip install dspy-ai datasets")
    sys.exit(1)


# Parse arguments
parser = argparse.ArgumentParser(description="Run Banking77 MIPROv2 using DSPy")
parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
parser.add_argument("--trials", type=int, default=20, help="Number of optimization trials")
parser.add_argument("--train-size", type=int, default=100, help="Training set size")
parser.add_argument("--val-size", type=int, default=50, help="Validation set size")
parser.add_argument("--auto", type=str, default="light", choices=["light", "medium", "heavy"], 
                    help="MIPROv2 auto preset")
parser.add_argument("--max-bootstrapped-demos", type=int, default=4, help="Max bootstrapped demos")
parser.add_argument("--max-labeled-demos", type=int, default=4, help="Max labeled demos")
parser.add_argument("--output", type=str, default="banking77_dspy_mipro_results.json", help="Output file")
args = parser.parse_args()

print("="*60)
print("Banking77 MIPROv2 Optimization with DSPy")
print("="*60)
print(f"Model: {args.model}")
print(f"Trials: {args.trials}")
print(f"Auto preset: {args.auto}")
print(f"Train size: {args.train_size}")
print(f"Val size: {args.val_size}")


# Banking77 labels
BANKING77_LABELS = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
    "automatic_top_up", "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed",
    "cancel_transfer", "card_about_to_expire", "card_acceptance", "card_arrival",
    "card_delivery_estimate", "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card",
    "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal",
    "pending_top_up", "pending_transfer", "pin_blocked", "receiving_money",
    "Refund_not_showing_up", "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account", "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account", "transfer_not_received_by_recipient",
    "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]


def load_banking77_dataset(train_size: int, val_size: int):
    """Load Banking77 dataset and convert to DSPy format."""
    print("\nLoading Banking77 dataset...")
    
    ds_train = load_dataset("banking77", split="train", trust_remote_code=False)
    ds_test = load_dataset("banking77", split="test", trust_remote_code=False)
    
    label_names = ds_train.features["label"].names
    
    def convert_to_examples(ds, max_size: int):
        examples = []
        for i, row in enumerate(ds):
            if i >= max_size:
                break
            label_idx = int(row["label"])
            label_text = label_names[label_idx]
            examples.append(dspy.Example(
                query=str(row["text"]),
                intent=label_text,
            ).with_inputs("query"))
        return examples
    
    trainset = convert_to_examples(ds_train, train_size)
    valset = convert_to_examples(ds_test, val_size)
    
    print(f"  Train: {len(trainset)} examples")
    print(f"  Val: {len(valset)} examples")
    
    return trainset, valset, label_names


def banking77_metric(example, prediction, trace=None):
    """Metric function for Banking77 classification."""
    expected = example.intent.lower().replace("_", " ").strip()
    predicted = prediction.intent.lower().replace("_", " ").strip() if prediction.intent else ""
    return expected == predicted


class Banking77Classifier(dspy.Signature):
    """Classify a customer banking query into one of the banking intent categories."""
    
    query: str = dspy.InputField(desc="The customer's banking query")
    intent: str = dspy.OutputField(desc="The predicted banking intent label (e.g., 'card_arrival', 'lost_or_stolen_card')")


def create_baseline_program():
    """Create the baseline DSPy program for Banking77 classification."""
    return dspy.ChainOfThought(Banking77Classifier)


def evaluate_program(program, dataset, metric_fn):
    """Evaluate a program on a dataset."""
    evaluator = dspy.Evaluate(
        devset=dataset,
        metric=metric_fn,
        num_threads=4,
        display_progress=True,
        return_all_scores=True,
    )
    result = evaluator(program)
    
    # Extract accuracy from result
    if isinstance(result, tuple):
        accuracy = result[0] / 100.0  # Convert percentage to decimal
    else:
        accuracy = float(result) / 100.0
    
    return accuracy


def main():
    # Configure DSPy
    print("\nConfiguring DSPy...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)
    
    lm = dspy.LM(model=f"openai/{args.model}", api_key=api_key)
    dspy.configure(lm=lm)
    print(f"  Using model: {args.model}")
    
    # Load dataset
    trainset, valset, label_names = load_banking77_dataset(args.train_size, args.val_size)
    
    # Create baseline program
    print("\nCreating baseline program...")
    baseline_program = create_baseline_program()
    
    # Evaluate baseline
    print("\nEvaluating baseline program...")
    start_time = time.time()
    baseline_accuracy = evaluate_program(baseline_program, valset, banking77_metric)
    baseline_time = time.time() - start_time
    print(f"  Baseline accuracy: {baseline_accuracy:.1%}")
    print(f"  Evaluation time: {baseline_time:.1f}s")
    
    # Run MIPROv2 optimization
    print("\n" + "="*60)
    print("Running MIPROv2 Optimization")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # Note: When using 'auto' mode, num_trials is determined by the preset
        # auto='light' ~7 trials, auto='medium' ~20 trials, auto='heavy' ~50 trials
        optimizer = dspy.MIPROv2(
            metric=banking77_metric,
            auto=args.auto,
            num_threads=4,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
        )
        
        optimized_program = optimizer.compile(
            baseline_program,
            trainset=trainset,
            minibatch=True,
            minibatch_size=25,
            minibatch_full_eval_steps=10,
        )
        
        optimization_time = time.time() - start_time
        print(f"\nOptimization completed in {optimization_time:.1f}s")
        
        # Evaluate optimized program
        print("\nEvaluating optimized program...")
        eval_start = time.time()
        optimized_accuracy = evaluate_program(optimized_program, valset, banking77_metric)
        eval_time = time.time() - eval_start
        print(f"  Optimized accuracy: {optimized_accuracy:.1%}")
        print(f"  Evaluation time: {eval_time:.1f}s")
        
        improvement = optimized_accuracy - baseline_accuracy
        
        # Extract optimized prompt if available
        optimized_prompt = None
        try:
            if hasattr(optimized_program, 'predict') and hasattr(optimized_program.predict, 'signature'):
                sig = optimized_program.predict.signature
                if hasattr(sig, 'instructions'):
                    optimized_prompt = sig.instructions
        except Exception:
            pass
        
        results = {
            "method": "dspy_miprov2",
            "status": "succeeded",
            "elapsed_seconds": optimization_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "model": args.model,
                "auto": args.auto,
                "trials": args.trials,
                "train_size": args.train_size,
                "val_size": args.val_size,
                "max_bootstrapped_demos": args.max_bootstrapped_demos,
                "max_labeled_demos": args.max_labeled_demos,
            },
            "results": {
                "baseline_accuracy": baseline_accuracy,
                "optimized_accuracy": optimized_accuracy,
                "improvement": improvement,
                "optimized_prompt": optimized_prompt,
            },
        }
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"{'Metric':<25} {'Value':<15}")
        print("-"*40)
        print(f"{'Baseline Accuracy':<25} {baseline_accuracy:>14.1%}")
        print(f"{'Optimized Accuracy':<25} {optimized_accuracy:>14.1%}")
        print(f"{'Improvement':<25} {improvement:>+14.1%}")
        print(f"{'Optimization Time':<25} {optimization_time:>13.1f}s")
        print("="*60)
        
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        
        results = {
            "method": "dspy_miprov2",
            "status": "failed",
            "elapsed_seconds": time.time() - start_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "model": args.model,
                "auto": args.auto,
                "trials": args.trials,
                "train_size": args.train_size,
                "val_size": args.val_size,
            },
            "error": str(e),
            "results": {
                "baseline_accuracy": baseline_accuracy,
            },
        }
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"banking77_dspy_mipro_{timestamp}.json"
    output_path = results_dir / output_filename
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    print("\nDone!")
    return results


if __name__ == "__main__":
    main()
