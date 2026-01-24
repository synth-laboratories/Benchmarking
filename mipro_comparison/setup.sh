#!/bin/bash
# Setup script for MIPRO comparison benchmark

set -e

echo "Setting up MIPRO comparison environment..."

# Check Python version
python3 --version

# Install dependencies
pip install -r requirements.txt

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY not set"
fi

if [ -z "$SYNTH_API_KEY" ]; then
    echo "Warning: SYNTH_API_KEY not set (required for Synth MIPRO)"
fi

# Create results directory
mkdir -p results

echo "Setup complete!"
echo ""
echo "To run the comparison:"
echo "  1. Set OPENAI_API_KEY for DSPy MIPROv2"
echo "  2. Set SYNTH_API_KEY for Synth MIPRO"
echo "  3. Run: python run_comparison.py --all"
