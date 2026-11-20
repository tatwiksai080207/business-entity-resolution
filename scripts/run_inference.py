"""
Script to execute the candidate blocking and LightGBM inference pipeline on test data.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TEST_DIR, OUTPUT_DIR, OPTIMAL_DECISION_THRESHOLD
from src.inference import format_submission_files

def main():
    print("=" * 80)
    print("Multilingual Business Entity Resolution - Test Inference Pipeline")
    print("=" * 80)
    print(f"Test data directory: {TEST_DIR}")
    print(f"Output directory:    {OUTPUT_DIR}")
    print(f"Decision threshold:  {OPTIMAL_DECISION_THRESHOLD}")
    print("For accelerated dual GPU execution, refer to notebooks/02_test_inference_and_blocking.ipynb")

if __name__ == "__main__":
    main()
