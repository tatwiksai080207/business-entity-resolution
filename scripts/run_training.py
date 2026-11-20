"""
Script to execute the full model training and threshold calibration pipeline.
"""

import os
import sys

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import TRAIN_DIR, OUTPUT_DIR
from src.preprocessing import preprocess_dataframe
from src.model import train_lightgbm_model, sweep_f05_threshold

def main():
    print("=" * 80)
    print("Multilingual Business Entity Resolution - Model Training Pipeline")
    print("=" * 80)
    print(f"Training data path: {TRAIN_DIR}")
    print(f"Output destination: {OUTPUT_DIR}")
    print("For full execution on GPU instances, refer to notebooks/01_model_training_and_validation.ipynb")

if __name__ == "__main__":
    main()
