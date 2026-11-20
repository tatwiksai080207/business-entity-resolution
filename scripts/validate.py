"""
Validation script to check submission file formats and integrity rules.
"""

import os
import sys
import subprocess

def main():
    matching_file = sys.argv[1] if len(sys.argv) > 1 else "submissions/sub5/matching_results.tsv"
    candidate_file = sys.argv[2] if len(sys.argv) > 2 else "submissions/sub5/candidate_pairs.tsv"
    test_dir = sys.argv[3] if len(sys.argv) > 3 else "student_resource/dataset/test"

    print("=" * 80)
    print("Validating submission files...")
    print(f"  Matching file:  {matching_file}")
    print(f"  Candidate file: {candidate_file}")
    print(f"  Test directory: {test_dir}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "student_resource/utils/validate_submission.py",
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", test_dir,
        "--check-ids",
    ]

    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
