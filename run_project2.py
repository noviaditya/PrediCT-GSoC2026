from pathlib import Path

from src.radiomics.agatston import compute_dataset_agatston
from src.radiomics.extractor import extract_radiomics

def main():
    project_root = Path(__file__).resolve().parent
    canonical_dir = project_root / "data_canonical"

    print("=" * 50)
    print("      PREDICT-COCA PROJECT 2 RUNTIME")
    print("=" * 50)

    print("\n[INFO] Project 2: Computing Agatston Risk Categories natively...")
    compute_dataset_agatston(canonical_dir)

    print("\n[INFO] Project 2: Executing PyRadiomics Texture Extractions natively...")
    extract_radiomics(canonical_dir)

    print("\n[DONE] Project 2 Pipeline Execution Ready!\n")

if __name__ == "__main__":
    main()
