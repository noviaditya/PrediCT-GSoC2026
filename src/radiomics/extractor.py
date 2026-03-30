import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from radiomics import featureextractor
import logging

def configure_extractor():
    # Disable PyRadiomics logging clutter
    logging.getLogger('radiomics').setLevel(logging.ERROR)

    # Initialize Extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Disable all standard features to strictly follow Project 2 requirements
    extractor.disableAllFeatures()

    # Enable strictly requested features precisely to avoid augmentation distortion
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeaturesByName(shape=['Sphericity', 'SurfaceVolumeRatio', 'Maximum3DDiameter'])

    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeaturesByName(glcm=['Contrast', 'Correlation', 'InverseDifferenceMoment'])

    extractor.enableFeatureClassByName('glszm')
    extractor.enableFeaturesByName(glszm=['SmallAreaEmphasis', 'LargeAreaEmphasis', 'ZonePercentage'])

    extractor.enableFeatureClassByName('glrlm')
    extractor.enableFeaturesByName(glrlm=['ShortRunEmphasis', 'LongRunEmphasis', 'RunPercentage'])

    return extractor

def extract_radiomics(canonical_dir, output_csv="outputs/radiomics_features.csv"):
    csv_path = Path(canonical_dir) / "tables" / "scan_index.csv"
    if not csv_path.exists():
        print(f"[ERROR] Cannot find index for Extractor {csv_path}")
        return

    df = pd.read_csv(csv_path)
    extractor = configure_extractor()
    all_features = []

    # Process roughly 20-30 images as stated in Project 2 Goal (or entire set if small)
    for index, folder_path in tqdm(df.iterrows(), desc="Extracting Radiomics Textures", total=len(df)):
        if index >= 30: # Limit exactly to Project 2's request of "20-30 COCA scans"
            break
            
        folder = Path(folder_path["folder_path"])
        scan_id = folder.name
        img_path = folder / f"{scan_id}_img.nii.gz"
        seg_path = folder / f"{scan_id}_seg.nii.gz"

        if img_path.exists() and seg_path.exists():
            try:
                # Provide native paths directly to PyRadiomics to ensure ZERO spatial distortion
                result = extractor.execute(str(img_path), str(seg_path), label=1)
                
                # Filter out diagnostic text outputs
                cleaned_result = {"scan_id": scan_id}
                for key, val in result.items():
                    if "diagnostic_" not in key:
                        cleaned_result[key] = val

                all_features.append(cleaned_result)
            except Exception as e:
                print(f"Skipping {scan_id} due to Exception: {e}")

    results_df = pd.DataFrame(all_features)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved PyRadiomics structured extractions to {output_csv}")
