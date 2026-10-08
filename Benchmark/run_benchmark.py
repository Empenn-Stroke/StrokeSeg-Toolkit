"""
Module: run_benchmark.py
Description: Orchestrates batch inference runs of the StrokeSeg2 application 
             on NIfTI files across different model variants and precisions.
"""

import os
import time
import shutil
import subprocess
from pathlib import Path

def clear_directory(target_path: Path) -> None:
    """
    Safely clears all contents of a given directory.
    """
    if target_path.exists():
        for item in target_path.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            except Exception as e:
                print(f"Warning: Could not delete {item}. ({e})")
    else:
        target_path.mkdir(parents=True, exist_ok=True)
    return None

def clean_preproc_files(root_path: Path) -> int:
    """
    Removes intermediate PREPROC.nii.gz files recursively from the input directory.
    """
    removed_count = 0
    for f in root_path.rglob("*"):
        try:
            if f.is_file() and f.name.endswith('PREPROC.nii.gz'):
                f.unlink()
                removed_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {f}. ({e})")
    return removed_count

def process_nifti_files(root_folder: str, model_name: str, exe_path: str, output_dir: str, app_log_dir: str, final_log_destination: str) -> None:
    """
    Runs batch inference over all NIfTI files in the root folder using the specified model.
    """
    output_path = Path(output_dir)
    log_path = Path(app_log_dir)
    root_path = Path(root_folder)
    
    print(f"Clearing output folder: {output_path}...")
    clear_directory(output_path)
    
    print(f"Clearing old logs in: {log_path}...")
    clear_directory(log_path)
    
    if root_path.exists() and root_path.is_dir():
        print(f"Removing preprocessed files in '{root_path}'...")
        clean_preproc_files(root_path)

        nifti_files = [f for f in root_path.rglob("*") if f.name.endswith(('.nii', '.nii.gz'))]

        if nifti_files:
            print(f"Found {len(nifti_files)} file(s). Starting processing for model: {model_name}...\n")
            print("-" * 50)
            
            time.sleep(5) # Wait for hardware monitoring to stabilize
            
            for nifti_file in nifti_files:
                file_path_str = str(nifti_file.resolve())
                print(f"Processing: {file_path_str}")
                
                command = [
                    exe_path,
                    "--input", file_path_str,
                    "-o", output_dir,
                    "--model", model_name,
                    "--verbose",
                    "--skip-preproc"
                ]
                try:
                    subprocess.run(command, check=True)
                    print(f"SUCCESS: Finished processing {nifti_file.name}\n")
                except subprocess.CalledProcessError as e:
                    print(f"FAILED: An error occurred while processing {nifti_file.name}. ({e})\n")
                except FileNotFoundError:
                    print(f"CRITICAL ERROR: Could not find the executable at {exe_path}")
                    break
                print("-" * 50)
                
            try:
                shutil.copytree(log_path, final_log_destination, dirs_exist_ok=True)
                print(f"SUCCESS: Logs backed up to {final_log_destination}")
            except Exception as e:
                print(f"FAILED to backup logs: {e}")
                
            clean_preproc_files(root_path)
        else:
            print("No NIfTI files found in the specified directory.")
    else:
        print(f"Error: The directory '{root_folder}' does not exist.")
        
    return None

if __name__ == "__main__":
    EXE_PATH = r"C:\Users\z0051vdu\source\repos\strokeseg2-app-build\Release\strokeseg2-app.exe"
    OUTPUT_DIR = r"C:\d\out"
    APP_LOG_DIR = r"C:\Users\z0051vdu\AppData\Roaming\Empenn - INRIA\StrokeSeg2"
    FOLDER_TO_SCAN = r"C:\d\Test_Set_ATLAS_2.1\images"
    
    for model in ["Teacher_fp32", "Nano_fp32", "Teacher_fp16", "Nano_fp16"]:
        final_dest = f"C:\\d\\{model}_logs"
        process_nifti_files(
            root_folder=FOLDER_TO_SCAN,
            model_name=model,
            exe_path=EXE_PATH,
            output_dir=OUTPUT_DIR,
            app_log_dir=APP_LOG_DIR,
            final_log_destination=final_dest
        )