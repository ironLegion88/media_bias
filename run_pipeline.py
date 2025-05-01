# run_pipeline.py
import subprocess
import os
import logging
import sys
import time

# --- Configuration ---
try:
    # Since run_pipeline.py is at the root, it can import config directly
    import config
except ImportError:
    print("Error: config.py not found in the project root where run_pipeline.py is located.")
    exit(1)

# Create logs directory if it doesn't exist
os.makedirs(config.LOG_DIR, exist_ok=True)

# --- Setup Logging for the pipeline itself ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = logging.FileHandler(config.PIPELINE_LOG_FILE)
log_handler.setFormatter(log_formatter)

pipeline_logger = logging.getLogger("PipelineRunner")
pipeline_logger.setLevel(logging.INFO)
# Prevent adding handler multiple times if script is re-run in same session
if not pipeline_logger.hasHandlers():
    pipeline_logger.addHandler(log_handler)
    # Add console handler for immediate feedback
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    pipeline_logger.addHandler(console_handler)


# --- Define Scripts to Run (use module paths) ---
# NOTE: Ensure filenames in src/ do NOT contain hyphens
SCRIPTS_TO_RUN_AS_MODULES = [
    #'src.article_scraper',      # Corresponds to src/article_scraper.py
    #'src.preprocessing',        # Corresponds to src/preprocessing.py
    'src.analysis_vader',       # Corresponds to src/analysis_vader.py
    'src.analysis_textblob',    # Corresponds to src/analysis_textblob.py
]

# --- Function to Execute a Script as a Module ---
def run_module(module_name):
    """Executes a Python module using 'python -m' and logs the outcome."""
    pipeline_logger.info(f"--- Executing module: {module_name} ---")
    start_time = time.time()
    try:
        # Ensure python executable used is the one running this script
        python_executable = sys.executable
        # Run using the '-m' flag to execute as a module
        result = subprocess.run(
            [python_executable, '-m', module_name],
            check=True, # Raises CalledProcessError if script returns non-zero exit code
            capture_output=True,
            text=True,
            encoding='utf-8',
            # Run from the project root directory (where config.py is)
            # This is usually the default behavior when running run_pipeline.py from root
            cwd=config.BASE_DIR
        )
        end_time = time.time()
        duration = end_time - start_time
        pipeline_logger.info(f"Successfully executed {module_name} in {duration:.2f} seconds.")
        # Log stdout/stderr from the script (optional, can be verbose)
        if result.stdout:
             pipeline_logger.debug(f"Output from {module_name}:\n{result.stdout[-500:]}") # Log last 500 chars
        if result.stderr:
             # Stderr isn't always an error (e.g., warnings), log as warning
             pipeline_logger.warning(f"Stderr from {module_name}:\n{result.stderr}")
        return True
    except FileNotFoundError:
        # This error now means python executable wasn't found, less likely
        pipeline_logger.error(f"Error: Python executable not found at {python_executable}")
        return False
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        pipeline_logger.error(f"Error executing {module_name} after {duration:.2f} seconds. Exit code: {e.returncode}")
        # Print the full error output which likely contains the traceback from the script
        pipeline_logger.error(f"Stdout: {e.stdout}")
        pipeline_logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        pipeline_logger.error(f"An unexpected error occurred while running {module_name} after {duration:.2f} seconds: {e}")
        return False

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    pipeline_logger.info("======== Starting Media Bias Analysis Pipeline ========")
    overall_start_time = time.time()
    all_successful = True

    # Check if src/__init__.py exists
    init_path = os.path.join(config.BASE_DIR, 'src', '__init__.py')
    if not os.path.exists(init_path):
        pipeline_logger.warning(f"src/__init__.py not found. Creating an empty one for package recognition.")
        try:
            with open(init_path, 'w') as f:
                pass # Create empty file
        except Exception as e:
            pipeline_logger.error(f"Could not create src/__init__.py: {e}. Module execution might fail.")
            # Optionally exit here if __init__.py is critical

    for module in SCRIPTS_TO_RUN_AS_MODULES:
        if not run_module(module):
            all_successful = False
            pipeline_logger.error(f"Pipeline halted due to error in {module}.")
            break # Stop the pipeline if a script fails
        pipeline_logger.info("-" * 50) # Separator

    overall_end_time = time.time()
    overall_duration = overall_end_time - overall_start_time

    if all_successful:
        pipeline_logger.info(f"======== Pipeline Completed Successfully in {overall_duration:.2f} seconds ========")
    else:
        pipeline_logger.error(f"======== Pipeline Failed after {overall_duration:.2f} seconds ========")

    pipeline_logger.info(f"Check individual logs in '{config.LOG_DIR}' and results in '{config.OUTPUT_DIR}'.")