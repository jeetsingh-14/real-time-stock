import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Configure logging to both console and file
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "pipeline.log")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_name):
    logger.info(f"Starting {script_name}")
    try:
        # Map script names to their locations in the new structure
        script_mapping = {
            "merge_sentiment_with_price.py": os.path.join("piplines", "merge_sentiment_with_price.py"),
            "train_models.py": os.path.join("training", "train_models.py"),
            "predict_real_time.py": os.path.join("piplines", "predict_real_time.py"),
            "save_to_bigquery.py": os.path.join("piplines", "save_to_bigquery.py")
        }

        # Get the mapped path or use the original if not in mapping
        mapped_script = script_mapping.get(script_name, script_name)

        # Get the absolute path to the script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), mapped_script)

        # Check if the script exists
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return False

        logger.info(f"Running script: {script_path}")

        # Run the script as a module instead of a script
        module_path = f"stockmarket_analysis.backend.{mapped_script.replace('.py', '').replace(os.path.sep, '.')}"
        logger.info(f"Running as module: {module_path}")

        # Run the script as a subprocess
        process = subprocess.Popen(
            [sys.executable, "-m", module_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Capture output
        stdout, stderr = process.communicate()

        # Log output
        if stdout:
            logger.info(f"Output from {script_name}:\n{stdout}")

        # Check for errors
        if process.returncode != 0:
            logger.error(f"Error running {script_name}:\n{stderr}")
            return False
        elif stderr:
            # Log stderr even if return code is 0 (warnings, etc.)
            logger.warning(f"Warnings from {script_name}:\n{stderr}")

        logger.info(f"Successfully completed {script_name}")
        return True
    except Exception as e:
        logger.error(f"Exception running {script_name}: {str(e)}")
        return False

def main():
    logger.info("Starting pipeline execution")

    # Step 1: Run merge_sentiment_with_price.py
    logger.info("Step 1: Merging sentiment with price data")
    merge_result = run_script("merge_sentiment_with_price.py")
    if not merge_result:
        logger.warning("Merge step failed, but continuing with pipeline")

    # Step 2: Run train_models.py
    logger.info("Step 2: Training models")
    train_result = run_script("train_models.py")
    if not train_result:
        logger.warning("Training step failed, but continuing with pipeline")

    # Step 3: Run predict_real_time.py
    logger.info("Step 3: Making real-time predictions")
    predict_result = run_script("predict_real_time.py")
    if not predict_result:
        logger.warning("Prediction step failed, but continuing with pipeline")

    # Step 4: Upload tables to BigQuery
    logger.info("Step 4: Uploading tables to BigQuery")
    bigquery_result = run_script("save_to_bigquery.py")
    if not bigquery_result:
        logger.warning("BigQuery upload failed")

    if merge_result and train_result and predict_result and bigquery_result:
        logger.info("Pipeline completed successfully")
    else:
        logger.warning("Pipeline completed with some errors")

    logger.info("Pipeline execution finished")


if __name__ == "__main__":
    main()
