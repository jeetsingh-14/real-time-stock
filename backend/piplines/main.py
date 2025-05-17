import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_name):
    logger.info(f"Starting {script_name}")
    try:
        # Get the absolute path to the script
        if script_name == "../backend/predict_real_time.py":
            # predict_real_time.py is in the backend directory
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", script_name)
        else:
            # Other scripts are in the root directory
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)

        # Check if the script exists
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return False

        logger.info(f"Running script: {script_path}")

        # Run the script as a subprocess
        process = subprocess.Popen(
            [sys.executable, script_path],
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
    # First try the backend version
    predict_result = run_script("predict_real_time.py")

    # If backend version fails, try the root directory version
    if not predict_result:
        logger.warning("Backend predict_real_time.py failed, trying root directory version")
        # Create a direct path to the root directory version
        root_predict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predict_real_time.py")

        if os.path.exists(root_predict_path):
            logger.info(f"Found predict_real_time.py in root directory: {root_predict_path}")
            # Run the script directly with its full path
            # process = subprocess.Popen(
            #     [sys.executable, root_predict_path],
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.PIPE,
            #     text=True
            # )

            process = subprocess.Popen(
                ['python', root_predict_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'  # <-- force utf-8 output
            )

            # Capture output
            stdout, stderr = process.communicate()

            # Log output
            if stdout:
                logger.info(f"Output from root predict_real_time.py:\n{stdout}")

            # Check for errors
            if process.returncode != 0:
                logger.error(f"Error running root predict_real_time.py:\n{stderr}")
                logger.warning("Both prediction scripts failed")
            elif stderr:
                # Log stderr even if return code is 0 (warnings, etc.)
                logger.warning(f"Warnings from root predict_real_time.py:\n{stderr}")
                predict_result = True
                logger.info("Successfully completed root predict_real_time.py")
            else:
                predict_result = True
                logger.info("Successfully completed root predict_real_time.py")
        else:
            logger.error("No predict_real_time.py found in root directory")
            logger.warning("Prediction step failed")

    # Log completion status
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
