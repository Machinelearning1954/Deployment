import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
import time

# --- Configuration --- 
# User needs to replace these placeholders with their actual values
IAM_ROLE_ARN = "arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/YOUR_SAGEMAKER_EXECUTION_ROLE_NAME"  # e.g., "arn:aws:iam::123456789012:role/SageMakerExecutionRole"
S3_BUCKET_NAME = "your-sagemaker-bucket-name"  # The S3 bucket you created and uploaded data to
S3_BASE_PREFIX = "imdb_sentiment_analysis" # Base prefix in S3 for this project

# Define S3 paths for data and output
S3_INPUT_TRAIN = f"s3://{S3_BUCKET_NAME}/{S3_BASE_PREFIX}/data/train/"
S3_INPUT_VALIDATION = f"s3://{S3_BUCKET_NAME}/{S3_BASE_PREFIX}/data/validation/"
S3_OUTPUT_PATH = f"s3://{S3_BUCKET_NAME}/{S3_BASE_PREFIX}/output/"

# Training script and source directory
SOURCE_DIR = "s3_source_code_upload" # A local directory that will be uploaded to S3
ENTRY_POINT_SCRIPT = "train.py"
REQUIREMENTS_FILE = "requirements.txt"

# Create a directory to package source code for S3 upload
# In a real scenario, the user would run this script from a directory containing
# train.py and requirements.txt, or specify their paths.
# For this script, we assume train.py and requirements.txt are in the same directory as this script
# or the user will place them in the SOURCE_DIR.

if not os.path.exists(SOURCE_DIR):
    os.makedirs(SOURCE_DIR)

# --- Ensure train.py and requirements.txt are in the SOURCE_DIR ---
# The user should place train.py and requirements.txt into the SOURCE_DIR directory
# before running this script, or modify the paths below.
# For example:
# import shutil
# shutil.copy("train.py", os.path.join(SOURCE_DIR, "train.py"))
# shutil.copy("requirements.txt", os.path.join(SOURCE_DIR, "requirements.txt"))

print(f"Please ensure your '{ENTRY_POINT_SCRIPT}' and '{REQUIREMENTS_FILE}' are placed in the '{SOURCE_DIR}' directory.")
print(f"This script will then package '{SOURCE_DIR}' and upload it to S3 for SageMaker training.")

# --- Hyperparameters for the training job ---
hyperparameters = {
    "epochs": 1,                 # Number of training epochs
    "train_batch_size": 16,      # Batch size for training
    "eval_batch_size": 32,       # Batch size for evaluation
    "learning_rate": 5e-5,       # Learning rate
    "model_name": "distilbert-base-uncased", # Model to use
    "warmup_steps": 100,
    "weight_decay": 0.01
}

# --- SageMaker Session and Estimator Configuration ---
def main():
    print("Starting SageMaker training job configuration...")
    
    if IAM_ROLE_ARN == "arn:aws:iam::YOUR_AWS_ACCOUNT_ID:role/YOUR_SAGEMAKER_EXECUTION_ROLE_NAME" or \
       S3_BUCKET_NAME == "your-sagemaker-bucket-name":
        print("ERROR: Please update IAM_ROLE_ARN and S3_BUCKET_NAME placeholders in this script before running.")
        return

    # Check if source files exist
    if not os.path.exists(os.path.join(SOURCE_DIR, ENTRY_POINT_SCRIPT)):
        print(f"ERROR: {ENTRY_POINT_SCRIPT} not found in {SOURCE_DIR}. Please add it.")
        return
    if not os.path.exists(os.path.join(SOURCE_DIR, REQUIREMENTS_FILE)):
        print(f"ERROR: {REQUIREMENTS_FILE} not found in {SOURCE_DIR}. Please add it.")
        return

    try:
        sagemaker_session = sagemaker.Session()
        region = sagemaker_session.boto_region_name
        print(f"SageMaker session initialized in region: {region}")
    except Exception as e:
        print(f"Error initializing SageMaker session: {e}")
        print("Please ensure your AWS credentials and SageMaker permissions are configured correctly.")
        return

    # Define the HuggingFace Estimator
    huggingface_estimator = HuggingFace(
        entry_point=ENTRY_POINT_SCRIPT,
        source_dir=SOURCE_DIR, # Path to the directory with the training script and requirements.txt
        instance_type="ml.m5.large",  # Choose an appropriate instance type. ml.g4dn.xlarge for GPU.
        instance_count=1,
        role=IAM_ROLE_ARN,
        transformers_version="4.26", # Specify a compatible transformers version
        pytorch_version="1.13",      # Specify a compatible PyTorch version
        py_version="py39",           # Specify a compatible Python version
        hyperparameters=hyperparameters,
        sagemaker_session=sagemaker_session,
        output_path=S3_OUTPUT_PATH, # Where to save model artifacts and other outputs
        # requirements_file=REQUIREMENTS_FILE, # Handled by source_dir if requirements.txt is inside
        disable_profiler=True # Disable profiler for this example
    )

    # Define data channels
    inputs = {
        "train": sagemaker.inputs.TrainingInput(
            s3_data=S3_INPUT_TRAIN,
            distribution="FullyReplicated",
            content_type="text/csv",
            s3_data_type="S3Prefix"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=S3_INPUT_VALIDATION,
            distribution="FullyReplicated",
            content_type="text/csv",
            s3_data_type="S3Prefix"
        )
    }

    # Generate a unique job name
    job_name = f"imdb-hf-distilbert-{time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())}"
    print(f"Generated SageMaker Training Job Name: {job_name}")

    # Launch the training job
    print("Launching SageMaker training job...")
    try:
        huggingface_estimator.fit(inputs, job_name=job_name, wait=True) # Set wait=False to run asynchronously
        print(f"Training job {job_name} completed.")
        print(f"Model artifacts are stored at: {huggingface_estimator.model_data}")
        print(f"You can find the training job in the AWS SageMaker console in region {region}.")
    except Exception as e:
        print(f"Error launching or during SageMaker training job: {e}")

if __name__ == "__main__":
    print("--- SageMaker Training Job Launcher ---")
    print("This script helps you launch a Hugging Face training job on Amazon SageMaker.")
    print("Before running:")
    print("1. Ensure you have AWS credentials configured (e.g., via AWS CLI `aws configure`).")
    print("2. Ensure the SageMaker Python SDK and Boto3 are installed (`pip install sagemaker boto3`).")
    print(f"3. Replace the placeholder values for `IAM_ROLE_ARN` and `S3_BUCKET_NAME` at the top of this script.")
    print(f"4. Create a directory named '{SOURCE_DIR}' in the same location as this script.")
    print(f"5. Place your '{ENTRY_POINT_SCRIPT}' (the SageMaker training script) and '{REQUIREMENTS_FILE}' into the '{SOURCE_DIR}' directory.")
    print(f"6. Ensure your preprocessed data (train.csv, validation.csv) has been uploaded to the S3 paths: ")
    print(f"   - Training data: {S3_INPUT_TRAIN}")
    print(f"   - Validation data: {S3_INPUT_VALIDATION}")
    print("-----------------------------------------")
    
    # A simple check to see if the user wants to proceed after reading instructions
    # In a real environment, you might not need this interactive prompt.
    # proceed = input("Have you completed the setup steps above and wish to proceed? (yes/no): ")
    # if proceed.lower() == "yes":
    #     main()
    # else:
    #     print("Exiting. Please complete the setup steps.")
    print("\nTo run this script, ensure all setup steps are complete and then execute it.")
    print("You might want to uncomment the 'main()' call or the interactive prompt section to run it directly.")
    print("For now, the main() function will not be called automatically to prevent accidental job launches.")
    print("Call main() directly or uncomment the prompt to run.")
    # main() # Uncomment this to run the script directly after setup

