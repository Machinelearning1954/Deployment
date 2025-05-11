import os
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import re

# Define constants
DATA_DIR = "/home/ubuntu/sagemaker_data"
S3_BUCKET_PLACEHOLDER = "your-sagemaker-bucket-name"  # User needs to replace this
S3_PREFIX_PLACEHOLDER = "imdb_sentiment_analysis/data" # User can adjust this

# Create local directory to store processed data
os.makedirs(DATA_DIR, exist_ok=True)

print("Starting data preparation for SageMaker...")

# 1. Data Acquisition
print("Loading IMDB dataset...")
try:
    imdb_dataset = load_dataset("imdb")
    print("IMDB dataset loaded successfully.")
except Exception as e:
    print(f"Error loading IMDB dataset: {e}")
    exit()

# Function to clean text
def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters and digits (optional, depends on model choice)
    # text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    return text

# 2. Initial Preprocessing and Cleaning
print("Preprocessing dataset...")

# Process train and test splits
processed_splits = {}
for split_name in ["train", "test"]:
    texts = [clean_text(example["text"]) for example in imdb_dataset[split_name]]
    labels = [example["label"] for example in imdb_dataset[split_name]]
    processed_splits[split_name] = pd.DataFrame({"text": texts, "label": labels})
    print(f"Processed {split_name} split: {len(processed_splits[split_name])} samples")

# 3. Data Splitting (further split original train into new train and validation)
print("Splitting training data into new train and validation sets...")
original_train_df = processed_splits["train"]
train_df, val_df = train_test_split(original_train_df, test_size=0.2, random_state=42, stratify=original_train_df["label"])
test_df = processed_splits["test"]

print(f"New training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# 4. Save to CSV for SageMaker
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
VALIDATION_CSV_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")

print(f"Saving processed data to CSV files in {DATA_DIR}...")
# SageMaker typically expects no header and the target variable in the first column for some built-in algorithms.
# For custom scripts (like Hugging Face), having headers is fine and often preferred.
# We will save with headers for clarity, as our training script will handle it.
train_df.to_csv(TRAIN_CSV_PATH, index=False, header=True)
val_df.to_csv(VALIDATION_CSV_PATH, index=False, header=True)
test_df.to_csv(TEST_CSV_PATH, index=False, header=True)
print(f"Train data saved to {TRAIN_CSV_PATH}")
print(f"Validation data saved to {VALIDATION_CSV_PATH}")
print(f"Test data saved to {TEST_CSV_PATH}")

# 5. Instructions for Uploading to S3
print("\n--- Instructions for Uploading to S3 ---")
print(f"Please upload the generated CSV files to your S3 bucket.")
print(f"Recommended S3 paths:")
print(f"  Training data: s3://{S3_BUCKET_PLACEHOLDER}/{S3_PREFIX_PLACEHOLDER}/train/train.csv")
print(f"  Validation data: s3://{S3_BUCKET_PLACEHOLDER}/{S3_PREFIX_PLACEHOLDER}/validation/validation.csv")
print(f"  Test data: s3://{S3_BUCKET_PLACEHOLDER}/{S3_PREFIX_PLACEHOLDER}/test/test.csv")
print("\nYou can use the AWS Management Console, AWS CLI, or Boto3 for uploading.")
print("Example AWS CLI commands:")
print(f"  aws s3 cp {TRAIN_CSV_PATH} s3://{S3_BUCKET_PLACEHOLDER}/{S3_PREFIX_PLACEHOLDER}/train/train.csv")
print(f"  aws s3 cp {VALIDATION_CSV_PATH} s3://{S3_BUCKET_PLACEHOLDER}/{S3_PREFIX_PLACEHOLDER}/validation/validation.csv")
print(f"  aws s3 cp {TEST_CSV_PATH} s3://{S3_BUCKET_PLACEHOLDER}/{S3_PREFIX_PLACEHOLDER}/test/test.csv")

print("\nExample Boto3 Python snippet (ensure AWS credentials and Boto3 are configured):")
print("""
import boto3
s3_client = boto3.client('s3')

def upload_to_s3(file_path, bucket, object_name):
    try:
        s3_client.upload_file(file_path, bucket, object_name)
        print(f"Successfully uploaded {file_path} to s3://{bucket}/{object_name}")
    except Exception as e:
        print(f"Error uploading {file_path}: {e}")

# Replace with your actual bucket and desired S3 paths
# upload_to_s3(TRAIN_CSV_PATH, S3_BUCKET_PLACEHOLDER, f"{S3_PREFIX_PLACEHOLDER}/train/train.csv")
# upload_to_s3(VALIDATION_CSV_PATH, S3_BUCKET_PLACEHOLDER, f"{S3_PREFIX_PLACEHOLDER}/validation/validation.csv")
# upload_to_s3(TEST_CSV_PATH, S3_BUCKET_PLACEHOLDER, f"{S3_PREFIX_PLACEHOLDER}/test/test.csv")
""")

print("\nData preparation script finished.")

