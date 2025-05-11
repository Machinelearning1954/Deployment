import os
from datasets import load_dataset, load_from_disk # Added load_from_disk
from transformers import DistilBertTokenizerFast

# Define constants
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
DATA_DIR = "/home/ubuntu/data"
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Define new paths for Hugging Face datasets format
TRAIN_DATA_DIR_HF = os.path.join(PROCESSED_DATA_DIR, "train_dataset_hf")
TEST_DATA_DIR_HF = os.path.join(PROCESSED_DATA_DIR, "test_dataset_hf")

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# No need to create TRAIN_DATA_DIR_HF and TEST_DATA_DIR_HF here, save_to_disk will create them.

print(f"Starting data preprocessing using {MODEL_NAME}...")

# Load tokenizer
print("Loading tokenizer...")
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    exit()

# Load dataset
print("Loading IMDB dataset...")
try:
    imdb_dataset = load_dataset("imdb", cache_dir=os.path.join(DATA_DIR, "raw"))
    print("IMDB dataset loaded successfully.")
    print(f"Dataset structure: {imdb_dataset}")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

print("Preprocessing training data...")
try:
    train_dataset = imdb_dataset["train"].map(preprocess_function, batched=True)
    # Keep set_format for PyTorch compatibility if needed by Trainer, but saving will be in Arrow format by default.
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print("Training data preprocessed and formatted.")
except Exception as e:
    print(f"Error preprocessing training data: {e}")
    exit()

print("Preprocessing testing data...")
try:
    test_dataset = imdb_dataset["test"].map(preprocess_function, batched=True)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print("Testing data preprocessed and formatted.")
except Exception as e:
    print(f"Error preprocessing testing data: {e}")
    exit()

# Save processed datasets using Hugging Face datasets.save_to_disk
print(f"Saving processed training data to {TRAIN_DATA_DIR_HF}...")
try:
    train_dataset.save_to_disk(TRAIN_DATA_DIR_HF)
    print("Processed training data saved.")
except Exception as e:
    print(f"Error saving processed training data: {e}")
    exit()

print(f"Saving processed testing data to {TEST_DATA_DIR_HF}...")
try:
    test_dataset.save_to_disk(TEST_DATA_DIR_HF)
    print("Processed testing data saved.")
except Exception as e:
    print(f"Error saving processed testing data: {e}")
    exit()

print("Data preprocessing complete.")
print(f"Processed train dataset saved at: {TRAIN_DATA_DIR_HF}")
print(f"Processed test dataset saved at: {TEST_DATA_DIR_HF}")

# Verify saved directories
if os.path.exists(TRAIN_DATA_DIR_HF) and os.path.isdir(TRAIN_DATA_DIR_HF) and os.path.exists(TEST_DATA_DIR_HF) and os.path.isdir(TEST_DATA_DIR_HF):
    print("Verification successful: Processed data directories exist.")
else:
    print("Verification failed: One or both processed data directories are missing or not directories.")

