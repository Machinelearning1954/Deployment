import os
import torch
from datasets import load_from_disk
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Define constants
MODEL_NAME = "distilbert-base-uncased"
PROCESSED_DATA_DIR = "/home/ubuntu/data/processed"
MODEL_OUTPUT_DIR = "/home/ubuntu/model_output"
TRAIN_DATA_PATH_HF = os.path.join(PROCESSED_DATA_DIR, "train_dataset_hf")
TEST_DATA_PATH_HF = os.path.join(PROCESSED_DATA_DIR, "test_dataset_hf")
NUM_LABELS = 2  # Positive and Negative

# Create model output directory if it doesn"t exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

print(f"Starting model training using {MODEL_NAME}...")

# Load processed datasets using load_from_disk
print(f"Loading processed training data from {TRAIN_DATA_PATH_HF}...")
try:
    train_dataset = load_from_disk(TRAIN_DATA_PATH_HF)
    print("Processed training data loaded successfully.")
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
except Exception as e:
    print(f"Error loading processed training data: {e}")
    exit()

print(f"Loading processed testing data from {TEST_DATA_PATH_HF}...")
try:
    test_dataset = load_from_disk(TEST_DATA_PATH_HF)
    print("Processed testing data loaded successfully.")
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
except Exception as e:
    print(f"Error loading processed testing data: {e}")
    exit()

# Load pre-trained model
print(f"Loading pre-trained model {MODEL_NAME} for sequence classification...")
try:
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define metrics computation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# Define training arguments
print("Defining training arguments...")
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir=os.path.join(MODEL_OUTPUT_DIR, "logs"),
    logging_strategy="epoch",
    eval_strategy="epoch", # Changed from evaluation_strategy to eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="tensorboard"
)
print("Training arguments defined.")

# Initialize Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
print("Trainer initialized.")

# Train the model
print("Starting model training...")
try:
    trainer.train()
    print("Model training complete.")
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# Evaluate the model
print("Evaluating the model...")
try:
    eval_results = trainer.evaluate()
    print("Model evaluation complete.")
    print(f"Evaluation results: {eval_results}")
except Exception as e:
    print(f"Error during model evaluation: {e}")
    exit()

# Save the fine-tuned model and tokenizer
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "fine_tuned_sentiment_model")
print(f"Saving fine-tuned model and tokenizer to {MODEL_SAVE_PATH}...")
try:
    trainer.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print(f"Tokenizer for \"{MODEL_NAME}\" can be reloaded using its name.")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

print("Model training and saving process finished.")
print(f"Find the trained model at: {MODEL_SAVE_PATH}")
print(f"Evaluation metrics: {eval_results}")

