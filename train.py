import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser()

    # --- SageMaker HPO and Training Job Parameters ---
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")

    # --- SageMaker Data Input Channels --- 
    # SM_CHANNEL_TRAIN, SM_CHANNEL_VALIDATION, SM_CHANNEL_TEST are environment variables set by SageMaker
    # The paths they point to are local paths inside the SageMaker training container, 
    # where SageMaker has downloaded the data from S3.
    parser.add_argument("--train_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    # parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TEST")) # Test data usually handled separately after training

    # --- SageMaker Model Output Channel ---
    # SM_MODEL_DIR is an environment variable set by SageMaker, pointing to the S3 location where the model should be saved.
    # Inside the container, it points to /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")) # For other artifacts

    args, _ = parser.parse_known_args()

    print(f"Received arguments: {args}")

    # Load data from CSV files in the SageMaker input channels
    train_file_path = os.path.join(args.train_dir, "train.csv")
    validation_file_path = os.path.join(args.validation_dir, "validation.csv")

    print(f"Loading training data from: {train_file_path}")
    train_df = pd.read_csv(train_file_path)
    print(f"Loading validation data from: {validation_file_path}")
    val_df = pd.read_csv(validation_file_path)

    # Convert pandas DataFrames to Hugging Face Datasets
    train_dataset_hf = Dataset.from_pandas(train_df)
    val_dataset_hf = Dataset.from_pandas(val_df)

    print(f"Training dataset size: {len(train_dataset_hf)}")
    print(f"Validation dataset size: {len(val_dataset_hf)}")

    # Load tokenizer
    print(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)

    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    print("Tokenizing datasets...")
    train_dataset_tokenized = train_dataset_hf.map(tokenize_function, batched=True)
    val_dataset_tokenized = val_dataset_hf.map(tokenize_function, batched=True)

    # Set format for PyTorch
    train_dataset_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    print("Tokenization and formatting complete.")

    # Load model
    print(f"Loading model: {args.model_name} for sequence classification")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    # Define metrics computation
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
    # Output directory for SageMaker is /opt/ml/output/data (SM_OUTPUT_DATA_DIR)
    # Model artifacts are saved to /opt/ml/model (SM_MODEL_DIR)
    training_output_dir = os.path.join(args.output_data_dir, "training_output")
    print(f"Defining training arguments. Output dir: {training_output_dir}")
    training_args = TrainingArguments(
        output_dir=training_output_dir, # Checkpoints and logs
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        logging_dir=f"{training_output_dir}/logs",
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard" # SageMaker will capture these logs
    )

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokenized,
        eval_dataset=val_dataset_tokenized,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer # Pass tokenizer for saving with model
    )

    # Train the model
    print("Starting model training...")
    trainer.train()
    print("Model training complete.")

    # Evaluate the model (on validation set as per HuggingFace Trainer standard)
    print("Evaluating the model on the validation set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model and tokenizer to SM_MODEL_DIR for SageMaker to pick up
    print(f"Saving model to {args.model_dir}...")
    trainer.save_model(args.model_dir) # This saves model and tokenizer
    # tokenizer.save_pretrained(args.model_dir) # Trainer.save_model should handle this if tokenizer is passed
    print(f"Model saved to {args.model_dir}")

    print("SageMaker training script finished.")

if __name__ == "__main__":
    main()

