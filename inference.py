import os
import json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# SageMaker will look for these functions

def model_fn(model_dir):
    """
    Load the model and tokenizer from the `model_dir`.
    SageMaker will call this function to load the model.
    """
    print(f"Loading model from: {model_dir}")
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval() # Set model to evaluation mode
        print("Model and tokenizer loaded successfully.")
        return {"model": model, "tokenizer": tokenizer}
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def input_fn(request_body, request_content_type):
    """
    Deserialize the InvokeEndpoint request body into an object we can perform prediction on.
    """
    print(f"Received request_content_type: {request_content_type}")
    print(f"Received request_body: {request_body}")
    if request_content_type == "application/json":
        try:
            data = json.loads(request_body)
            if isinstance(data, str):
                # Single string input
                return [data]
            elif isinstance(data, dict) and "inputs" in data:
                # JSON object with "inputs" key, which can be a string or list of strings
                inputs_data = data["inputs"]
                if isinstance(inputs_data, str):
                    return [inputs_data]
                elif isinstance(inputs_data, list):
                    return inputs_data
                else:
                    raise ValueError("Invalid format for \"inputs\" field. Expected string or list of strings.")
            elif isinstance(data, list):
                # List of strings
                return data
            else:
                raise ValueError("Invalid JSON input format. Expected a JSON string, a JSON list of strings, or a JSON object with an \"inputs\" key.")
        except Exception as e:
            print(f"Error in input_fn processing JSON: {e}")
            raise ValueError(f"Could not parse JSON input: {request_body}. Error: {e}")
    elif request_content_type == "text/plain":
        return [request_body.strip()] # Treat as a single text input
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Please use application/json or text/plain.")

def predict_fn(input_data, model_artifacts):
    """
    Perform prediction on the deserialized object.
    """
    print(f"Performing prediction on input_data: {input_data}")
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions = []
    try:
        for text_input in input_data:
            inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class_id = torch.argmax(probabilities, dim=-1).item()
                predictions.append({
                    "label": "positive" if predicted_class_id == 1 else "negative",
                    "score": probabilities[0][predicted_class_id].item(),
                    "probabilities": {
                        "negative": probabilities[0][0].item(),
                        "positive": probabilities[0][1].item()
                    }
                })
        print(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise

def output_fn(prediction_output, accept_content_type):
    """
    Serialize the prediction output to the desired response content type.
    """
    print(f"Received accept_content_type: {accept_content_type}")
    print(f"Serializing prediction_output: {prediction_output}")
    if accept_content_type == "application/json":
        try:
            return json.dumps(prediction_output), "application/json"
        except Exception as e:
            print(f"Error in output_fn serializing to JSON: {e}")
            raise ValueError(f"Could not serialize prediction to JSON. Error: {e}")
    else:
        raise ValueError(f"Unsupported accept type: {accept_content_type}. Please use application/json.")

# Example for local testing (not used by SageMaker directly but helpful for development)
if __name__ == "__main__":
    # This block is for local testing only.
    # SageMaker will save the model to /opt/ml/model in the container.
    # For local testing, you would need a model saved in a similar structure.
    # Create dummy model_dir for local testing if needed
    model_dir_local = "./model_artifacts_test"
    if not os.path.exists(model_dir_local):
        print(f"Local model directory {model_dir_local} not found. Cannot run local test.")
        # As a very basic placeholder for local testing structure:
        # You would need to save a real model and tokenizer here.
        # For example, after running train.py locally and saving the model.
        # DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased").save_pretrained(model_dir_local)
        # DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).save_pretrained(model_dir_local)
        exit()

    print("--- Local Inference Test --- ")
    model_artifacts_loaded = model_fn(model_dir_local)
    
    # Test case 1: Single string in JSON
    test_input_json_single_str = json.dumps("This is a wonderful movie!")
    deserialized_input_1 = input_fn(test_input_json_single_str, "application/json")
    prediction_1 = predict_fn(deserialized_input_1, model_artifacts_loaded)
    serialized_output_1, _ = output_fn(prediction_1, "application/json")
    print(f"Test 1 Input (JSON single string): {test_input_json_single_str}")
    print(f"Test 1 Output: {serialized_output_1}")

    # Test case 2: List of strings in JSON
    test_input_json_list = json.dumps(["This is a wonderful movie!", "This is a terrible film."])
    deserialized_input_2 = input_fn(test_input_json_list, "application/json")
    prediction_2 = predict_fn(deserialized_input_2, model_artifacts_loaded)
    serialized_output_2, _ = output_fn(prediction_2, "application/json")
    print(f"Test 2 Input (JSON list): {test_input_json_list}")
    print(f"Test 2 Output: {serialized_output_2}")

    # Test case 3: JSON object with "inputs" key (single string)
    test_input_json_object_single = json.dumps({"inputs": "Another great experience."}) 
    deserialized_input_3 = input_fn(test_input_json_object_single, "application/json")
    prediction_3 = predict_fn(deserialized_input_3, model_artifacts_loaded)
    serialized_output_3, _ = output_fn(prediction_3, "application/json")
    print(f"Test 3 Input (JSON object single): {test_input_json_object_single}")
    print(f"Test 3 Output: {serialized_output_3}")

    # Test case 4: JSON object with "inputs" key (list of strings)
    test_input_json_object_list = json.dumps({"inputs": ["Loved it!", "Hated it.", "It was okay."]})
    deserialized_input_4 = input_fn(test_input_json_object_list, "application/json")
    prediction_4 = predict_fn(deserialized_input_4, model_artifacts_loaded)
    serialized_output_4, _ = output_fn(prediction_4, "application/json")
    print(f"Test 4 Input (JSON object list): {test_input_json_object_list}")
    print(f"Test 4 Output: {serialized_output_4}")

    # Test case 5: Plain text input
    test_input_plain_text = "This is a plain text review. It was fantastic."
    deserialized_input_5 = input_fn(test_input_plain_text, "text/plain")
    prediction_5 = predict_fn(deserialized_input_5, model_artifacts_loaded)
    serialized_output_5, _ = output_fn(prediction_5, "application/json") # Assuming JSON output is always desired
    print(f"Test 5 Input (Plain text): {test_input_plain_text}")
    print(f"Test 5 Output: {serialized_output_5}")

    print("--- Local Inference Test Complete ---")

