from datasets import load_dataset
from transformers import AutoTokenizer
import json, argparse, os

def download_and_save_dataset():
    """
    Downloads a dataset from Hugging Face, selects specified columns, and saves it as a JSON file.
    """
    # # dataset_name = "nyu-mll/glue"
    # dataset_name = "yahma/alpaca-cleaned"
    # data_dir = None
    # # data_dir = "mrpc"
    # data_split = "train"
    # selected_columns = ["instruction"]
    # # selected_columns = None

    output_dir = "./tmp_datasets"
    os.makedirs(output_dir, exist_ok=True)

    # # Load dataset from Hugging Face
    # dataset_info = f"{dataset_name}/{data_dir}/{data_split}" if data_dir else f"{dataset_name}/{data_split}"
    # json_filename = f"{dataset_info.replace('/', '_')}.json"

    # print(f"Loading dataset: {dataset_info}, split: {data_split}")
    # dataset = load_dataset(dataset_name, data_dir=data_dir, split=data_split)
    
    # # Get available columns
    # available_columns = dataset.column_names
    # print(f"Available columns: {available_columns}")

    # # Select specified columns if they exist
    # if selected_columns:
    #     assert all(col in available_columns for col in selected_columns), "Invalid column selected"
    #     dataset = dataset.remove_columns([col for col in available_columns if col not in selected_columns])

    # # Save dataset as JSON
    # json_path = os.path.join(output_dir, json_filename)
    # dataset.to_json(json_path, orient="records")

    # print(f"Dataset saved as JSON at: {json_path}")

    # model_name = "meta-llama/Llama-3.1-70B-Instruct"
    model_name = "DreamGallery/task-14-meta-llama-Meta-Llama-3.1-8B-Instruct"
    max_length = 10000
    output_file = os.path.join(output_dir, "s1K_tokenized.json")

    # Load a sample dataset; adjust the dataset name and split as needed.
    dataset = load_dataset("simplescaling/s1K_tokenized", split="train")  # using a subset for speed

    # Load a pre-trained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to tokenize text and add a token count.
    def tokenize_count(example):
        # Tokenize the text field
        tokens = tokenizer.tokenize(example["text"])
        # Save the number of tokens to a new field
        example["token_count"] = len(tokens)
        return example

    # Apply the function to each example in the dataset.
    tokenized_dataset = dataset.map(tokenize_count)

    # Filter entries with token_count less than max_length.
    filtered_dataset = tokenized_dataset.filter(lambda example: example["token_count"] < max_length)

    # Extract the original text field from the filtered examples.
    text_list = filtered_dataset["text"]

    # Save the text list to a JSON file.
    with open(output_file, "w") as f:
        json.dump(text_list, f, indent=2)

def main():
    download_and_save_dataset()

if __name__ == "__main__":
    main()
