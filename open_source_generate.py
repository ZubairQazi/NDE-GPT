import argparse
import csv
import json
import logging
import os
import random
import string

import pandas as pd
import torch
from bs4 import BeautifulSoup
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from utils.test_functions import get_model_outputs

logging.basicConfig(level=logging.INFO)
# Define arguments
parser = argparse.ArgumentParser(
    description="Gather output data from an open source model"
)
parser.add_argument(
    "-s",
    "--sample",
    action="store_true",
    help="Process a single text sample",
)
parser.add_argument(
    "-d",
    "--dataset",
    type=str,
    help="Process a dataset, must provide a path",
)
parser.add_argument(
    "-m",
    "--model-path",
    type=str,
    help="Path to a stored model directory (expected to contain 'model/' and 'tokenizer/' folders). Leave blank and use --hf-path for non-local huggingface models",
)
parser.add_argument(
    "--storage-path",
    type=str,
    default="/nvme/models",
    help="Path to the model storage directory",
)
parser.add_argument(
    "-o",
    "--output-path",
    type=str,
    help="Path for output file. Leave blank for default output file path (default: outputs/<random_ascii>.csv)",
)
parser.add_argument(
    "--hf-path",
    type=str,
    default="mistralai/Mixtral-8x7B-v0.1",
    help="Path for huggingface model (default: mistralai/Mixtral-8x7B-v0.1)",
)
args = parser.parse_args()

if not args.sample and not args.dataset:
    parser.error("Please specify either --sample or --dataset flag")
if args.sample and args.dataset:
    parser.error("Please specify only one of --sample or --dataset flag")
if args.dataset and not args.dataset:
    parser.error("Please provide a path to the dataset using the --dataset option")

with open("templates/open_source_template.txt", "r") as template_file:
    template = template_file.read()

with open("EDAM/edam_topics.txt", "r") as edam_file:
    full_edam_topics = edam_file.readlines()
full_edam_topics = [topic.strip() for topic in full_edam_topics]

formatted_topics = "\n".join(full_edam_topics)
template = template.replace("<topics>", formatted_topics)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model_path:
    logging.info(f"\nLoading tokenizer and model from disk: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(f"{args.model_path}/tokenizer")
    model = AutoModelForCausalLM.from_pretrained(
        f"{args.model_path}/model",
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model_name = os.path.basename(args.model_path)
else:
    # mistralai/Mixtral-8x7B-Instruct-v0.1
    logging.info("\nGrabbing huggingface token from config")
    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        hf_token = config["api_keys"]["huggingface"]

    logging.info(f"Loading tokenizer and model from hf")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mixtral-8x7B-v0.1", token=hf_token
    )
    # mistralai/Mixtral-8x7B-Instruct-v0.1
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_path,
        token=hf_token,
        device_map="auto",
        load_in_4bit=True,
    )

    model_name = os.path.basename(args.hf_path)

    logging.info(f"Saving tokenizer and model to disk: {args.model_path}/{model_name}/")
    tokenizer.save_pretrained(
        f"{args.storage_path}/{model_name}/tokenizer", from_pt=True
    )
    model.save_pretrained(f"{args.storage_path}/{model_name}/model", from_pt=True)


if args.sample:
    # Code for inputting one text sample
    abstract = input("Input an Abstract: ")
    prompt = template.replace("<abstract>", abstract)

    prompt = prompt.replace("<num_terms>", "3")

    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2500,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    parsed_output = output.split("\n")[-1]
    print("Generated Output:")
    print(parsed_output)

elif args.dataset:
    # Code for inputting a dataset
    dataset = pd.read_csv(args.dataset)

    text_column = input(
        f"\nEnter the column corresponding to text data ({','.join(dataset.columns)}): "
    )
    truth_column = input(
        f"Enter the column corresponding to ground truth, or empty for None ({','.join(dataset.columns)}): "
    )
    identifier_column = input(
        f"Enter the column corresponding to an identifier ({','.join(dataset.columns)}): "
    )

    dataset[text_column] = dataset[text_column].apply(
        lambda text: BeautifulSoup(text, "html.parser").get_text()
    )
    if truth_column:
        dataset[truth_column] = dataset[truth_column].apply(
            lambda text: BeautifulSoup(text, "html.parser").get_text()
        ).apply(lambda labels: set(map(str.strip, next(csv.reader([labels])))))

    logging.info("\nGenerating model outputs")
    predictions = get_model_outputs(
        data=dataset,
        template=template,
        tokenizer=tokenizer,
        model=model,
        device=device,
        text_column=text_column,
        truth_column=truth_column,
        num_outputs=3,
    )

    if not args.output_path:
        random_name = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=8)
        )
        output_path = f"outputs/{random_name}.csv"
        logging.warning(f"\nUsing '{output_path}'. This may overwrite previous results.")
    else:
        output_path = args.output_path
    logging.info(f"Saving model outputs to {output_path}")
    output_df = pd.DataFrame(
        zip(
            dataset[identifier_column],
            dataset[text_column],
            [model_name] * len(dataset),
            predictions,
        ),
        columns=[identifier_column, text_column, "Model", "Predictions"],
    )
    if truth_column:
        output_df["Ground Truth"] = dataset[truth_column]

    output_df["Filtered Predictions"] = (
        output_df["Predictions"]
        .apply(lambda preds: set(map(str.strip, next(csv.reader([preds])))))
        .apply(
            lambda preds: ", ".join(
                [pred for pred in preds if pred in full_edam_topics]
            )
        )
    )
    output_df.to_csv(output_path, index=False)

else:
    print("Invalid choice. Please choose either 1 or 2.")
