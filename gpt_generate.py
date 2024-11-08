import asyncio
import json
import logging
import os
import random
import string
import sys
import time
from typing import Any
from tqdm import tqdm
import csv

import openai
import pandas as pd
import tiktoken
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

with open("config.json", "r") as config_file:
    config = json.load(config_file)

openai_api_key = config["api_keys"]["openai"]
openai_org_id = config["project_org_ids"]["openai_org"]
openai_project_id = config["project_org_ids"]["openai_project"]

dataset_path = input("Enter dataset path (CSV): ")

dataset = pd.read_csv(dataset_path, lineterminator="\n")
logging.info(f"Loaded dataset with {len(dataset)} rows.")

output_path = input("Enter output file path with file extension (empty for default): ")
if output_path == "" or not os.path.isdir(os.path.dirname(output_path)):
    random_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    output_path = os.path.join("outputs", f"{random_name}.csv")
logging.warning(f"Using '{output_path}'. This may overwrite previous results.")

backup_path = input("Enter backup file path with file extension (empty for default): ")
if backup_path == "":
    base_name = os.path.basename(dataset_path)
    file_name_without_extension = os.path.splitext(base_name)[0]
    backup_path = os.path.join("outputs", "backups", f"{file_name_without_extension}_backup.txt")
logging.info(f"Using '{backup_path}' as backup file. Append mode will be used.")

# Get the number of lines in the backup file if it exists. If the file is empty set to default starting row (0)
output, starting_index = [], 0
if os.path.exists(backup_path):
    with open(backup_path, "r") as f:
        reader = csv.reader(f)
        # Skip the first row which contains the header
        next(reader, None)
        output = list(reader)
        num_responses = len(output)
        starting_index = max(0, num_responses)
        logging.info(
            f"Backup file '{backup_path}' already exists and contains {starting_index} responses. Starting from row index {starting_index}."
        )
else:
    backup_dir = os.path.dirname(backup_path)
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    # Create the CSV file with a header
    with open(backup_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if os.stat(backup_path).st_size == 0:  # Check if the file is empty
            writer.writerow(['id', 'response'])  # Write the header
    num_responses = 0

MAX_TOKENS_PER_MINUTE = 1000000
MAX_REQUESTS_PER_MINUTE = 3500
MAX_CONTEXT_LENGTH = 16385

# Originally sourced from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a
@retry(
    wait=wait_random_exponential(multiplier=2, min=3, max=60),
    stop=stop_after_attempt(3),
)
async def get_response(message, model, client, max_tokens=248):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            timeout=120,
        )
        return response.choices[0].message.content
    except openai.RateLimitError as e:
        print(f"Rate limit exceeded: {e} \n Retrying...")
        raise
    except openai.BadRequestError as e:
        error_code = e.response.json()['error']['code']
        print(f"Bad request --> Error code: {e.status_code} - {error_code}. Returning empty string...")
        return ""


async def dispatch_openai_requests(
    messages_list: list[list[dict[str, Any]]],
    encoding: tiktoken.Encoding,
    model: str = "gpt-3.5-turbo-0125",
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        List of responses from OpenAI API.
    """

    client = AsyncOpenAI(api_key=openai_api_key, organization=openai_org_id, project=openai_project_id)

    total_tokens = 0
    total_requests = 0
    start_time = time.time()

    response_aggregate, responses = [], []
    for i, (id, message) in enumerate(messages_list):

        tokens = encoding.encode(message[0]["content"])
        num_tokens = len(tokens)

        # Check if rate limit has been reached
        if (
            total_tokens + num_tokens > MAX_TOKENS_PER_MINUTE
            or total_requests >= MAX_REQUESTS_PER_MINUTE
        ):
            sleep_time = 60 - (time.time() - start_time)
            if sleep_time > 0:
                logging.info(
                    f"Reached token or rate limit. Sleeping for {sleep_time} seconds..."
                )
                time.sleep(sleep_time)
            total_tokens = 0
            total_requests = 0
            start_time = time.time()

        total_tokens += num_tokens
        total_requests += 1

        response = await get_response(message, model, client)
        if not response:
            logging.warning(f"Empty response received for prompt at index {i}.")
            response = "NO RESPONSE RECEIVED"
        responses.append([id, response.replace("\n", "\t").replace("\\n", "\t")])
        response_aggregate.append([id, response])

        # Save every 10 responses
        if (i + 1) % 10 == 0:
            logging.info(f"Saving responses for indices {i + num_responses-9} to {i + num_responses}...")
            with open(backup_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(responses)
            responses = []

    # Save any remaining responses
    if responses:
        logging.info(
                f"Saving remaining responses for indices {i + num_responses-len(responses)} to {i + num_responses}..."
            )
        with open(backup_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(responses)
        responses = []

    return response_aggregate


try:
    ## Build prompts and feed to GPT

    if "Name" in dataset.columns and "Description" in dataset.columns and '_id' in dataset.columns:
        text_column = "Description"
        title_column = "Name"
        identifier_column = "_id"

    else:
        text_column = input(
            f"Enter the column corresponding to text data ({', '.join(dataset.columns)}): "
        )
        title_column = input(
            f"Enter the column corresponding to a title ({', '.join(dataset.columns)}): "
        )
        identifier_column = input(
            f"Enter the column corresponding to an ID ({', '.join(dataset.columns)}): "
        )
        if text_column not in dataset.columns:
            logging.error(f"Text column {text_column} not found in dataset.")
            sys.exit()

    logging.info("Loading and constructing templates...")
    with open("templates/measurement_techniques.txt", "r") as template_file:
        template = template_file.read()
    with open("EDAM/edam_topics.txt", "r") as edam_file:
        full_edam_topics = edam_file.readlines()
    full_edam_topics = [topic.strip() for topic in full_edam_topics]
    formatted_topics = "\n".join(full_edam_topics)
    if "<topics>" in template:
        template = template.replace("<topics>", formatted_topics)

    dataset[text_column] = dataset[text_column].fillna("").astype(str)
    dataset[title_column] = dataset[title_column].fillna("").astype(str)
    dataset[identifier_column] = dataset[identifier_column].fillna("").astype(str)

    encoding_model = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    # Encode the template to get the length
    template_without_placeholders = template.replace("<title>", "").replace("<abstract>", "").replace("<num_terms>", "")
    template_tokens = encoding_model.encode(template_without_placeholders)
    template_length = len(template_tokens)

    logging.info("Building prompts from dataset...")
    prompts = []
    for idx, row in tqdm(dataset.iloc[starting_index:].iterrows(), total=dataset.shape[0] - starting_index):

        text = row[text_column] if row[text_column] != "" else "No Description"
        title = row[title_column] if row[title_column] != "" else "No Title"
        id = row[identifier_column] if row[identifier_column] != "" else "No ID"

        text_tokens = encoding_model.encode(text)
        title_tokens = encoding_model.encode(title)
        total_length = template_length + len(text_tokens) + len(title_tokens)

        while total_length > MAX_CONTEXT_LENGTH:
            # Calculate how many tokens we need to remove from the text
            excess_tokens = total_length - MAX_CONTEXT_LENGTH
            logging.warning(f"Prompt at index {idx} exceeds the maximum length of {MAX_CONTEXT_LENGTH} tokens. Truncating text by {excess_tokens} tokens to fit...")
            # Truncate the text to fit the context length
            text_tokens = text_tokens[:len(text_tokens) - excess_tokens]
            # Convert tokens back to string
            text = encoding_model.decode(text_tokens)
            # Reconstruct the prompt with the truncated text
            if not text:
                text = 'No Description'
            prompt = template.replace("<abstract>", text)
            prompt = prompt.replace("<title>", title)
            prompt = prompt.replace("<num_terms>", "3")
            # Recalculate the total length
            total_length = len(encoding_model.encode(prompt))

        # If the total length is within the limit, construct the prompt normally
        if total_length <= MAX_CONTEXT_LENGTH:
            prompt = template.replace("<abstract>", text)
            prompt = prompt.replace("<title>", title)
            prompt = prompt.replace("<num_terms>", "3")

        prompts.append((id, [{"role": "user", "content": prompt}]))

    logging.info("Querying GPT...")
    predictions = asyncio.run(
        dispatch_openai_requests(
            messages_list=prompts,
            encoding=encoding_model
        )
    )

    # Concatenate the backup responses with the new predictions
    predictions = output + predictions

    # Strip extra starting/ending quotes from predictions output
    predictions = [
        [id, response.strip('"') if response.count('"') == 2 else response]
        for id, response in predictions
    ]

    # Save results to a CSV file
    output_df = pd.DataFrame(
        zip(
            dataset[identifier_column],
            dataset[title_column],
            dataset[text_column],
            ["gpt-3.5-turbo"] * len(dataset),
            [response for _, response in predictions],
        ),
        columns=[identifier_column, title_column, text_column, "Model", "Predictions"],
    )

    logging.info(f"Saving {len(output_df)} results to '{output_path}'...")
    output_df.to_csv(output_path, lineterminator="\n", index=False)

except SystemExit:
    logging.info("Caught system exit. Gracefully exiting.")
    
except Exception as e:
    logging.error(f"Caught an exception: {e}.")
