import json
import sys
from tqdm import tqdm
import logging
import pandas as pd

import tiktoken
from openai import OpenAI

MAX_TOKENS_PER_MINUTE = 1000000
MAX_REQUESTS_PER_MINUTE = 3500
MAX_CONTEXT_LENGTH = 16385

model = "gpt-3.5-turbo"

logging.basicConfig(level=logging.INFO)

with open("config.json", "r") as config_file:
    config = json.load(config_file)

openai_api_key = config["api_keys"]["openai"]
openai_org_id = config["project_org_ids"]["openai_org"]
openai_project_id = config["project_org_ids"]["openai_project"]

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key, organization=openai_org_id, project=openai_project_id)

# Check if any batch jobs exist
existing_batches = client.batches.list()
if existing_batches:
    logging.info(f"Found {len(existing_batches.data)} existing batch jobs.")
    for batch in existing_batches.data:
        logging.info(f"Batch ID: {batch.id}, Status: {batch.status}")
        if batch.status == "failed":
            logging.error(f"Batch ID: {batch.id} failed, Message: {' '.join(error.message for error in batch.errors.data)}")        
            if batch.status == "running":
                logging.warning("There is a running batch job. Please wait for it to complete before starting a new one.")
                sys.exit()
else:
    logging.info("No existing batch jobs found.")

dataset_path = input("Enter dataset path (CSV): ")

dataset = pd.read_csv(dataset_path, lineterminator="\n")
logging.info(f"Loaded dataset with {len(dataset)} rows.")

rebuild_jsonl = input("Do you want to rebuild the JSONL file? (y/n): ")
if rebuild_jsonl.lower() == "y":
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
        requests = []
        for idx, row in tqdm(dataset.iterrows(), total=dataset.shape[0]):

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

            request = {
                "custom_id": f"{id}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 248
                }
            }

            requests.append(request)

        # Convert the list of dictionaries into a JSONL string
        jsonl_string = "\n".join(json.dumps(request) for request in requests)

        # Write the JSONL string to a file
        with open("datasets/batch_requests/batchinput.jsonl", "w") as file:
            file.write(jsonl_string)

    except SystemExit:
        logging.info("Caught system exit. Gracefully exiting.")
        
    except Exception as e:
        logging.error(f"Caught an exception: {e}.")

# # Upload the file to OpenAI
# batch_input_file = client.files.create(file=open("datasets/batch_requests/batchinput.jsonl", "rb"), purpose="batch")
# batch_input_file_id = batch_input_file.id

# # Create a batch with the uploaded file
# client.batches.create(
#     input_file_id=batch_input_file_id,
#     endpoint="/v1/chat/completions",
#     completion_window="24h",
#     metadata={"description": "GPT Topics Categorization Batch Job"}
# )

# Define a function to split a list into chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

logging.info("Chunking requests into multiple batches.")
chunk_size = 50000
for i, chunk in enumerate(chunks(requests, chunk_size)):
    jsonl_string = "\n".join(json.dumps(request) for request in chunk)
    with open(f"datasets/batch_requests/batchinput_{i}.jsonl", "w") as file:
        file.write(jsonl_string)

    # Upload the file to OpenAI
    batch_input_file = client.files.create(file=open(f"datasets/batch_requests/batchinput_{i}.jsonl", "rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id

    # Create a batch with the uploaded file
    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"GPT Topics Categorization Batch Job {i}"}
    )