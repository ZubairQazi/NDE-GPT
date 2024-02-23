import ast
import asyncio
import csv
import json
import logging
import pickle
import random
import string
import sys
from typing import Any

import openai
import pandas as pd
from bs4 import BeautifulSoup
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

with open("config.json", "r") as config_file:
    config = json.load(config_file)

openai_api_key = config["api_keys"]["openai"]

output_path = input("Enter file path for model output (empty for default): ")
if output_path == "":
    random_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    output_path = f"outputs/{random_name}.csv"
    logging.warning(f"Using '{output_path}'. This may overwrite previous results.")

dataset = pd.read_csv(input("Enter dataset path (CSV): "), lineterminator="\n")


# Originally sourced from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a
async def dispatch_openai_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
    max_tokens=256,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        max_tokens: Maximum number of tokens to generate.

    Returns:
        List of responses from OpenAI API.
    """
    client = AsyncOpenAI(api_key=openai_api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    async def get_response(message):
        response = await client.chat.completions.create(
            model=model,
            messages=message,
            max_tokens=max_tokens,
            timeout=120,
        )
        return response.choices[0].message.content

    response_aggregate, responses = [], []
    for i, message in enumerate(messages_list):
        response = await get_response(message)
        responses.append(response)
        response_aggregate.append(response)

        # Save every 10 responses
        if (i + 1) % 10 == 0:
            logging.info(f"Saving responses {i-9} to {i}...")
            with open("outputs/responses.txt", "a") as f:
                for resp in responses:
                    f.write(resp + "\n")
            responses = []  # Clear the list of responses

    # Save any remaining responses
    if responses:
        with open("outputs/responses.txt", "a") as f:
            for response in responses:
                f.write(response + "\n")

    return response_aggregate


try:
    ## Build prompts and feed to GPT
    text_column = input(
        f"Enter the column corresponding to text data ({','.join(dataset.columns)}): "
    )
    identifier_column = input(
        f"Enter the column corresponding to an identifier/title ({','.join(dataset.columns)}): "
    )
    if text_column not in dataset.columns:
        logging.error(f"Text column {text_column} not found in dataset.")
        sys.exit()

    logging.info("Loading and constructing templates...")
    with open("templates/prompt_template.txt", "r") as template_file:
        template = template_file.read()
    with open("EDAM/edam_topics.txt", "r") as edam_file:
        full_edam_topics = edam_file.readlines()
    full_edam_topics = [topic.strip() for topic in full_edam_topics]
    formatted_topics = "\n".join(full_edam_topics)
    template = template.replace("<topics>", formatted_topics)

    logging.info("Building prompts from dataset...")
    prompts = []
    for idx, row in dataset.iterrows():
        text = row[text_column]

        prompt = template.replace("<title>", row[identifier_column])
        prompt = prompt.replace("<abstract>", text)
        prompt = prompt.replace("<num_terms>", "3")

        prompts.append([{"role": "user", "content": prompt}])

    logging.info("Querying GPT...")
    predictions = asyncio.run(
        dispatch_openai_requests(
            messages_list=prompts,
            model="gpt-3.5-turbo-0125",
        )
    )

    # Strip extra starting/ending quotes from predictions output
    predictions = [
        response.strip('"') if response.count('"') == 2 else response
        for response in predictions
    ]

    # Save results to a CSV file
    output_df = pd.DataFrame(
        zip(
            dataset[identifier_column],
            dataset[text_column],
            ["gpt-3.5-turbo"] * len(dataset),
            predictions,
        ),
        columns=[identifier_column, text_column, "Model", "Predictions"],
    )

    output_df.to_csv(output_path, lineterminator="\n", index=False)

except SystemExit:
    print("Saving results and gracefully exiting.")
    # TODO: Save results
    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)
