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

dataset = pd.read_csv(input("Enter dataset path (CSV): "))


# Originally sourced from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a
async def dispatch_openai_requests(
    messages_list: list[list[dict[str, Any]]],
    model: str,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.

    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    client = AsyncOpenAI(api_key=openai_api_key)
    responses = [
        await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=message,
            max_tokens=2200,
            timeout=120,
        )
        for message in messages_list
    ]

    return responses


try:
    ## Build prompts and feed to GPT
    text_column = input(
        f"Enter the column corresponding to text data ({','.join(dataset.columns)}): "
    )
    identifier_column = input(
        f"Enter the column corresponding to identifiers ({','.join(dataset.columns)}): "
    )
    if text_column not in dataset.columns:
        logging.error(f"Text column {text_column} not found in dataset.")
        sys.exit()

    dataset[text_column] = dataset[text_column].apply(
        lambda text: BeautifulSoup(text, "html.parser").get_text()
    )
    dataset[identifier_column] = dataset[identifier_column].apply(
        lambda text: BeautifulSoup(text, "html.parser").get_text()
    )

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

        prompt = template.replace("<abstract>", text)
        prompt = prompt.replace("<num_terms>", "3")

        prompts.append([{"role": "user", "content": prompt}])

    logging.info("Querying GPT...")
    predictions = asyncio.run(
        dispatch_openai_requests(
            messages_list=prompts,
            model="gpt-3.5-turbo",
        )
    )

    # for i, x in enumerate(predictions):
    #     print(f"Response {i+1}: {x.choices[0].message.content}")

    # Save results to a CSV file
    output_df = pd.DataFrame(
        zip(
            dataset[identifier_column],
            dataset[text_column],
            ["gpt-3.5-turbo"] * len(dataset),
            [x.choices[0].message.content for x in predictions],
        ),
        columns=[identifier_column, text_column, "Model", "Predictions"],
    )
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

except SystemExit:
    print("Saving results and gracefully exiting.")
    # TODO: Save results
    with open(output_path, "wb") as f:
        pickle.dump(predictions, f)
