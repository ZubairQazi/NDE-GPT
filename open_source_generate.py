import ast
import contextlib
import io
import json
import os

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaForCausalLM)

with open("config.json", "r") as config_file:
    config = json.load(config_file)
    hf_token = config['api_keys']['huggingface']

model_storage_path = '/nvme/models'


with open('templates/open_source_template.txt', 'r') as template_file:
    template = template_file.read()

with open('EDAM/edam_topics.txt', 'r') as edam_file:
    full_edam_topics = edam_file.readlines()
full_edam_topics = [topic.strip() for topic in full_edam_topics]

formatted_topics = "\n".join(full_edam_topics)
template = template.replace("<topics>", formatted_topics)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Add user inputted options for models
folder_path = f"{model_storage_path}/mixtral-8x7b-model"

if os.path.exists(folder_path):
    print("The model is already downloaded. Loading from", folder_path)
    tokenizer = AutoTokenizer.from_pretrained(f"{model_storage_path}/mixtral-8x7b-tokenizer")
    model = AutoModelForCausalLM.from_pretrained(f"{model_storage_path}/mixtral-8x7b-model", device_map="auto", load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
else:
    # mistralai/Mixtral-8x7B-Instruct-v0.1
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1", token=hf_token)
    tokenizer.save_pretrained(f"{model_storage_path}/mixtral-8x7b-tokenizer", from_pt=True)

    # mistralai/Mixtral-8x7B-Instruct-v0.1
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1", token=hf_token, device_map="auto", load_in_4bit=True)
    model.save_pretrained(f"{model_storage_path}/mixtral-8x7b-model", from_pt=True)


abstract = input('Input an Abstract: ')
prompt = template.replace('<abstract>', abstract)

num_terms = 10
prompt = prompt.replace('<num_terms>', str(num_terms))

model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
generated_ids = model.generate(**model_inputs, max_new_tokens=2500, do_sample=True, pad_token_id=tokenizer.eos_token_id)
output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

parsed_output = output.split('\n')[-1]
print('Generated Output:')
print(parsed_output)