from tqdm import tqdm


def get_model_outputs(
    data,
    template,
    tokenizer,
    model,
    device,
    text_column="Abstract",
    truth_column="Ground Truth",
    num_outputs=10,
    max_new_tokens=2500,
):
    outputs = []
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        abstract = row[text_column]

        prompt = template.replace("<abstract>", abstract)
        if truth_column:
            ground_truth = row[truth_column]
            prompt = prompt.replace("<num_terms>", str(len(ground_truth)))
        else:
            prompt = prompt.replace("<num_terms>", str(num_outputs))

        model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed_output = output.split("\n")[-1]

        outputs.append(parsed_output)

    return outputs
