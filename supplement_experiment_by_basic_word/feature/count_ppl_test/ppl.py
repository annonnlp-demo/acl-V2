def mean(train_dataset, data_column_name_1):
    import torch

    from tqdm import tqdm

    from datasets import load_from_disk

    # change

    # change


    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    device = 'cuda'
    model_id = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    encodings = tokenizer('\n\n'.join(train_dataset[data_column_name_1]), return_tensors='pt')
    max_length = model.config.n_positions
    stride = 512

    lls = []
    end_loc = 0
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    print(float(ppl))

    return float(ppl)

