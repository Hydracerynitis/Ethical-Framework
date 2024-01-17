import multiprocessing
from IPython.display import display

def generate_continuation(pipeline,promts):
    output=[]
    for prom in promts:
        generation = pipeline(prom, max_length=50, do_sample=False, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(prom,'')
        output.append(continuation)
    display(f"{len(output)} numbers of continuation is generated with {multiprocessing.current_process().name}")
    return output

def generate_continuation_honest(pipeline,promts,tokenizer):
    output=[]
    for prom in promts:
        generation = pipeline(prom, max_length=len(tokenizer(prom)['input_ids'])+10, num_return_sequences=20, pad_token_id=50256)
        continuation = generation[0]['generated_text'].replace(prom,'')
        output.append(continuation)
    display(f"{len(output)} numbers of continuation is generated with {multiprocessing.current_process().name}")
    return output


def mixtral_generate(model,tokenizer,prompts,honest):
    outputs=[]
    for p in prompts:
        inputs=tokenizer(p, return_tensors="pt")
        if(honest):
            output = model.generate(**inputs, max_length=len(tokenizer(p)['input_ids'])+10, num_return_sequences=20).to("cuda")
        else:
            output = model.generate(**inputs, max_new_tokens=40).to("cuda")
        generation=tokenizer.decode(output[0], skip_special_tokens=True).replace(p,"")
        outputs.append(generation)
    print(f"{len(outputs)} numbers of continuation is generated with {multiprocessing.current_process().name}")
    return outputs

def phi_generate(model,tokenizer,promts,honest):
    outputs=[]
    for p in promts:
        inputs=tokenizer(p, return_tensors="pt", return_attention_mask=False).to("cuda")
        if(honest):
            output = model.generate(**inputs, max_length=len(tokenizer(p)['input_ids'])+10, num_return_sequences=20).to("cuda")
        else:
            output = model.generate(**inputs, max_new_tokens=40).to("cuda")
        generation = tokenizer.batch_decode(output)[0].replace(p,'')
        outputs.append(generation)
    display(f"{len(outputs)} numbers of continuation is generated with {multiprocessing.current_process().name}")
    return outputs