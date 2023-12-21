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

def phi_generate(model,tokenizer,promts):
    output=[]
    for prom in promts:
        inputs=tokenizer(prom,return_tensors="pt",return_attention_mask=False)
        generation = model.generate(**inputs,max_length=200)
        continuation = tokenizer.batch_decode(generation)[0]
        output.append(continuation)
    display(f"{len(output)} numbers of continuation is generated with {multiprocessing.current_process().name}")
    return output