from transformers import AutoTokenizer, LlamaForCausalLM

PATH_TO_CONVERTED_WEIGHTS='/data/unisound/plms/uniift-13b-v0.0.1/hf'
PATH_TO_CONVERTED_TOKENIZER='/data/unisound/plms/uniift-13b-v0.0.1/hf'

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Human:{}\nAssistant:".format("Hey, are you consciours? Can you talk to me?")
inputs = tokenizer(prompt, return_tensors="pt")
gen_kwargs = {"max_length": 128, "num_beams": 1, "do_sample": False, "top_p": 0.95,
                        "temperature": 0.8, }


# Generate
generate_ids = model.generate(inputs.input_ids.cuda(), **gen_kwargs)
token = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(token)
# "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
