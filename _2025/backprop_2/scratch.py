import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device='cuda'

model_id = "meta-llama/Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(inputs['input_ids'])
my_probs=torch.nn.functional.softmax(outputs.logits, dim=-1)

for i in torch.argsort(my_probs[0,-1, :].detach().cpu(), descending=True)[:5]:
    print(i, round(my_probs[0, -1, i].item(),5), tokenizer.decode([i]))