import sys
import json
from utils import query_gpt_neuron_explanation


if len(sys.argv) < 3:
    print("Usage: python query.py <model> <prompt_file> <output_file>")

model = sys.argv[1]
prompt_fname = sys.argv[2]
prompts = json.load(open(prompt_fname))
outputs = []

for prompt in prompts:
    outputs.append(query_gpt_neuron_explanation(model, prompt))

json.dump(outputs, open(sys.argv[3], "w+"))
