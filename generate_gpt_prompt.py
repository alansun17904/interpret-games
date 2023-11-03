import sys
import json
import pickle


feature_names = [
    "Opponent's first move is C",
    "Opponent's first move is D",
    "Opponent's second move is C",
    "Opponent's second move is D",
    "Player's previous move is C",
    "Player's previous move is D",
    "Player's second previous move is C",
    "Player's second previous move is D",
    "Opponent's previous move is C",
    "Opponent's previous move is D",
    "Opponent's second previous move is C",
    "Opponent's second previous move is D",
    "Total opponent cooperations",
    "Total opponent defections",
    "Total player cooperations",
    "Total player defections",
    "Round number",
]

feature_types = [
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
    "R",
    "R",
    "R",
    "R",
    "R",
]


if len(sys.argv) < 3:
    print(
        "Usage: python generate_gpt_prompt.py <prompt_header_file> <input_file> <output_file>"
    )
    sys.exit(1)

prompt_header_fname = sys.argv[1]
activations_fname = sys.argv[2]
output_fname = sys.argv[3]

prompt_header = open(prompt_header_fname).read()

prompts = []
activations = pickle.load(open(activations_fname, "rb")).T
print("Activations shape:", activations.shape)
for activation in activations:
    prompt = prompt_header + "\n"
    prompt += "Neuron 2\n<start>"
    for i, feature_name in enumerate(feature_names):
        if feature_types[i] == "B":
            fmt_activation = round(activation[i], 3)
        else:
            fmt_activation = int(activation[i])
        prompt += f"{feature_types[i]}\t{feature_name}\t{fmt_activation}\n"
    prompt += "<end>\nConclusion: "
    prompts.append(prompt)
json.dump(prompts, open(output_fname, "w+"))
