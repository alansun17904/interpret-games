import os
import openai
import matplotlib.pyplot as plt


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")


def neuron_corr_map(corr_matrix, fname="neuron-corrmap.pdf"):
    """This function plots the correlation matrix of the given
    correlation map of neuron activations"""
    neurons = corr_matrix.shape[0]
    plt.figure(figsize=(10, 10))
    plt.matshow(corr_matrix)
    plt.title("Correlation Map of Neuron Activations")
    plt.xticks(range(neurons), range(1, neurons + 1))
    plt.yticks(range(neurons), range(1, neurons + 1))
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    plt.savefig(fname)


def query_gpt_neuron_explanation(model, prompt):
    """This function queries the GPT `model` given the `prompt`
    to interpret a singular neuron."""
    if openai.api_key is None or openai.organization is None:
        raise ValueError("OpenAI API key and organization must be set.")
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        top_p=1,
    )
    return response["choices"][0]["message"].content
