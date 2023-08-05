import pickle
import numpy as np
import axelrod as axl
from typing import List
from axelrod.load_data_ import load_weights
from axelrod.strategies.ann import ANN, compute_features
from axelrod.action import Action
from axelrod.player import Player


C, D = Action.C, Action.D
relu = np.vectorize(lambda x: max(x, 0))
nn_weights = load_weights()


def activate(bias, hidden, output, features):
    """Compute the output of a single layer neural network
    and return the hidden activations."""
    hidden_values = np.dot(hidden, features) + bias
    hidden_activations = relu(hidden_values)
    output = np.dot(hidden_activations, output)
    return hidden_activations, output


def aggregate_features_wa(hidden_activations, input_features):
    ha = np.array(hidden_activations) + np.ones_like(
        hidden_activations
    )  # (num_rounds x num_neurons)
    fi = np.array(input_features) + np.ones_like(
        input_features
    )  # (num_rounds x num_features)
    wa = np.dot(ha.T, fi)
    return wa


def aggregate_features_wa_normed(hidden_activations, input_features):
    """For each neuron, this function aggregates the input features
    by performing a weighted average based on the neuron's activations. This
    results in an `activation profile` of the neuron.

    Returns a (num_neurons x len(input_features)) matrix. Force entries between
    0-10 and then round to integers.
    """
    ha = np.array(hidden_activations) + np.ones_like(hidden_activations)
    wa = aggregate_features_wa(hidden_activations, input_features)
    normed = wa / ha.sum(axis=0).reshape(-1, 1)
    return np.round(10 * normed / normed.max(axis=1, keepdims=True))


def aggregate_features_max(hidden_activations, input_features):
    """For each neuron, this function aggregrates the input features
    by returning the input features that correspond to the maximum activation
    of the neuron.

    Returns a (num_neurons x len(input_features)) matrix.
    """
    max_inputs = []
    for i in range(len(hidden_activations[0])):  # loop through neurons
        max_inputs.append(
            input_features[1:][np.argmax(np.array(hidden_activations)[:, i])]
        )
    return np.array(max_inputs)


def corr_map(hidden_activations, input_features, eps=1e-3):
    """This function finds the covariance between the neuron activites of
    different neurons."""
    fi = np.array(input_features)
    weighted_inputs = []  # (num_neurons x num_rounds x num_features)
    for i in range(len(hidden_activations[0])):
        weighted_inputs.append(np.array(input_features) * fi[:, i].reshape(-1, 1))
    wi = np.array(weighted_inputs) + eps
    return np.round(np.corrcoef(wi.reshape(wi.shape[0], -1)), 2)


class ANNRecorder(ANN):
    """A modification to the ANN class that stores all of the inputs and the
    resulting hidden activations when the `strategy` method is called."""

    def __init__(self, num_features: int, num_hidden: int, weights: List[float] = None):
        super().__init__(num_features, num_hidden, weights)
        self.inputs = []
        self.activations = []

    def strategy(self, opponent: Player) -> Action:
        """This method is called by the player to determine the next action."""
        features = compute_features(self, opponent)
        hidden_activations, output = activate(
            self.bias_weights,
            self.input_to_hidden_layer_weights,
            self.hidden_to_output_layer_weights,
            features,
        )
        self.inputs.append(features)
        self.activations.append(hidden_activations)
        return C if output > 0 else D

    def reset(self):
        self.inputs = []
        self.activations = []


class EvolvedANNRecorder(ANNRecorder):
    def __init__(self, weight_file_name):
        num_features, num_hidden, weights = nn_weights[weight_file_name]
        super().__init__(
            num_features=num_features, num_hidden=num_hidden, weights=weights
        )


if __name__ == "__main__":
    players = (axl.Alternator(), EvolvedANNRecorder("Evolved ANN 5"))
    match = axl.Match(players, turns=100)
    match.play()
    print(match.sparklines())
    print(match.winner())
    wa = aggregate_features_wa_normed(players[1].activations, players[1].inputs)
    print(aggregate_features_max(players[1].activations, players[1].inputs))
    print(corr_map(players[1].activations, players[1].inputs))
    pickle.dump(wa, open("wa.pkl", "wb+"))
