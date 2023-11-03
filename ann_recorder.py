import re
import sys
import base64
import pickle
import numpy as np
import pandas as pd
import axelrod as axl
from typing import List
from utils import neuron_corr_map
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
    """For each neuron, this function aggregates the input features
    by performing a weighted average based on the neuron's activations. This
    results in an `activation profile` of the neuron.

    Returns a (num_neurons x len(input_features)) matrix. The ij-entry
    represents the expected value of the jth feature given that the ith neuron
    has fired.
    """
    ha = np.array(hidden_activations) + np.ones_like(
        hidden_activations
    )  # (num_rounds x num_neurons)
    ha = ha / ha.sum(axis=0, keepdims=True)
    fi = np.array(input_features)  # (num_rounds x num_features)
    wa = np.dot(fi.T, ha)
    return wa


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


def corr_map(hidden_activations, eps=1e-3):
    """This function finds the covariance between the neuron activites of
    different neurons."""
    return np.round(np.corrcoef(hidden_activations.T), 2)


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
    def __init__(self, weight_file_name, local=True):
        if not local:
            num_features, num_hidden, weights = nn_weights[weight_file_name]
        else:
            df = pd.read_csv(weight_file_name, header=None)
            best_player = df.iloc[df[1].idxmax()]
            print(f"avg score: {best_player[1]} | pstdev: {best_player[2]} | best score: {best_player[3]}") 
            # deserialize the weights of the play 
            player_data = pickle.loads(base64.b64decode(best_player[4]))
            num_features = player_data["num_features"]
            num_hidden = player_data["num_hidden"]
            weights = player_data["weights"]

            
        super().__init__(
            num_features=num_features, num_hidden=num_hidden, weights=weights
        )


def main():
    bs = [s() for s in axl.basic_strategies]
    if len(sys.argv) == 1:
        print("\n".join([b.name for b in bs]))
        sys.exit()
    if len(sys.argv) < 3:
        sys.exit(1)
    NETWORK_NAME = sys.argv[1]
    OPPONENT = sys.argv[2]
    # replace underscores with spaces in the opponent name
    search_name = OPPONENT.replace("_", " ")
    num_neurons = re.search('\d+', NETWORK_NAME).group()

    if OPPONENT != "all":
        # get the player 
        player = [b for b in bs if b.name == search_name]
        if len(player) == 0:
            print(f"no player with name '{OPPONENT}' found")
            sys.exit(1)
        print("Playing against", player[0].name)
        players = (player[0], EvolvedANNRecorder(NETWORK_NAME))
        match = axl.Match(players, turns=20)
        match.play()
        print(match.sparklines())
        print(match.winner())
    else:
        for b in bs:
            print("Playing against", b.name)
            players = (b, EvolvedANNRecorder(NETWORK_NAME))
            match = axl.Match(players, turns=20)
            match.play()
            print(match.sparklines())
            print(match.winner())
    wa = aggregate_features_wa(players[1].activations, players[1].inputs)
    corr = corr_map(wa)
    neuron_corr_map(
        corr,
        fname=f"visuals/neuron-corrmap-{num_neurons}-{OPPONENT}.pdf"
    )
    pickle.dump(wa, open(f"data/wa-{num_neurons}-{OPPONENT}.pkl", "wb+"))


if __name__ == "__main__":
    main()
