import itertools


def generate_subsets(players):
    subsets = []
    for r in range(1, len(players) + 1):
        subsets.extend(itertools.combinations(players, r))
    return subsets


def calculate_shapley_value(players, coalition_value_func):
    n = len(players)
    shapley_values = {player: 0 for player in players}

    for player in players:
        for subset in generate_subsets(players):
            if player not in subset:
                coalition_size = len(subset)
                subset_with_player = subset + (player,)
                coalition_value = coalition_value_func(subset)
                coalition_value_with_player = coalition_value_func(subset_with_player)
                marginal_contribution = coalition_value_with_player - coalition_value
                shapley_values[player] += marginal_contribution / coalition_size

    return shapley_values


def coalition_value_func(subset):
    # Example coalition value function
    values = {'A': 10, 'B': 20, 'C': 30, 'D': 40}
    return sum([values[player] for player in subset])


players = ['A', 'B', 'C', 'D']
shapley_values = calculate_shapley_value(players, coalition_value_func)

# Print the Shapley values
for player, value in shapley_values.items():
    print(f"Shapley value for player {player}: {value}")
