import numpy as np

class Deck:
    num_of_samples = 10000000  # Reduced number of samples

    def __init__(self, num_of_cards, deck_string):
        self.num_of_cards = int(num_of_cards)
        deck = np.array([int(card) for card in deck_string.split(' ')], dtype=np.int32)  # Use smaller int type
        self.unique_cards, self.probas = np.unique(deck, return_counts=True)
        self.probas = self.probas.astype(np.float32) / self.num_of_cards  # Use float32 for probabilities

    # def expectation(self):
    #     samples_matrix = np.random.choice(self.unique_cards, size=Deck.num_of_samples, p=self.probas)
    #     samples_matrix = samples_matrix.astype(np.int32).reshape(4, -1)  # Use int32 for samples
    #     min_indices = np.argmin(samples_matrix, axis=0)
    #     col_indices = np.arange(samples_matrix.shape[1], dtype=np.int32)  # Use int32 for indices
    #     samples_matrix[min_indices, col_indices] = 0
    #     column_sums = samples_matrix.sum(axis=0).astype(np.float32)  # Sum as float32 to save memory
    #     mean_sum = column_sums.mean()
    #     return mean_sum

    def expectation_of_minimum(self):
        # Number of draws
        n = 4

        # Calculate the expectation of the minimum value
        expectation_min = 0
        for i, xi in enumerate(self.unique_cards):
            # Probability that all n draws are >= xi
            prob_all_ge_xi = np.sum(self.probas[i:]) ** n

            # Probability that all n draws are > xi
            prob_all_gt_xi = np.sum(self.probas[i+1:]) ** n if i+1 < len(self.unique_cards) else 0

            # Probability that xi is the minimum
            prob_xi_min = prob_all_ge_xi - prob_all_gt_xi

            # Add to the expectation
            expectation_min += xi * prob_xi_min

        return expectation_min

    def expectation(self):
        # Calculate the mean of the distribution
        mean_distribution = np.sum(self.unique_cards * self.probas)

        # Generate samples: 4 samples per set, many sets
        # num_sets = Deck.num_of_samples // 4
        # samples_sets = np.random.choice(self.unique_cards, size=num_sets * 4, p=self.probas)
        # samples_sets = samples_sets.reshape(num_sets, 4)

        # Calculate the mean of the minimum of each set
        # mean_min = np.mean(np.min(samples_sets, axis=1))

        # Calculate the adjusted mean
        adjusted_mean = (4 * mean_distribution - self.expectation_of_minimum())

        return adjusted_mean

    # def expectation(self):
    #     # Calculate the expected contribution of each card
    #     contributions = np.zeros_like(self.unique_cards, dtype=np.float32)
    #     for i, card in enumerate(self.unique_cards):
    #         # Probability of being picked
    #         p_pick = self.probas[i]

    #         # Probability of this card being among the three largest in a set of four
    #         # This is 1 minus the probability of being the smallest in a set of four
    #         # which is the probability it is picked all four times
    #         p_top3 = 1 - (p_pick ** 4)

    #         # Expected contribution is the card value times the probability of being in the top 3
    #         contributions[i] = card * p_top3

    #     # Sum the contributions of all cards
    #     expected_sum = np.sum(contributions)

    #     return expected_sum


    
num_of_decks = int(input())
decks = []
for _ in range(num_of_decks):
    num_of_cards = input()
    cards = input()
    decks.append(Deck(num_of_cards, cards))

expectations = np.array([deck.expectation() for deck in decks])

print(np.argmax(expectations) + 1, f"{round(float(np.max(expectations)), 3):.7f}")

