import numpy as np

class Deck:
    def __init__(self, num_of_cards, deck_string):
        self.num_of_cards = int(num_of_cards)
        deck = np.array([int(card) for card in deck_string.split(' ')], dtype=np.int32)
        self.unique_cards, counts = np.unique(deck, return_counts=True)
        self.probas = counts.astype(np.float32) / self.num_of_cards

    def expectation_of_minimum(self):
        n = 4
        cumulative_probas = np.cumsum(self.probas[::-1])[::-1] ** n
        prob_xi_min = cumulative_probas - np.roll(cumulative_probas, -1)
        prob_xi_min[-1] = cumulative_probas[-1] 
        return np.sum(self.unique_cards * prob_xi_min)

    def expectation(self):
        mean_distribution = np.dot(self.unique_cards, self.probas)
        adjusted_mean = (4 * mean_distribution - self.expectation_of_minimum())
        return adjusted_mean

num_of_decks = int(input())
expectations = []
for _ in range(num_of_decks):
    num_of_cards, cards = input(), input()
    deck = Deck(num_of_cards, cards)
    expectations.append(deck.expectation())

max_expectation = max(expectations)
max_index = expectations.index(max_expectation) + 1
print(f"{max_index} {round(float(max_expectation), 3):.7f}")
