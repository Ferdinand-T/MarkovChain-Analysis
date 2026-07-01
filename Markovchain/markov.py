import random
import math
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import networkx as nx



class MarkovChain:
    # order = how many previous words we look at before picking the next one.
    # order=1 means "given this one word, what comes next" (classic Markov chain).
    # Higher order = more coherent but less varied text (fewer repeated states to learn from).
    def __init__(self, order=1, show_heatmap=True, show_graph=True, show_hexbin=True):
        self.order = order
        self.show_heatmap = show_heatmap
        self.show_graph = show_graph
        self.show_hexbin = show_hexbin
        self.transitions = {}
     

    @staticmethod
    def _shannon_entropy(items):
        # Shannon entropy = how "surprising"/unpredictable a sequence of items is,
        # measured in bits. All items showing up equally often -> high entropy (unpredictable).
        # One item dominating -> low entropy (predictable). Used here just to describe
        # the generated text, not to influence generation itself.
        if not items:
            return 0.0
        items = [str(item).lower() for item in items]
        counts = Counter(items)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _bigram_entropy(words):
        # Same idea as _shannon_entropy, but over consecutive word *pairs* instead of
        # single words. This captures repetition/structure that single-word entropy misses
        # (e.g. "the the the" repeated has low word entropy already, but a chain that
        # loops through the same phrases will show low bigram entropy too).
        words = [str(w).lower() for w in words]
        if len(words) < 2:
            return 0.0
        bigrams = [tuple(words[i:i+2]) for i in range(len(words) - 1)]
        counts = Counter(bigrams)
        total = sum(counts.values())
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _plot_heatmap(self, matrix):
        # Straightforward view: darker/lighter cells show which state -> state
        # transitions are more or less likely. Great for spotting dominant patterns.
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, cmap="YlGnBu", annot=False, fmt=".2f")
        plt.title(f"Markov Chain Transition Matrix (Order {self.order})")
        plt.show()

    def _plot_graph(self, matrix):
        # Turns the matrix into an actual graph: one node per state, one directed
        # edge per transition that actually happens (probability > 0). Good for
        # seeing the "shape" of the chain (chains, loops, hubs) rather than raw numbers.
        G = nx.DiGraph()
        for from_state in matrix.index:
            for to_state in matrix.columns:
                prob = matrix.at[from_state, to_state]
                if prob > 0:
                    G.add_edge(from_state, to_state, weight=round(prob, 3))
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=400, node_color="lightblue",
                font_size=8, font_weight="bold", arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'), font_size=4)
        plt.title(f"Markov Chain State Transition Graph (Order {self.order})")
        plt.show()

    def _plot_hexbin(self, matrix):
        # Scatters every non-zero transition as a point (from-index, to-index) and bins
        # them into hexagons. Useful mainly for very large matrices where a heatmap would
        # just look like solid noise, this shows density of activity instead.
        plt.figure(figsize=(10, 8))
        x, y = [], []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix.iat[i, j] > 0:
                    x.append(i)
                    y.append(j)
        plt.hexbin(x, y, gridsize=30, cmap='viridis', mincnt=1)
        plt.colorbar(label='Count of Transitions')
        plt.title(f"Markov Chain Transition Hexbin (Order {self.order})")
        plt.xlabel('From State')
        plt.ylabel('To State')

        # If there are a lot of states, labeling every single tick would be an
        # unreadable mess, so we only label a spaced-out sample of them instead.
        state_labels = list(matrix.index)
        matrix_size = matrix.shape[0]
        if matrix_size > 20:
            step = max(1, matrix_size // 20)
            tick_indices = list(range(0, matrix_size, step))
            tick_labels = [state_labels[i] for i in tick_indices]
            plt.xticks(tick_indices, tick_labels, rotation=45, ha='right', fontsize=8)
            plt.yticks(tick_indices, tick_labels, fontsize=8)
        else:
            plt.xticks(range(matrix_size), state_labels, rotation=45, ha='right', fontsize=8)
            plt.yticks(range(matrix_size), state_labels, fontsize=8)

        plt.tight_layout()
        plt.show()

    def train(self, file_path):
        print(f"Training Markov Chain (order={self.order}) on {file_path}...")
        with open(os.path.join(os.path.dirname(__file__), file_path), 'r', encoding="utf-8") as f:
            text = f.read()

        words = text.split()
        print(f"Loaded {len(words)} words.")

        # Step 1 - walk through the text and record, for every `order`-word window,
        # which word actually followed it (there can be many, hence a list).
        # e.g. order=1, state=("the",) -> ["cat", "dog", "cat", ...]
        for i in range(len(words) - self.order):
            state = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            if state not in self.transitions:
                self.transitions[state] = [next_word]
            else:
                self.transitions[state].append(next_word)

        # last state has no natural successor, loop back to the beginning
        last_state = tuple(words[-self.order:])
        if last_state not in self.transitions:
            self.transitions[last_state] = [words[0]]

        # Step 2 - turn those "state -> [next words seen]" lists into an actual
        # probability matrix: rows/cols are states, cell = P(next_state | state).
        states = sorted(self.transitions.keys())
        state_index = {state: idx for idx, state in enumerate(states)}
        matrix_size = len(states)
        matrix = [[0.0] * matrix_size for _ in range(matrix_size)]

        for state, next_words in self.transitions.items():
            cur = state_index[state]
            counts = Counter(next_words)
            total = sum(counts.values())
            for next_word, count in counts.items():
                # Sliding the window forward by one word: drop the oldest word in
                # the state and tack the new one on the end.
                next_state = tuple(list(state[1:]) + [next_word]) if self.order > 1 else (next_word,)
                if next_state in state_index:
                    matrix[cur][state_index[next_state]] = count / total

        # Wrap it in a DataFrame so states can be looked up by their (joined) text
        # label instead of raw indices.
        self.transition_matrix = pd.DataFrame(
            matrix,
            index=[' '.join(s) for s in states],
            columns=[' '.join(s) for s in states]
        )

        if self.show_heatmap:
            self._plot_heatmap(self.transition_matrix)
        if self.show_graph:
            self._plot_graph(self.transition_matrix)
        if self.show_hexbin:
            self._plot_hexbin(self.transition_matrix)

        print("Training done.")

    def analyze(self, output):
        words = output.split()
        num_words = len(words)
        num_unique = len(set(words))
        lexical_diversity = num_unique / num_words if num_words > 0 else 0
        word_entropy = self._shannon_entropy([w for w in words if w])
        char_entropy = self._shannon_entropy([ch for ch in output if ch])
        bigram_entropy = self._bigram_entropy(words)

        print(f"--- Text Statistics (Order {self.order}) ---")
        print(f"Total words:       {num_words}")
        print(f"Unique words:      {num_unique}")
        print(f"Lexical diversity: {lexical_diversity:.4f}")
        print(f"Word entropy:      {word_entropy:.4f} bits")
        print(f"Char entropy:      {char_entropy:.4f} bits")
        print(f"Bigram entropy:    {bigram_entropy:.4f} bits")

        return {
            "total_words": num_words,
            "unique_words": num_unique,
            "lexical_diversity": lexical_diversity,
            "word_entropy": word_entropy,
            "char_entropy": char_entropy,
            "bigram_entropy": bigram_entropy,
        }

    def generate(self, length=50, starting_position=None):
        # Pick a random starting state if the caller didn't ask for a specific one.
        if starting_position is None:
            starting_position = random.choice(self.transition_matrix.index.tolist())
        if starting_position not in self.transition_matrix.index:
            raise ValueError("Starting position not found in trained model.")

        current_state = starting_position
        output = [current_state]
        found, not_found = 0, 0

        # Walk the chain: at each step, look up the row for the current state and
        # roll a weighted die over its possible next states (weighted by probability).
        # If a state has no recorded transitions (a dead end), fall back to jumping
        # to a random state instead of getting stuck.
        for _ in range(length - self.order):
            next_states = self.transition_matrix.loc[current_state]
            next_states = next_states[next_states > 0]
            if next_states.empty:
                not_found += 1
                next_state = random.choice(self.transition_matrix.index.tolist())
            else:
                next_state = random.choices(next_states.index, weights=next_states.values)[0]
                found += 1
            output.append(next_state.split()[-1])
            current_state = next_state

        print(f"Transitions — found: {found}, random fallback: {not_found}")
        return " ".join(output).replace("__END__", "").replace("__BEGIN__", "").strip()


if __name__ == "__main__":
    markov_chain = MarkovChain(order=1, show_heatmap=False, show_graph=False, show_hexbin=False)
    markov_chain.train("your_text_file.txt")  # Replace with your actual text file path
    generated_text = markov_chain.generate(length=200)
    print(generated_text)
    print()
    markov_chain.analyze(generated_text)
