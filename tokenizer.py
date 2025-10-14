import torc
import torch
from collections import Counter


class Tokenizer:
    def __init__(self, text="", max_vocab_size=100, save=True, load=False, DEBUG=True):
        self.MAX_VOCAB_SIZE = max_vocab_size
        self.text = text
        self.vocab = None
        self.vocab_size = 0
        self.transformations = []

        self.create_vocab()

        if load:
            weights = torch.load("tokenizer_weights.pt")
            self.transformations = weights["transformations"]
            self.itc = weights["itc"]
            self.cti = weights["cti"]
        else:
            self.itc, self.cti = Tokenizer.create_dicts(text)
            encoded = self.encode(self.text)
            self.transformations = self.train(encoded)

        if save:
            torch.save(
                {
                    "transformations": self.transformations,
                    "itc": self.itc,
                    "cti": self.cti,
                },
                "tokenizer_weights.pt",
            )

        if DEBUG:
            print("Training completed:")
            print(f"\tGot {len(self.transformations)} transformations")
            print(f"\tThese are some of the most used ones:")
            for key, token in self.transformations[:10]:
                print("\t\t'" + self.itc[key[0]] + self.itc[key[1]] + "'")

    def train(self, encoded):
        it = self.MAX_VOCAB_SIZE
        transformations = []
        while it - self.vocab_size:
            new_key, new_token = self.get_next_token(encoded)
            transformations.append((new_key, new_token))
            Tokenizer.join_pair(encoded, new_key, new_token)
            it -= 1
        return transformations

    @staticmethod
    def create_dicts(text):
        itc = dict(list(enumerate(set(list(text)))))
        cti = {v: k for k, v in itc.items()}
        return itc, cti

    def create_vocab(self):
        self.vocab = set(list(self.text))
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        # itc, cti = Tokenizer.create_dicts(text)
        return [self.cti[c] for c in list(text)]

    def tokenize(self, encoded):
        for key, token in reversed(self.transformations):
            encoded = Tokenizer.join_pair(encoded, key, token)
        return encoded

    def decode(self, encoded):
        decoded = [self.itc[c] for c in encoded]
        return decoded

    def get_next_token(self, encoded):
        # encoded = encoded + [0] * (len(encoded) % 2)
        even = torch.tensor(encoded[:-1]).reshape(-1, 1)
        odd = torch.tensor(encoded[1:]).reshape(-1, 1)

        all_pairs = torch.cat((even, odd), dim=1)
        # pairs = {}
        # for pair in all_pairs.tolist():
        #     pair = tuple(pair)
        #     try:
        #         pairs[pair] += 1
        #     except:
        #         pairs[pair] = 1
        #
        all_pairs = [
            tuple(pair) for pair in all_pairs.numpy()
        ]  # or t.cpu().numpy() if on GPU
        pairs = dict(Counter(all_pairs))

        new_key = max(pairs, key=pairs.get)

        # print(new_key)

        new_token = self.vocab_size

        self.itc[new_token] = self.itc[new_key[0]] + self.itc[new_key[1]]
        self.vocab_size += 1
        # print("'" + self.itc[new_key[0]] + self.itc[new_key[1]] + "'")
        return new_key, new_token

    @staticmethod
    def join_pair(encoded, key, new_token):
        changed = 0
        for i in range(len(encoded) - 2):
            if tuple(encoded[i : i + 2]) == key:
                del encoded[i : i + 2]
                changed += 1
                encoded.insert(i, new_token)

        return encoded


if __name__ == "__main__":
    # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    text = open("input.txt").read()

    tk = Tokenizer(text, save=False, load=True)

    assert text[:10000] == "".join(tk.decode(tk.tokenize(tk.encode(text[:10000]))))
    pritn(tk.vocab_size)
