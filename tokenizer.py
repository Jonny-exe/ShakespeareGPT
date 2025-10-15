import torch
from collections import Counter


class Tokenizer:
    def __init__(self, text="", max_vocab_size=120, save=True, load=False, DEBUG=True):
        self.MAX_VOCAB_SIZE = max_vocab_size
        self.text = text
        self.vocab = None
        self.vocab_size = 0
        self.transformations = []
        self.pairs = None

        self.create_vocab()

        if load:
            weights = torch.load("tokenizer_weights.pt")
            self.transformations = weights["transformations"]
            self.itc = weights["itc"]
            self.cti = weights["cti"]
        else:
            self.itc, self.cti = Tokenizer.create_dicts(text)
            encoded = self.encode(self.text)
            self.pairs = Tokenizer.get_pair_frequencies(encoded)
            self.transformations = self.train(encoded)

        if save:
            file_path = "tokenizer_weights.pt"
            torch.save(
                {
                    "transformations": self.transformations,
                    "itc": self.itc,
                    "cti": self.cti,
                },
                file_path,
            )
            print(f"Saving token weights to {file_path}")

        if DEBUG:
            print("Training completed:")
            print(f"\tGot {len(self.transformations)} transformations")
            print(f"\tThese are some of the most used ones:")
            for key, token in self.transformations[:10]:
                print("\t\t'" + self.itc[key[0]] + self.itc[key[1]] + "'")

    def train(self, encoded):
        transformations = []

        while self.vocab_size < self.MAX_VOCAB_SIZE:
            new_key, new_token = self.get_next_token()
            transformations.append((new_key, new_token))
            # print(self.itc)

            encoded = self.join_pair(encoded, new_key, new_token)
            # print(self.vocab_size)
        return transformations

    @staticmethod
    def create_dicts(text):
        itc = dict(list(enumerate(set(list(text)))))
        cti = {v: k for k, v in itc.items()}
        return itc, cti

    def create_vocab(self):
        self.vocab = set(list(self.text))
        print(f"{self.vocab=}")
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        # itc, cti = Tokenizer.create_dicts(text)
        return [self.cti[c] for c in list(text)]

    @staticmethod
    def load_tokenized():
        encoded = torch.load("tokenized_data.pt")["data"]
        return encoded

    def tokenize(self, encoded, save=False):
        i = 0
        print(self.transformations[:20])
        print(self.itc)

        for key, token in self.transformations:
            encoded = self.join_pair(encoded, key, token)
            print(
                f"{i} / {len(self.transformations)} \t {self.itc[key[0]] + self.itc[key[1]]}"
            )
            i += 1

        if save:
            file_path = "tokenized_data.pt"
            torch.save({"data": encoded}, file_path)
            print(f"Saving tokenized text to {file_path}")
        return encoded

    def decode(self, encoded):
        decoded = [self.itc[c] for c in encoded]
        return decoded

    @staticmethod
    def get_pair_frequencies(encoded):
        pairs = Counter()
        for i in range(len(encoded) - 1):
            pairs[(encoded[i], encoded[i + 1])] += 1
        return pairs

    def get_next_token(self):
        # encoded = encoded + [0] * (len(encoded) % 2)
        # even = torch.tensor(encoded[:-1]).reshape(-1, 1)
        # odd = torch.tensor(encoded[1:]).reshape(-1, 1)

        # all_pairs = torch.cat((even, odd), dim=1)
        # pairs = {}
        # for pair in all_pairs.tolist():
        #     pair = tuple(pair)
        #     try:
        #         pairs[pair] += 1
        #     except:
        #         pairs[pair] = 1
        #
        # all_pairs = [
        #     tuple(pair) for pair in all_pairs.numpy()
        # ]  # or t.cpu().numpy() if on GPU
        # pairs = dict(Counter(all_pairs))

        new_key = max(self.pairs, key=self.pairs.get)
        # print(new_key)

        new_token = self.vocab_size

        self.itc[new_token] = self.itc[new_key[0]] + self.itc[new_key[1]]
        self.vocab_size += 1
        # print("'" + self.itc[new_key[0]] + self.itc[new_key[1]] + "'")
        return new_key, new_token

    def join_pair(self, encoded, key, new_token):
        changed = 0
        result = []

        last_join = False
        removed = 0
        i = 0
        while i < len(encoded) - 1:
            join = (encoded[i], encoded[i + 1]) == key
            if join:
                result.append(new_token)

                if self.pairs is not None:
                    # print("-1: ", self.itc[key[0]] + self.itc[key[1]])
                    removed += 1
                    self.pairs[key] -= 1
                    self.pairs[(encoded[i - 1], encoded[i])] -= 1
                    self.pairs[(encoded[i + 1], encoded[i + 2])] -= 1

                    self.pairs[(result[-2], result[-1])] += 1
                    self.pairs[(new_token, encoded[i + 2])] += 1

            else:
                result.append(encoded[i])

            # if last_join:
            #     self.pairs[(result[-2], result[-1])] += 1

            i += 2 if join else 1

            last_join = join
            # if self.pairs is not None:
            #     self.pairs[key] += 1
        if not last_join:
            result.append(encoded[-1])
        # print(self.pairs[key], removed)

        return result


if __name__ == "__main__":
    # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    text = open("input.txt").read()
    print(f"{len(text)=}")

    tk = Tokenizer(text, max_vocab_size=200, save=True, load=False)
    # print("".join(tk.decode(tk.tokenize(tk.encode(text[:10000])))))
    # print(text[:10000])
    assert text[:10000] == "".join(tk.decode(tk.tokenize(tk.encode(text[:10000]))))
    print("|".join(tk.decode(tk.tokenize(tk.encode(text[:10000])))))

    tk.tokenize(tk.encode(text), save=True)  # save text
    print(tk.vocab_size)
