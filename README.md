# Shakespeare GPT

This is a small implementation the original implementation from the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper.
Since the original paper was made for text translation, I only used the decoder part, to create a GPT.


This is what my model generates: (you can read a longer version [here](./output10000.txt)
<pre>
IO:
Nay!

OXFORD:
Art my maid is gloves affection, I am yift,
My lord, first, and good valour together;
And all the wacks we holl.
What news the prince watch is that way of her,
And yet but that life

</pre>
The entire project is very inspired by [micrograd](https://github.com/karpathy/micrograd)

Everything was made for educational purpouses.


## Contents
I focused mainly on the transformer (the interesting part), but this project has 2 other side models which helped me understand the matter better.
This project has 3 separate things, each one has its own jupyter notebook:
 * A [bigram](https://en.wikipedia.org/wiki/Bigram)
 * A very small sized text model
 * The actual "Attention is all you need" paper implementation


 A few notes:
* The first two were made in order to introduce myself to the matter of text generation. Therefore they are implemented on a very low level.
* The first two models predict (create) new "name sounding" words.
* The big model predicts (creates) sheakepeare-resk texts.

## Results

These are the results I obtained from each model:
*I'm sure could fine tune the hyperparamteres better for each model, but that wasn't the point for this project.*

Bigram:
  * PSOPENEY
  * BLEONDARSKICAT
  * AKEITUR
  * BATTEVERA
  * QUHEWAN
  * ROSAWAWSESKSHARINTTLSAIEKIT

Small model:
  * WALFAILMAN
  * TATTIN
  * DUS
  * SETTE
  * BURG
  * MAGNIS

Transformer:
<pre>
IO:
Nay!

OXFORD:
Art my maid is gloves affection, I am yift,
My lord, first, and good valour together;
And all the wacks we holl.
What news the prince watch is that way of her,
And yet but that life

</pre>


## Datasets used
1. For the transformer, I used a collection of shakespeare works as training data. This was taken from [here](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

2. For the names I used a large list of names. Taken from [here]( https://gist.githubusercontent.com/craigmartin97/e98a9e2a267c379e47be1191d9431de2/raw/c09c7356e85e39e41faa92a665b7ef0b3b840b6a/last-names.txt)
