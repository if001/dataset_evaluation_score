# readme

- distinct-N
- diversity coefficient
- Self-BLEU
-  LLM cluster score
-  DScore

## distinct-N
生成された文章群におけるユニークなn-gram（n語連続）の割合を測定します。例えば、Distinct-1はユニークな単語の割合、Distinct-2はユニークな2語連続のフレーズの割合を示します。この指標は、生成された文章の語彙的多様性を評価するのに有効です

https://github.com/neural-dialogue-metrics/Distinct-N

## diversity coefficient
https://github.com/brando90/beyond-scale-language-data-diversity/tree/main/src/diversit

## Self-BLEU
BLEUの仕組みを応用して、生成されたテキスト群全体の多様性を評価します。BLEUが「生成文」対「正解文」の類似度を測るのに対し、Self-BLEUは「生成文」対「他の生成文」の類似度を測ります。

* Self-BLEUスコアが高い場合: *
これは、テキスト群の中の各文が、他の文と非常に似ている（BLEUスコアが高い）ことを意味します。つまり、生成されたテキストの多様性が低く、モデルが同じような文ばかりを生成している「モード崩壊 (mode collapse)」が起きている可能性を示唆します。 

* Self-BLEUスコアが低い場合: *
これは、テキスト群の中の各文が、他の文とあまり似ていない（BLEUスコアが低い）ことを意味し、多様性に富んだテキストが生成されていることを示します。

https://arxiv.org/abs/1802.01886
https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py

## LLM cluster score
## DCScore
https://github.com/bluewhalelab/dcscore


