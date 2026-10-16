# Word-level autoregressive text generation on Alice in Wonderland

## Problem and objective

Predict the next word from the preceding 12 tokens and generate 50-token continuations. This project compares a Simple RNN, a bidirectional RNN, an LSTM, and a GRU under the same data split, embedding size, hidden width, optimizer, batch size, and maximum training epochs. The bidirectional model reads both directions **within the observed context window**; the target and future text are never supplied to it. It can therefore be used autoregressively by rebuilding the observed window after each sampled word.

## Dataset and preprocessing

Lewis Carroll, *Alice's Adventures in Wonderland*, Project Gutenberg eBook 11: https://www.gutenberg.org/ebooks/11 . Plain text used by the script: https://www.gutenberg.org/cache/epub/11/pg11.txt . The script downloads the text on its first run to `data/pg11.txt`; this downloaded copy is excluded from Git. Gutenberg headers and footers are removed using their book markers. Text is lowercased and tokenized into words (including internal apostrophes) and the punctuation `.,!?;:`. Other formatting and symbols are discarded. For example, `Alice was beginning to` becomes `[alice, was, beginning, to]`.

The cleaned corpus has **31,934 tokens**. Chronological, disjoint token partitions contain **25,547 training**, **3,193 validation**, and **3,194 test** tokens. Each example is a length-12 context and its immediately following token; sliding by one gives 25,535 training windows, 3,181 validation windows, and 3,182 test windows. Windows never cross partitions. The **1,280-token vocabulary** is learned only from training words occurring at least twice, plus `<pad>` and `<unk>`. Unseen and rare validation/test words map to `<unk>` (9.71% and 10.61% of their tokens, respectively). Padding is used only for short generation seeds. Unknown and padding symbols are disallowed as *generated* words, although unknown words remain possible targets during evaluation.

## Models and experimental conditions

All models use 48-dimensional embeddings, a single recurrent layer with hidden size 64, and a linear vocabulary classifier. For the bidirectional RNN, the final forward state and first backward state summarize the **observed** context. Parameter counts differ by recurrent architecture, so training conditions are comparable but model capacity is not exactly matched. All models use Adam (learning rate 0.002), batch size 128, gradient norm clipping at 1, random seed 42, CPU with two PyTorch threads, and eight epochs. The epoch with lowest validation cross-entropy is selected independently for each model; only then is the held-out test set evaluated once. Training time excludes test evaluation and text generation.

## Run

Use Python 3.10 or newer and an internet connection for the initial corpus download.

```bash
python -m pip install -r requirements.txt
python train.py
```

The program writes `results/metrics.json`, `results/generated_samples.md`, and `results/validation_loss.png`. To make a shorter trial, use `python train.py --epochs 2 --max-train-windows 3000`; those trial values will not reproduce the table below. PyTorch's CPU kernels and wall-clock timing can vary across machines.

## Results from the supplied run

Cross-entropy and perplexity are lower when better; next-word top-1 accuracy is higher when better. Test metrics use 3,182 held-out windows. Training times are measured on this environment and should only be compared within this run.

| Model | Parameters | Selected epoch | Train time (s) | Test loss | Test perplexity | Test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple RNN | 151,936 | 8 | 8.2 | 4.196 | 66.422 | 21.0% |
| Bidirectional RNN | 241,152 | 5 | 14.2 | 4.278 | 72.072 | 20.2% |
| LSTM | 173,824 | 8 | 11.4 | 4.170 | 64.748 | 22.5% |
| GRU | 166,528 | 8 | 14.4 | **4.161** | **64.143** | **23.0%** |

Values are rounded here; exact results and per-epoch histories are in [`metrics.json`](metrics.json). The plot [`validation_loss.png`](validation_loss.png) shows validation loss by epoch. The bidirectional RNN's training loss kept falling after epoch 5 while validation loss rose, suggesting overfitting on this small corpus. Its bidirectionality also cannot see future target words during generation, which limits the benefit often seen in tasks with full-sequence access.

## Text generation and temperature analysis

Each model continues the **same two seeds** (`alice was beginning to`; `the rabbit said to alice`) for 50 sampled tokens at temperatures **0.5, 1.0, and 1.5**. Sampling scales logits by `1 / temperature` before softmax. Lower temperatures sharpen frequent-word probabilities; higher temperatures flatten them. The complete 24 outputs are in [`generated_samples.md`](generated_samples.md).

The table reports distinct bigram fraction averaged across the two continuations per setting. A lower value generally indicates more reuse of phrases; it is not a direct measure of grammatical quality.

| Model | T = 0.5 | T = 1.0 | T = 1.5 |
| --- | ---: | ---: | ---: |
| Simple RNN | 0.878 | 0.990 | 1.000 |
| Bidirectional RNN | 0.949 | 1.000 | 1.000 |
| LSTM | 0.888 | 0.969 | 1.000 |
| GRU | 0.796 | 0.990 | 1.000 |

At **0.5**, frequent patterns such as `said to herself` and `and the` recur, especially for the GRU, but some clauses stay locally plausible and characters remain in the story world. At **1.0**, phrase variety rises, while dialogue and subject agreement become less reliable. At **1.5**, distinct bigrams approach 1.0 but unusual word combinations and abrupt character/topic shifts make the text less coherent. Immediate identical-word repetition is rare in these short samples (zero in most settings); repeated *phrases* and familiar sentence frames matter more here. None of the four models sustains a coherent plot over 50 tokens, consistent with the small single-book dataset and 12-token context.

The GRU is the best **overall** for this experiment: it has the lowest test loss and perplexity, the highest test accuracy, and at low temperature often produces locally plausible Alice-themed clauses. The LSTM is close. Use **temperature 0.5** when local coherence matters more than variety; temperature 1.0 gives more lexical variety with weaker continuity. The selected model is based primarily on held-out prediction metrics, not an unsupported claim of human-level prose quality.

## Limitations and reproducibility

This is one corpus, one split, and one initialization. The vocabulary discards rare word identities, and the experiment does not report statistical uncertainty across seeds. `distinct-2` can reach 1.0 simply because a short sample contains few repeated bigrams. The train-time comparison is sensitive to hardware and implementation. The CPU results in `metrics.json` are the actual output from the supplied script. No pretrained models are used.
