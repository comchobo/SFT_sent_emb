# Sentence embedding model for Korean language

Works were done while in Onboarding program of Ahancompany corporation ltd.,

You can find the fine-tuned model in [Huggingface](https://huggingface.co/sorryhyun/sentence-embedding-klue-large)

I wrote all detail in SFT_sent_emb.ipynb. V100 Colab GPU was used for training.

## Evaluation Results

| Organization | Backbone Model | KlueSTS average | KorSTS average |
| -------- | ------- | ------- | ------- |
| team-lucid | DeBERTa-base | 54.15(Finetuned) | 29.72 |
| monologg | Electra-base | 66.97(Finetuned) | 40.98 |
| LMkor | Electra-base | 70.98(Finetuned) | 43.09 |
| deliciouscat | DeBERTa-base | (Finetuned) | 67.65 |
| BM-K    | Roberta-base | 82.93 | **85.77**(Finetuned) |
| Klue*    | Roberta-large | **86.71**(Finetuned) | 71.70 |
| Klue* (Hyperparameter searched) | Roberta-large | 86.21(Finetuned) | 75.54 |

Asterisks denote 'Ours'
