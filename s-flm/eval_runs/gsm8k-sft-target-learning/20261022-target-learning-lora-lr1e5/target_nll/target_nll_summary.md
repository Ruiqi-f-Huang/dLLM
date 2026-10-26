# GSM8K PoT teacher-forced validation loss

All checkpoints were evaluated with the same validation split, EMA weights, objective, seed, and batch settings. Lower NLL is better. `code_eos_nll` is the monitored non-padding supervised target region; `nll` follows the configured objective/mask.

| Checkpoint | val/code_eos_nll | val/nll | val/pad_nll | val/ppl |
|---|---:|---:|---:|---:|
| Base EMA | 1.864711 | 0.585519 | 0.002774 | 1.795923 |
| Step 100 EMA | 1.854016 | 0.582171 | 0.002773 | 1.789921 |
| Step 200 EMA | 1.760879 | 0.553019 | 0.002770 | 1.738494 |
| Step 711 EMA | 0.936112 | 0.294858 | 0.002730 | 1.342936 |
