Experiment figures and tables / 实验图表

figures/ : 7 core figures + appendices A1 / A2. PNG (300 dpi), PDF, SVG.
tables/results_tables.md : four completed Markdown tables, also split into table_1.md through table_4.md.
data/ : numerical inputs for every figure, question-level Fig.6 data and SHA256 source manifest.
generate_figures.py : reproduces figures, CSVs and combined Markdown tables from workspace evidence.
requirements-lock.txt : exact Python package versions used (Python 3.14).
preview.png : contact sheet for quick inspection.

Rebuild:
  python3 -m venv /tmp/dllm-figures
  /tmp/dllm-figures/bin/pip install -r Result/requirements-lock.txt
  /tmp/dllm-figures/bin/python Result/generate_figures.py

Figure notes:
1. Full SFT logs have 71 batch loss samples but only one epoch NLL record.
   Fig.1 uses trainer/loss, not an invented dense Train/Validation NLL series.
   CSV step+1 is the completed optimizer update count; generation checkpoints use saved steps.
2. Fig.2 uses the same random-MASK 16-question run throughout, not the FP32 code-only run.
   MASK axis is 50/90/100%; panel (b) excludes docstrings and PAD.
3. Fig.3 and Table 2 use the independent 500-step run, legacy full-output scoring.
4. Fig.4 uses successful programs / all-answer region. These are teacher-forced MASK states.
5. Fig.5 uses the same 8 questions and four shared independent MASK plans, step200 minus step0.
6. Fig.6 uses both original and fresh sampling batches (32 draws per question, 128 questions).
   Absolute-accuracy CIs are recomputed with 10,000 question-cluster bootstrap replicates,
   NumPy RNG seed 20261026; paired-difference CIs are taken from archived comparison.json.
   Repeated draws and sampling seeds are not independent questions or training-seed repeats.
7. Fig.7 uses same-machine 632-question comparison, 16 draws/question/temperature.
   Its CI contains zero: no demonstrated extra gain, not proof of equivalence or degradation.
A1. Positive 8/32/128 training loss and independent-MASK success logP are archived real records.
   Independent MASK means different masks on training programs, not unseen questions.

A2. Final Test comparison: absolute question-bootstrap CIs and archived paired difference CIs.
    Original 6853 records verified unchanged by pre-resume byte-prefix hashes.

Tables:
- Table 1 separates temperature, sampling seeds and hardware protocols; no cross-group ranking.
- Table 3 now uses the complete archived final-Test evidence: 7914 generations.
  Positive32 - Base = +0.493 pp, paired 95% CI [-0.493, +1.516] pp; no established stable gain.
- Table 4 keeps unrecorded/unextracted costs explicit. Timings and PyTorch peak allocated memory
  are scoped to each logged session, not total project cost. C200->711 is 511 resumed updates.
- Test has prior exploratory use; this is disclosed in the final benchmark.
- Final Test cost rows distinguish complete Base/SFT evaluations from Positive32 resume-only timing.

All visible labels are English to avoid missing Chinese-font glyphs in exported scientific plots.
No additional model training or generation evaluation was run to produce these figures.

final-test-20261026-01/ contains the complete remote experiment evidence, including outputs,
protocol, samplers/precision audits, original failure snapshot and repair metadata.
data/final_test_verification.json records local integrity checks.
