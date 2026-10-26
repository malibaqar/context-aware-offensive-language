# Context-Aware Offensive Language Detection
### NLP evaluation workflow
**Python · Transformers · scikit-learn · DistilBERT**

A reconstructed evaluation component inspired by the academic project **Context-Aware Offensive Language Detection Using Intent Classification and AI Agents**.

The original project explored a DistilBERT classifier and a DistilGPT2 context generator. **This repository currently contains evaluation only**; it does not contain the original training pipeline, context-generation integration, dataset or model checkpoint.

## What it does
- Reads a held-out CSV with `tweet` and `class` columns.
- Runs a supplied text-classification checkpoint.
- Maps model label names to dataset classes explicitly.
- Reports sample count, accuracy, macro-F1 and a confusion matrix.

## Run

```bash
python -m venv .venv
# Activate the environment for your operating system.
pip install -r requirements.txt
python evaluate.py --model ./checkpoint --csv heldout.csv --labels '{"LABEL_0":"0","LABEL_1":"1","LABEL_2":"2"}'
```

**Check the actual training label order first.** The mapping above is an example, not a verified mapping for a missing checkpoint. A base DistilBERT model is not a trained offensive-language classifier.

See [RUN.md](RUN.md) for data-splitting and evaluation notes.

## Evaluation design
Keep the evaluation set separate before training. Inspect per-class errors and macro-F1 alongside accuracy, especially when class frequencies differ. Historical project results are not reproduced by this repository and are not presented as current benchmarks.

## Limits and responsible use
Offensive-language labels depend on context and annotation choices. Generated context is hypothetical, not evidence of a speaker’s intent. A classifier’s output should not be treated as a definitive judgment about a person.

## Provenance
Educational reconstruction prepared with AI assistance for Muhammad Ali Baqar on 27 October 2026. Python syntax was checked; trained-model inference was not run. Original data, checkpoints and experiment logs remain unavailable. Respect dataset licensing and privacy before adding examples.
