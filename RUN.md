Install requirements in a virtual environment. Supply a locally trained classifier checkpoint and a held-out CSV with tweet,class columns.
Run `python evaluate.py --model ./checkpoint --csv heldout.csv --labels '{"LABEL_0":"0","LABEL_1":"1","LABEL_2":"2"}'` only after checking the actual training label order.
Split data before training and keep evaluation data untouched. A base DistilBERT checkpoint is not a trained offensive-language classifier. This reconstruction implements evaluation only; training and the original context-generation integration are not recovered.
