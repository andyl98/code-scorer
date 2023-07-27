from typing import List, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CodeScorer:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=1).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device

    def score(self, codes: Union[str, List[str]]) -> List[float]:
        if isinstance(codes, str):
            codes = [codes]
        if len(codes) == 0:
            return []
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                codes,
                truncation=True,
                padding=True,
                max_length=1000,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = [float(x) for x in logits.squeeze(-1).tolist()]
            return preds


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="weights")
    args = parser.parse_args()
    scorer = CodeScorer(args.model)
    while True:
        print("now accepting from stdin (end with \"<END>\" on a line)")
        lines = []
        while True:
            line = input()
            if line.endswith("<END>") or line.endswith("<end>"):
                line = line[:-5]
                lines.append(line)
                break
            lines.append(line)
        code = "\n".join(lines)
        print(scorer.score(code))
