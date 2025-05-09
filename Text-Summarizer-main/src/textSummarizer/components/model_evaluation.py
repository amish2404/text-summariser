from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from evaluate import load  # Updated import
import torch
import pandas as pd
from tqdm import tqdm
from src.textSummarizer.entity import ModelEvaluationConfig


class ModelEvaluation:

    def __init__(self, config):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """
        Split the dataset into smaller batches that we can process simultaneously.
        Yield successive batch-sized chunks from list_of_elements.
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                                    column_text="dialogue", column_summary="summary"):
        """
        Calculate metric on the test dataset.
        """
        dialogue_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        summary_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for dialogue_batch, summary_batch in tqdm(zip(dialogue_batches, summary_batches), total=len(dialogue_batches)):
            inputs = tokenizer(
                dialogue_batch, 
                max_length=1024, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )
            
            # Decode the generated texts
            decoded_summaries = [
                tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for s in summaries
            ]
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            # Add predictions and references to the metric
            metric.add_batch(predictions=decoded_summaries, references=summary_batch)

        # Compute the final score
        score = metric.compute()
        return score

    def evaluate(self):
        """
        Main evaluation method.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained("artifacts/model_trainer/tokenizer")
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained("artifacts/model_trainer/pegasus-samsum-model").to(device)

        # Loading data
        dataset_samsum_pt = load_from_disk("artifacts/data_transformation/samsum_dataset")

        # Using the updated `evaluate` library
        rouge_metric = load("rouge")  # Updated call
        score = self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer, batch_size=2, column_text="dialogue", column_summary="summary"
        )

        # Extracting values directly since score contains float values
        rouge_dict = {rn: score[rn] for rn in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}

        df = pd.DataFrame(rouge_dict, index=['pegasus'])
        df.to_csv("artifacts/model_evaluation/metrics.csv", index=False)
