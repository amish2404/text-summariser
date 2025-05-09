from src.textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = AutoTokenizer.from_pretrained("artifacts/model_trainer/tokenizer")
        self.pipe = pipeline("summarization", model="artifacts/model_trainer/pegasus-samsum-model", tokenizer=self.tokenizer)

    def predict(self, text, summary_type="short"):
        if summary_type == "short":
            gen_kwargs = {
                "max_length": 50,      # For 2 lines
                "min_length": 20,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }
        else:
            gen_kwargs = {
                "max_length": 150,    # For 4–5 lines
                "min_length": 60,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }

        print("Dialogue:")
        print(text)

        output = self.pipe(text, **gen_kwargs)[0]["summary_text"]

        print("\nModel Summary:")
        print(output)

        return output
