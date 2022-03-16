import logging
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

logging.basicConfig(level=logging.ERROR)


class MyNer:
    def __init__(self, modelDir):
        self.ner = self.__initPipeline(modelDir)

    def __initPipeline(self, modelDir):
        model = AutoModelForTokenClassification.from_pretrained(modelDir)
        tokenizer = AutoTokenizer.from_pretrained(modelDir, model_max_length=512)
        model.eval()
        return pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="none")

    def __savePipeline(self):
        path = 'my_model_dir'
        self.ner.save_pretrained(path)
        with open(path + '/manifest.json', 'w') as file:
            json.dump({
                'type': 'huggingface_pipeline',
                'pipeline_class': type(self.ner).__name__,
                'tokenizer_class': type(self.ner.tokenizer).__name__,
                'model_class': type(self.ner.model).__name__,
            }, file, indent=2)

    def predict(self, text):
        return str(self.ner(text)).replace("'", '"')
