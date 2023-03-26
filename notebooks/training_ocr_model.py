# Databricks notebook source
# MAGIC %md
# MAGIC # Setup Environment
# MAGIC This notebook is running on:
# MAGIC - Runtime: Databricks Runtime Version 12.2 LTS ML (GPU)
# MAGIC - Driver: g4dn.4xlarge 64 GB Memory, 1 GPU
# MAGIC - Worker: g4dn.4xlarge 64 GB Memory, 1 GPU (1 Node)

# COMMAND ----------

!pip install --upgrade pip
!pip install -q "transformers==4.27.2"
!pip install -q datasets sentencepiece tensorboard

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data download and loading
# MAGIC Start by downloading the receipt dataset to DBFS.

# COMMAND ----------

# DOWNLOAD DATA
# Create DBFS (Database Filesystem)
dbfs_path = "/users/julienguyen"
dbutils.fs.mkdirs(dbfs_path)

# clone repository
!git clone https://github.com/zzzDavid/ICDAR-2019-SROIE.git
# copy data
!cp -r ICDAR-2019-SROIE/data ./
# clean up
!rm -rf ICDAR-2019-SROIE
!rm -rf data/box

# move to DBFS
dbutils.fs.cp("file:/databricks/driver/data", "dbfs:"+dbfs_path, recurse=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data preparation

# COMMAND ----------

import os
import json
from pathlib import Path
import shutil

# define paths
base_path = Path("data")
metadata_path = base_path.joinpath("key")
image_path = base_path.joinpath("img")
# define metadata list
metadata_list = []

# parse metadata to aggregate receipt labels with picture filename
for file_name in metadata_path.glob("*.json"):
    with open(file_name, "r") as json_file:
        # load json file
        data = json.load(json_file)
        # create "text" column with json string
        text = json.dumps(data)
        # add to metadata list if image exists
        if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
            metadata_list.append({"text":text,"file_name":f"{file_name.stem}.jpg"})
            # delete json file

# write results in jsonline file
with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
    for entry in metadata_list:
        json.dump(entry, outfile)
        outfile.write('\n')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading the dataset
# MAGIC 
# MAGIC Dataset features below need to contain image and text
# MAGIC 
# MAGIC If it doesen't contain 2 features (only image) you might need to look at code above again

# COMMAND ----------

from datasets import load_dataset

# define paths
base_path = Path("data")
metadata_path = base_path.joinpath("key")
image_path = base_path.joinpath("img")

# load dataset
dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

print(f"Dataset has {len(dataset)} images")
print(f"Dataset features are: {dataset.features.keys()}")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's visualize some example from the dataset.

# COMMAND ----------

import random

random_sample = random.randint(0, len(dataset))

print(f"Random sample is {random_sample}")
print(f"OCR text is {dataset[random_sample]['text']}")
dataset[random_sample]["image"].resize((250,400))


# COMMAND ----------

# MAGIC %md
# MAGIC # Prepare dataset for Donut
# MAGIC We want it to generate the "text" based on the image we pass it.
# MAGIC 
# MAGIC To easily create those documents the ClovaAI team has created a json2token method, which we extract and then apply.
# MAGIC 
# MAGIC Source: [ClovaAI](https://github.com/clovaai/donut/blob/master/donut/model.py#L497)

# COMMAND ----------

new_special_tokens = [] # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>" # eos token of tokenizer

def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                output += (
                    fr"<s_{k}>"
                    + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                    + fr"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join(
            [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
        )
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut(sample):
    # create Donut-style input
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
    # convert all images to RGB
    image = sample["image"].convert('RGB')
    return {"image": image, "text": d_doc}

def preprocess_image_for_inference(image):
    return image.convert('RGB')

# Below command might take around 10 mins
proc_dataset = dataset.map(preprocess_documents_for_donut)

print(f"Sample: {proc_dataset[45]['text']}")
print(f"New special tokens: {new_special_tokens + [task_start_token] + [eos_token]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tokenize text and Encode the images
# MAGIC Next step is to tokenize our text and encode the images into tensors. Therefore we need to load `DonutProcessor`, add our new special tokens and adjust the size of the images when processing from [1920, 2560] to [720, 960] to need less memory for faster training.

# COMMAND ----------

from transformers import DonutProcessor

# Load processor
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
# add new special tokens to tokenizer
processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})

# COMMAND ----------

def transform_and_tokenize(sample, processor=processor, split="train", max_length=512, ignore_id=-100):
    # create tensor from image
    try:
        pixel_values = processor(
            sample["image"], random_padding=split == "train", return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}

# resizing the image to smaller sizes from [1920, 2560] to [960,1280]
processor.feature_extractor.size = [720,960] # should be (width, height)
processor.feature_extractor.do_align_long_axis = False

# need at least 32-64GB of RAM to run this
processed_dataset = proc_dataset.map(transform_and_tokenize, remove_columns=["image","text"])

# Split dataset for train/test set
processed_dataset = processed_dataset.train_test_split(test_size=0.1)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the Donut Model and Configure HF Model Trainer
# MAGIC 
# MAGIC Donut consists of an image Transformer encoder and an autoregressive text Transformer decoder to perform document understanding tasks such as document image classification, form understanding and visual question answering.
# MAGIC 
# MAGIC Here, we are loading the [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base) model with the `VisionEncoderDecoderModel class` The donut-base includes only the pretrained weights.
# MAGIC 
# MAGIC 
# MAGIC In addition to loading our model, we are resizing the `embedding` layer to match newly added tokens and adjusting the `image_size` of our encoder to match our dataset.

# COMMAND ----------

import torch
from transformers import VisionEncoderDecoderModel, VisionEncoderDecoderConfig

# Load model from huggingface.co
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Resize embedding layer to match vocabulary size
new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
print(f"New embedding size: {new_emb}")

# COMMAND ----------

# Adjust our image size and output sequence lengths
model.config.encoder.image_size = processor.feature_extractor.size[::-1] # (height, width)
model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))

# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]

# COMMAND ----------

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# hyperparameters used for multiple args
hf_repository_id = "hf_output_dir"

# Arguments for training
training_args = Seq2SeqTrainingArguments(
    output_dir=hf_repository_id,
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    save_total_limit=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    predict_with_generate=True,
    optim="adamw_torch"
)

# Create Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Tuning with Hugging Face and MLFlow
# MAGIC 
# MAGIC Training an OCR model using Hugging Face and MLFlow can be a powerful approach to building a robust and accurate OCR system. Hugging Face provides a rich set of tools for building and fine-tuning deep learning models, while MLFlow offers a streamlined way to manage the model training process, track metrics and experiment configurations, and reproduce models.
# MAGIC 
# MAGIC Once you have trained and validated your OCR model, you can save it as a custom MLFlow model and deploy it to a real-time endpoint using Model Serving. By managing the deployment and monitoring of your model, you will be able to track its performance in production over time.
# MAGIC 
# MAGIC The tuning takes about 15 minutes to complete for one epoch.

# COMMAND ----------

import mlflow
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from sys import version_info

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
      f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
      "pip",
      {"pip": [
          f"mlflow<3,>=2.1",
          f"boto3==1.21.32",
          f"cffi==1.15.0",
          f"cloudpickle==2.0.0",
          f"configparser==5.2.0",
          f"defusedxml==0.7.1",
          f"dill==0.3.6",
          f"fsspec==2022.2.0",
          f"ipython==8.5.0",
          f"pandas==1.4.2",
          f"pillow==9.0.1",
          f"scipy==1.7.3",
          f"sentencepiece==0.1.97",
          f"tensorflow==2.11.0",
          f"torch==1.13.1",
          f"transformers==4.27.2"
        ],
      },
    ],
    "name": "mlflow-env"
}


class CustomHuggingFaceModel(mlflow.pyfunc.PythonModel):
    
    def load_context(self, context=None, path=None):
        device = torch.device('cpu') # "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = processor
        
        if context:
            model_path = context.artifacts['pytorch_model']
        else:
            model_path = path
        
        self.model = mlflow.pytorch.load_model(model_path, map_location=torch.device('cpu')).eval()
    
    def _analyze_receipt(self, image_url: str):
        device = torch.device('cpu') 
        
        # download from url image jpg
        downloaded_img = requests.get(image_url).content
        formatted_img = Image.open(BytesIO(downloaded_img))
        array_img = processor(formatted_img, return_tensors="pt").pixel_values.squeeze()
        pixel_values = torch.tensor(array_img).unsqueeze(0)
        
        print("decoding image")
        task_prompt = "<s>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        print("starting inference")
        # run inference
        outputs = self.model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=self.model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id= self.processor.tokenizer.pad_token_id,
            eos_token_id= self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        print("processing output")
        prediction =  self.processor.batch_decode(outputs.sequences)[0]
        prediction =  self.processor.token2json(prediction)
        
        return prediction
    
    
    def predict(self, context, input_df):
        prediction_label = "predictions"
        device = torch.device('cpu')
        input_df[prediction_label] = input_df["url"].apply(self._analyze_receipt)
        return {prediction_label: input_df[prediction_label].values.tolist()}

# COMMAND ----------

from PIL import Image
import requests
from io import BytesIO

df_url = pd.DataFrame([{"url": "https://upload.wikimedia.org/wikipedia/commons/0/0b/ReceiptSwiss.jpg"}])
df_url

# COMMAND ----------

import logging

logging.getLogger("mlflow").setLevel(logging.DEBUG)

# COMMAND ----------

MLFLOW_FLATTEN_PARAMS = True


# COMMAND ----------

#Old version
import pickle 
import mlflow

model_artifact_path = "julie_hugging_face_model_artifact"

# Start training
with mlflow.start_run() as run:
    trainer.train()
    
    mlflow.pytorch.log_model(trainer.model, artifact_path="pytorch-model", pickle_module=pickle)
    run_id = run.info.run_id

    pytorch_model_uri = f"runs:/{run_id}/pytorch-model"
    artifacts = {
        "pytorch_model": pytorch_model_uri
    }
    # log custom model
    mlflow.pyfunc.log_model(artifacts = artifacts, python_model=CustomHuggingFaceModel(), artifact_path = model_artifact_path, conda_env=conda_env, input_example=df_url)


# COMMAND ----------

custom_model_uri = f"runs:/{run_id}/{model_artifact_path}"
model_name = 'ocr_hf_receipts'
model_details = mlflow.register_model(model_uri=custom_model_uri, name=model_name)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# from transformers import pipeline
# from mlflow.models.signature import infer_signature
# import pickle 
# import mlflow
# model_output_dir = "./julie_hugging_face_model"
# pipeline_output_dir = "./julie_hugging_face_model_pipeline"
# model_artifact_path = "julie_hugging_face_model_artifact"

# MLFLOW_FLATTEN_PARAMS = True

# # Start training
# with mlflow.start_run() as run:
#     run_id = run.info.run_id
    
#     #trainer.train()

# #     mlflow.pytorch.log_model(trainer.model, artifact_path="pytorch-model", pickle_module=pickle)
# #     pytorch_model_uri = f"runs:/{run_id}/pytorch-model"
# #     artifacts = {"pytorch_model": pytorch_model_uri}
    
# #     mlflow.pyfunc.log_model(artifacts=artifacts, python_model=CustomHuggingFaceModel(), artifact_path=model_artifact_path)

#     dbutils.fs.rm("file:/databricks/driver/pytorch-model", True)
#     mlflow.pytorch.save_model(pytorch_model=trainer.model, path="pytorch-model", pickle_module=pickle)

#     artifacts = {"pytorch_model": "./pytorch-model"}
#     mlflow.pyfunc.log_model(artifacts=artifacts, python_model=CustomHuggingFaceModel(), artifact_path=model_artifact_path, input_example=df_url)
#     #mlflow.log_metrics(trainer.evaluate())
    

# COMMAND ----------

# MAGIC %md
# MAGIC ## Testing Model Wrapper Input

# COMMAND ----------

from pyspark.sql.functions import struct, col
import mlflow
logged_model = 'runs:/9587f647ae6b47f3abd2055ab2eac2c3/julie_hugging_face_model_artifact'
# Load model as model
loaded_model_class = mlflow.pyfunc.load_model(model_uri=logged_model)
#loaded_model_class.predict(df_url)

# COMMAND ----------


