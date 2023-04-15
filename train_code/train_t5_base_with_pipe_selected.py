import sys

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

import utils_re_pipe
import utils_re_pipe_strict
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy

from transformers import T5ForConditionalGeneration, AutoTokenizer
import wandb
import confuse
import torch
import numpy as np
from transformers import DataCollatorForSeq2Seq, TrainerCallback
import preprocessing_selected as preprocessing


#Create Row With special Tokens found by NLP pipe  replaced by Class_name
def create_label_and_sub(row):
    doc = nlp(row['text'])
    sub_text, memory = preprocessing.replace_special_terms(doc,nlp)
    print(sub_text)
    print(memory)

    triplets = []
    xml_str = ''
    for relation in row['relations']:
        for entity in row['entities']:
            if entity['id'] == relation['from_id']:
                head_label = entity['label'].strip()
                head_text = (row['text'][entity['start_offset']:entity['end_offset']]).strip()
            if entity['id'] == relation['to_id']:
                subj_label = entity['label'].strip()
                subj_text = (row['text'][entity['start_offset']:entity['end_offset']]).strip()
        if 'subj_label' in locals() and 'head_label' in locals():  # fixes bug: if Entity Text and Relation Entity Text are not equal
            # print(row)
            head = {'text': head_text, 'label': '<' + head_label + '>'}
            subj = {'text': subj_text, 'label': '<' + subj_label + '>'}
            triplets.append({
                'head': {
                    'label': head['label'].lower().strip(),
                    'text': head['text'].strip()
                },
                'subj': {
                    'label': subj['label'].lower().strip(),
                    'text': subj['text'].strip()
                },
                'rel': relation['type'].strip()
            })
    for triplet in triplets:
        xml_str = xml_str + '<triplet>' + triplet['head']['text'] + triplet['head']['label'] + triplet['subj']['text'] + \
                  triplet['subj']['label'] + triplet['rel']

    for el in memory:
        if el[1] in xml_str:
            xml_str = xml_str.replace(el[1], el[0])



    max_source_length = 512
    target = tokenizer(xml_str, truncation=True, padding="longest", max_length=max_source_length,
                       add_special_tokens=True)  # padding='max_length',
    source = tokenizer(sub_text, truncation=True, padding="longest", max_length=max_source_length,
                       add_special_tokens=True)
    return {
        'input_ids': source['input_ids'],
        'attention_mask': source['attention_mask'],
        'labels': target['input_ids'],
        'text': sub_text,
        'text_label' : xml_str,
    }


def generate_andLog__examples(dataset):
    columns = ["sample_text", "model_output", "relations_actual", "expected_relations"]
    table = wandb.Table(columns=columns)

    for i in range(6):
        row = dataset[i]
        expected_relations = utils_re_pipe.parse_relations_from_row(row)
        sample_text = row['text']
        input_ids = tokenizer(sample_text, max_length=1024, truncation=True, padding='max_length',
                              return_tensors='pt').to(
            "cuda")
        summaries = model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                                   max_length=256)
        model_output = \
        [tokenizer.decode(s, skip_special_tokens=False, clean_up_tokenization_spaces=True) for s in summaries][0]
        relations_actual = utils_re_pipe.parse_model_output(model_output)
        table.add_data(sample_text, model_output, relations_actual, expected_relations)

    wandb.log({"examples": table})


class Eval_Call(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = utils_re_pipe.get_metrics_from_dataset(dataset=dataset_s["validation"], model=model, tokenizer=tokenizer)
        wandb.log(metrics)
        generate_andLog__examples(dataset_s["validation"])


def compute_metrics(eval):
    with torch.no_grad():
        metrics = utils_re_pipe.get_metrics_from_dataset(dataset=dataset_s["validation"], model=model, tokenizer=tokenizer)
        generate_andLog__examples(dataset_s["validation"])
   # metrics_str = {"strict_" + k: v for k, v in metrics_str.items()}

    #metrics = metrics_str | metrics
    return metrics
if len(sys.argv) < 2:
    print("Config File Name missing")
    exit()

congig_name = 'config.yaml'

congig_name = sys.argv[1]

config = confuse.Configuration("Train", __name__)
config.set_file(congig_name)
base_model_ckpt = config['base_model_ckpt'].get()

experiment = config["experiment"].get()
display_name = "preprocessing_selected"
ds_name = config["dataset_name"].get()
local_ds = config["local_dataset"].get()

wandb.init(project=experiment, name=display_name)

if local_ds:
    dataset = load_from_disk(ds_name)
else:
    dataset = load_dataset(ds_name)

ner_labels = []
for entities in dataset["train"]['entities']:
    for entity in entities:
        if '<' + (entity['label'].lower().strip()) + '>' not in ner_labels:
            ner_labels.append('<' + (entity['label'].lower().strip()) + '>')

relation_labels = []
for relations in dataset["train"]['relations']:
    for relation in relations:
        if '<' + (relation['type'].lower()) + '>' not in relation_labels:
            relation_labels.append('<' + (relation['type'].lower()) + '>')

special_tokens = [i for i in ner_labels]
special_tokens = special_tokens + [i for i in relation_labels] + ['<triplet>']

model_ckpt = base_model_ckpt
token_repo = base_model_ckpt
tokenizer = AutoTokenizer.from_pretrained(token_repo, use_auth_token=True)
model = T5ForConditionalGeneration.from_pretrained(model_ckpt, use_auth_token=True)
tokenizer.add_tokens(ner_labels)
model.resize_token_embeddings(len(tokenizer))

nlp = preprocessing.setup_pipeline()

dataset_s = dataset.filter(lambda example: example["relations"] != [])
dataset_s = dataset_s_p = dataset_s.map(create_label_and_sub)
#dataset_s_p = dataset_s_p.map(lambda example: {}, remove_columns=['id', 'text', 'entities', 'relations'])
print(dataset_s_p)

epochs = config["epochs"].get()
seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
training_args = TrainingArguments(output_dir='output',
                                  num_train_epochs=epochs,
                                  warmup_steps=500,
                                  per_device_train_batch_size=1,
                                  per_device_eval_batch_size=1,
                                  weight_decay=0.01,
                                  logging_steps=50,
                                  push_to_hub=False,
                                  evaluation_strategy='steps',
                                  eval_steps=100,
                                  save_steps=1e6,
                                  gradient_accumulation_steps=16,
                                  load_best_model_at_end=True,
                                  metric_for_best_model='f1',
                                  eval_accumulation_steps=1,
                                  report_to="wandb")

trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer,
                  data_collator=seq2seq_data_collator,
                  train_dataset=dataset_s_p["train"],
                  eval_dataset=dataset_s_p["validation"],)
                  #callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()

trainer.save_model("models/" + display_name)


metrics = utils_re_pipe.get_metrics_from_dataset(dataset=dataset_s["test"], model=model, tokenizer=tokenizer)
print(metrics)
#f = open(experiment + "_result.txt", "a")
#f.write(str(metrics))
#f.close()
wandb.run.summary["test_accuracy"] = metrics




metrics = utils_re_pipe_strict.get_metrics_from_dataset(dataset=dataset_s["test"], model=model, tokenizer=tokenizer)
print(metrics)
wandb.run.summary["test_accuracy_strict"] = metrics

wandb.finish()