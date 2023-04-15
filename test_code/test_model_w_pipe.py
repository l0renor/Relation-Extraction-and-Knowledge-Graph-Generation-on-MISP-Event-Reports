from transformers import  T5ForConditionalGeneration, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import utils_re
import utils_re_strict
import utils_re_pipe
import preprocessing as preprocessing

model_ckpt = "Olec/cyber_rebel"
token_repo = model_ckpt# '/tmp/pycharm_project_229/train_t5/models/preprocessing_selected_small'


tokenizer = AutoTokenizer.from_pretrained(token_repo)

model =  T5ForConditionalGeneration.from_pretrained(model_ckpt).to("cuda")


dataset = load_dataset("Olec/cyber-threat-intelligence_v2")



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

nlp = preprocessing.setup_pipeline()
#ds = dataset["test"].shard(num_shards=10, index=1)
dataset = dataset.map(create_label_and_sub)

print("Metrics_test:")
met = utils_re_pipe.get_metrics_from_dataset(dataset["test"],model, tokenizer)
print(met)

print("Metrics_val:")
met = utils_re_pipe.get_metrics_from_dataset(dataset["validation"],model, tokenizer)
print(met)



#Metrics_test:
#{'precision': 0.37314814814814806, 'recall': 0.4463734567901234, 'f1': 0.4064894147513486}
#Metrics_val:
#{'precision': 0.3473684210526316, 'recall': 0.38316885964912284, 'f1': 0.3643914288535598}