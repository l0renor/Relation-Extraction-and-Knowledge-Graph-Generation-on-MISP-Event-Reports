import Levenshtein as lev
def parse_relations_from_row(row):
    triplets = []

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
    return triplets


def create_str(triplets):
    xml_str = ''
    for triplet in triplets:
        xml_str = xml_str + '<triplet>' + triplet['head']['text'] + triplet['head']['label'] + triplet['subj']['text'] + \
                  triplet['subj']['label'] + triplet['rel']
    return xml_str


def parse_model_output(model_output):
    # clean special tokens from model output
    special_tokens = ['<pad>', '<s>', '</s>']
    for special_token in special_tokens:
        model_output = model_output.replace(special_token, "")
    #model_output = model_output.replace("<unk>", "<")
    triples_out = []
    triples = model_output.split('<triplet>')[1:]
    for triple in triples:
        try:
            triple_details = triple.replace('>', '<').split('<')
            triples_out.append(
                {
                    'head': {
                        'label': '<' + triple_details[1] + '>',
                        'text': triple_details[0].strip()
                    },
                    'subj': {
                        'label': '<' + triple_details[3] + '>',
                        'text': triple_details[2].strip()
                    },
                    'rel': triple_details[4].strip()
                })
        except:
            pass
    return triples_out


def get_metrics(relations_actual, relations_expected):
    true_positive = []  # die ermittelte und die tasächlcihe klasse ist korrekt -> actual wird in expected gefunden
    false_positive = []  # die ermittelte klasse ist korrekt und die tasächlcihe klasse ist falsch -> zu viel gefunden
    false_negative = []  # die ermittelte klasse ist falsch und die tasächlcihe klasse ist korrekt -> zu wenig gefunden

    # true positive
    for relation_actual in relations_actual:
        for relation_expected in relations_expected:
            if lev.distance(relation_expected['head']['text'], relation_actual['head']['text']) < 3 and \
                    relation_expected['head']['label'] == relation_actual['head']['label'] and lev.distance(
                    relation_expected['subj']['text'], relation_actual['subj']['text']) < 3 and \
                    relation_expected['subj']['label'] == relation_actual['subj']['label'] and relation_expected[
                'rel'] == relation_actual['rel']:
                # if entity_expected['name'] == entity_actual['name'] and entity_expected['label'] == entity_actual['label']:
                true_positive.append(relation_actual)
                relations_expected.remove(relation_expected)
                break

    # remove all thos who are not already moved out from entities_actual
    for relation_true in true_positive:
        for relation_actual in relations_actual:
            if lev.distance(relation_true['head']['text'], relation_actual['head']['text']) < 3 and \
                    relation_true['head']['label'] == relation_actual['head']['label'] and lev.distance(
                    relation_true['subj']['text'], relation_actual['subj']['text']) < 3 and relation_true['subj'][
                'label'] == relation_actual['subj']['label'] and relation_true['rel'] == relation_actual['rel']:
                # if entity_true['name'] == entity_actual['name'] and entity_true['label'] == entity_actual['label']:
                relations_actual.remove(relation_actual)

    # zu viel gefunden -> false positive
    false_positive = relations_actual

    # zu wenig gefunden -> false_negative
    false_negative = relations_expected

    if len(true_positive) <= 0:
        return {'precision': 0, 'recall': 0, 'f1': 0}

    precision = len(true_positive) / (len(true_positive) + len(false_positive))
    recall = len(true_positive) / (len(true_positive) + len(false_negative))
    return {'precision': precision, 'recall': recall}


def process_line(row, model, tokenizer):
    expected_relations = parse_relations_from_row(row)
    reference = create_str(expected_relations)
    sample_text = row['text']
    input_ids = tokenizer(sample_text, max_length=1024, truncation=True, padding='max_length', return_tensors='pt').to(
        "cuda")
    summaries = model.generate(input_ids=input_ids['input_ids'], attention_mask=input_ids['attention_mask'],
                               max_length=256)
    model_output = decoded_summaries = \
    [tokenizer.decode(s, skip_special_tokens=False, clean_up_tokenization_spaces=True) for s in summaries][0]
    relations_actual = parse_model_output(model_output)
    return get_metrics(relations_actual, expected_relations)


def get_metrics_from_dataset(dataset, model, tokenizer):
    precision = 0
    recall = 0
    f1 = 0
    counter = 0
    for row in dataset:
        # for row in dataset:
        if len(row['entities']) > 0:
            metric = process_line(row, model, tokenizer)
            precision += metric['precision']
            recall += metric['recall']
            counter += 1
    precision = precision / counter
    recall = recall / counter
    if precision + recall == 0:
        return {'precision': 0, 'recall': 0, 'f1': 0}
    f1 = 2 * ((precision * recall) / (precision + recall))
    return {'precision': precision, 'recall': recall, 'f1': f1}
