import re
import spacy
import os

regex_table = {
    'APT': ['[Aa][Pp][Tt]\s?\d\d\d?', '[Tt][Aa]\d\d\d?'],
    'DOM': [
        r"[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)(?!\w)"],
    'IPV4': ['\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b', '^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'],
    'IPV6': [
        '(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))'],
    'EMAIL': ['[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$',
              '(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])'],
    'URL': [
        '((?:https?|ftp:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))'],
    'VULID': ['CVE-\d{4}-\d{4,7}'],
    'SHA2': [r"[A-Fa-f0-9]{64}"],
    'SHA1': [r"[0-9a-f]{40}"],
    'MD5': [r"[a-f0-9]{32}"],
    'REGISTRYKEY': [
        r"(HKEY_LOCAL_MACHINE\\|HKLM\\|HKEY_CURRENT_USER\\|HKCU\\|HKLM\\|HKEY_USERS\\|HKU\\)([a-zA-Z0-9\s_@\-\^!#.\:\/\$%&+={}\[\]\\*])+"],
}


valid_labels = []


def get_wordlist_for_entity(filename):
    wordlist = []
    with open(filename,encoding="utf8") as fs:
                for line in fs.readlines():
                    if line.strip() not in wordlist:
                        wordlist.append(line.strip())
    return wordlist


def wordlist_to_pattern(wordlist, label):
    valid_labels.append(label)
    pattrns = []
    for word in wordlist:
        pattern = []
        if len(word) > 2:
            for word in word.strip().split(' '):
                pattern.append({'LOWER': word.strip().lower()})
            pattrns.append({"label": label, "pattern": pattern})
    return pattrns



def setup_pipeline():
    base_dir = r"wordlists"
    nlp = spacy.load('en_core_web_trf')
    ner = nlp.get_pipe("ner")

    ruler = nlp.add_pipe("entity_ruler")

    patterns = []
    for label in regex_table:
         for regex in regex_table[label]:
             patterns.append({"label": label, "pattern": [{"TEXT": {"REGEX": regex}}]}, )
             valid_labels.append(label)
    directory = os.fsencode(base_dir)
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        type = filename.split("#")[1]
        patterns = patterns + wordlist_to_pattern(get_wordlist_for_entity(base_dir + "/" + filename),type)
    ruler.add_patterns(patterns)
    return nlp



def replace_special_terms(doc,nlp):
    new_sents = list()
    memroy = list()
    labels = list()
    input_text = doc.text
    for label in regex_table:
        re_expression_list = regex_table[label]
        for expression in re_expression_list:
            for match in re.finditer(expression, doc.text):

                start, end = match.span()
                span = doc.char_span(start, end)
                # This is a Span object or None if match doesn't map to valid token sequence
                if span is not None:
                    if label in valid_labels:
                        print("Found match:", span.text)
                        memroy.append((label, span.text))
                        input_text = input_text.replace(span.text, label)



    step_two_doc =  nlp(input_text)

    for sent in step_two_doc.sents:
        if len(sent.text.strip()) > 2:
            new_sent = sent.text
            labels = list()
            i = 0
            for e in sent.ents:
                if e.label_ in valid_labels:
                    labels.append([e.start_char - sent.start_char, e.end_char - sent.start_char, e.label_])
                    beginning = e.start_char - sent.start_char
                    end = e.end_char - sent.start_char
                    label = e.label_
                    to_replace = sent.text[beginning: end]
                    new_sent = new_sent.replace(to_replace,label)

                    memroy.append((label,to_replace))
            new_sents.append(new_sent)

    step_two_text = " ".join(new_sents)
    return step_two_text, memroy

######resub

def resub_special_terms(memory, text):
    text_array = text.split(" ")
    for mem in memory:

        length = len(text_array)
        i = 0
        while i < length:
            if mem[0] == text_array[i]:
                text_array[i] = mem[1]
            i += 1

    output_text = " ".join(text_array)
    return output_text

if __name__ == "__main__":
    from datasets import load_dataset
    import os

    os.environ["http_proxy"] = 'http://10.158.0.79:80'
    os.environ["https_proxy"] = 'http://10.158.0.79:80'
    # dataset = load_dataset('mrmoor/cyber-threat-intelligence', split='train', use_auth_token=True)
    dataset = load_dataset('mrmoor/cyber-threat-intelligence', split='train')
    print(dataset)



    test_text = """
    Dear xy,
    
    We have had a failed spearphishing attempt targeting our CEO recently with the following details:
    
    Our CEO received an E-mail on 03/02/2021 15:56 containing a personalised message about a report card for their child. The attacker pretended to be working for the school of the CEO’s daughter, sending the mail from a spoofed address (john.doe@luxembourg.edu). John Doe is a teacher of the student. The email was received from throwaway-email-provider.com (137.221.106.104).
    
    The e-mail contained a malicious file (find it attached) that would try to download a secondary payload from https://evilprovider.com/this-is-not-malicious.exe (also attached, resolves to 2607:5300:60:cd52:304b:760d:da7:d5). It looks like the sample is trying to exploit CVE-2015-5465. After a brief triage, the secondary payload has a hardcoded C2 at https://another.evil.provider.com:57666 (118.217.182.36) to which it tries to exfiltrate local credentials. This is how far we have gotten so far. Please be mindful that this is an ongoing investigation, we would like to avoid informing the attacker of the detection and kindly ask you to only use the contained information to protect your constituents.
    
    Best regards,
    
    """
    # replace becuase of split(" ")
    test_text = test_text.replace("\n", " ")

    test_text = test_text.replace("(", " ")
    test_text = test_text.replace(")", " ")

    nlp = setup_pipeline()

    # for i in range(30):
    #     test_text = dataset[i]["text"]
    #     print(test_text)
    #     doc = nlp(test_text)
    #     sub_text, memory = replace_special_terms(doc,nlp)
    #     print("Sub text")
    #     print(sub_text)

    print(test_text)
    doc = nlp(test_text)
    sub_text, memory = replace_special_terms(doc,nlp)
    print("Sub text")
    print(sub_text)
    #resub_text = resub_special_terms(memory,sub_text)
    #print("resub text")
    #print(resub_text)