import sys
import pickle
import torch
import re
import numpy as np
import pyter 
from transformers import AutoModel, AutoTokenizer
from flask import Flask, render_template, request, redirect, url_for
from vncorenlp import VnCoreNLP
from nltk.translate.bleu_score import sentence_bleu, modified_precision, brevity_penalty
from nltk.translate.nist_score import sentence_nist
from test import translate_doc
from tqdm import tqdm
from underthesea import sent_tokenize
from khmernltk import sentence_tokenize
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states = True)
clf_model = pickle.load(open('logistic_model_more_data.sav', 'rb'))
annotator = VnCoreNLP("VnCoreNLP-master\\VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')


def get_random_name():
    #name = str(datetime.now()).replace(' ','-') + str(random.randint(0,100))
    name = str(np.random.randint(0, 100))
    return name

def tokenize_sentences(doc):
    encoded_sentences = tokenizer(doc, padding=True, truncation=True, return_tensors='pt')
    # encoded_sentences = []
    # for sen in doc: 
    #     encoded_sentences.append(tokenizer(sen, return_tensors='pt')['input_ids'])
    
    return encoded_sentences['input_ids'], encoded_sentences['attention_mask']
    #return encoded_sentences
def read_file(file_path, language='vi', sentence_segment=False, sen=False):
    res = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        res = [re.sub('\n|\u200b', '', line).strip() for line in lines]
        res = [line for line in res if len(line) > 0]
        if sen:
            return res[0] 
        else:
            if sentence_segment: 
                res_segmented = []
                if language == 'vi':
                    for line in res:
                        res_segmented += sent_tokenize(line)
                else: 
                    for line in res:
                        res_segmented += sentence_tokenize(line)
                # punctuation = '[!"#$%&\'()*+,./;<=>?@[\\]^_`{|}~]'
                # res = [re.sub(punctuation, '', line) for line in res]
                return res_segmented
            else:
                return res

def preprocess_text(text, sen=False):
    punctuation = '[!"#$%&\'()*+,./;<=>?@[\\]^_`{|}~]'
    if not sen: 
        res = []
        res = [re.sub(punctuation, '', line) for line in text]
        res = [line.lower() for line in res]
        return res
    else:
        text = re.sub(punctuation, '', text)
        return text.lower()




def segment_sentence(sen):
    word_segmented_sen = annotator.tokenize(sen)
    # word_segmented_sen = annotator.annotate(sen)['sentences'][0]
    res = ''
    # for word in word_segmented_sen:
    #     if word['nerLabel'] not in ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC']:
    #         res += word['form'] + ' '
    for sen in word_segmented_sen:
        for word in sen:
            res += word + ' '
    return res.strip() 
def concatenate_hidden_states(embed):
    return torch.cat(embed[2][-4:], axis = 1)
def cosine_similarity(embed1, embed2):
    return torch.dot(embed1.reshape(768), embed2.reshape(768))/(torch.norm(embed1) * torch.norm(embed2))

def compare_similarity(vi_doc_ids, vi_doc_mask, trans_doc_ids, trans_doc_mask):
    match_table = []
    for i in tqdm(range(vi_doc_ids.shape[0])):
        match_table.append([])
        vi_embed = phobert(vi_doc_ids[i].reshape(1, -1), attention_mask=vi_doc_mask[i].reshape(1, -1))
        for j in tqdm(range(trans_doc_ids.shape[0])):
            trans_embed = phobert(trans_doc_ids[j].reshape(1, -1), attention_mask=trans_doc_mask[j].reshape(1, -1))
            # 4 last layers
            similarity = float(cosine_similarity(torch.mean(concatenate_hidden_states(trans_embed), dim = 1), torch.mean(concatenate_hidden_states(vi_embed), dim = 1)))
            # last layer
            # similarity = cosine_similarity(torch.mean(sen1[0], dim = 1), torch.mean(sen2[0], dim = 1))
            # output of 
            #similarity = cosine_similarity(sen1[1], sen2[1])
            match_table[i].append(similarity)
    return match_table

# def compare_similarity(vi_doc_ids, vi_doc_mask, trans_doc_ids, trans_doc_mask):
    

def get_sentence_similarity(vi_sen, km_sen):
    vi_sen = vi_sen.strip()
    km_sen = km_sen.strip()
    with open('km_sen.txt', 'w', encoding='utf-8') as f:
        f.write(km_sen)
    trans_file = 'trans_' + get_random_name() + '.txt'
    translate_doc('km_sen.txt', trans_file)
    trans_sen = read_file(trans_file, sen=True)
    trans_sen_preprocessed = preprocess_text(trans_sen, sen=True)
    vi_sen_preprocessed = preprocess_text(vi_sen, sen=True)
    vi_sen_segmented = segment_sentence(vi_sen_preprocessed)
    trans_sen_segmented = segment_sentence(trans_sen_preprocessed)
    vi_sen_ids =tokenizer(vi_sen_segmented, return_tensors='pt')['input_ids']
    trans_sen_ids = tokenizer(trans_sen_segmented, return_tensors='pt')['input_ids']
    vi_sen_embed = phobert(vi_sen_ids.reshape(1, -1)) 
    trans_sen_embed = phobert(trans_sen_ids.reshape(1, -1))
    cosine = float(cosine_similarity(torch.mean(concatenate_hidden_states(trans_sen_embed), dim = 1), torch.mean(concatenate_hidden_states(vi_sen_embed), dim = 1)))
    feature_vec = [cosine]
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 1))
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 2))
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 3))
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 4))
    feature_vec.append(get_nist_score(trans_sen_segmented, vi_sen_segmented, 1))
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 2))
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 3))
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 4))
    feature_vec.append(get_bleu_score(trans_sen_segmented, vi_sen_segmented, 5))
    feature_vec.append(get_ter_score(trans_sen_segmented, vi_sen_segmented)) 
    feature_vec = np.array(feature_vec).reshape(1, -1)
    pred_proba = clf_model.predict_proba(feature_vec)[0][1]
    label = clf_model.predict(feature_vec)[0]
    return pred_proba, label, trans_sen
def get_bleu_score(candidate, reference, ngrams):
    candidate = candidate.split(' ')
    reference = [reference.split(' ')]
    return float(modified_precision(reference, candidate, ngrams))
    #return sentence_bleu(reference, candidate, smoothing_function=chencherry.method1)
def get_nist_score(candidate, reference, ngrams):
    candidate = candidate.split(' ')
    reference = [reference.split(' ')]
    return float(sentence_nist(reference, candidate, ngrams))
def get_ter_score(candidate, reference):
    return pyter.ter(candidate.split(), reference.split())


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('text.html')

@app.route('/', methods=['GET', 'POST'])
def process_input():
    if request.method == 'POST':
        vi_sen = request.form['vi']
        km_sen = request.form['km']
        proba, label, trans_sen = get_sentence_similarity(vi_sen, km_sen)
        output_str = ''
        output_str += 'Vietnamese sentence: ' + vi_sen + ' <br/> '
        output_str += 'Khmer sentence: ' + km_sen + ' <br/> '
        output_str += 'Translation: ' + trans_sen + ' <br/> '
        output_str += 'Score: ' + str(proba) + ' <br/> '
        output_str += 'Predicted label: ' + str(label) + ' <br/>'
        return output_str

@app.route('/doc')
def doc():
    return render_template('file.html')

@app.route('/doc', methods=['GET', 'POST'])
def process_file():
    if request.method == 'POST':
        vi_file = request.form['vi']
        km_file = request.form['km']
        num_sens = 1
        if (request.form['num'] != ''):
            num_sens = int(request.form['num'])
        threshold = -1
        if (request.form['thresh'] != ''):
            threshold = float(request.form['thresh'])
        
        km_doc = read_file(km_file, language='km', sentence_segment=True)
        
        # km_doc_preprocessed = preprocess_text(km_doc)
        vi_doc = read_file(vi_file, language='vi', sentence_segment=True)
        vi_doc_preprocessed = preprocess_text(vi_doc)
        trans_file = 'trans_' + get_random_name() + '.txt'
        
        src_file = 'src_' + get_random_name() + '.txt'
        
        with open(src_file, 'w', encoding='utf-8') as f: 
            for line in km_doc:
                f.write(line + '\n')
        translate_doc(src_file, trans_file)
        trans_doc = read_file(trans_file, language='vi')
        
        print(src_file)
        print(trans_file)
        print('Trans doc len: ', len(trans_doc))
        print('Khmer doc len: ', len(km_doc))
        print('Vi doc len: ', len(vi_doc))
        trans_doc_preprocessed = preprocess_text(trans_doc)
        
        vi_doc_segmented = []
        trans_doc_segmented = []
        for sen in vi_doc_preprocessed:
            vi_doc_segmented.append(segment_sentence(sen))
        for sen in trans_doc_preprocessed: 
            trans_doc_segmented.append(segment_sentence(sen))
        print('Trans doc len: ', len(trans_doc_segmented))
        print('Vi doc len: ', len(vi_doc_segmented))
        trans_doc_ids, trans_doc_mask = tokenize_sentences(trans_doc_segmented)
        vi_doc_ids, vi_doc_mask = tokenize_sentences(vi_doc_segmented)
        print('Trans_doc shape: ', trans_doc_ids.shape)
        print('Vi_doc shape: ', vi_doc_ids.shape)
        match_table = compare_similarity(vi_doc_ids, vi_doc_mask, trans_doc_ids, trans_doc_mask)
        print(num_sens)
        print(threshold)
        result = []
        for i in range(len(match_table)):
            for j in range(len(match_table[i])):
                similarity_vec = [match_table[i][j]]
                vi_sen = vi_doc_segmented[i]
                trans_sen = trans_doc_segmented[j]
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 1))
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 2))
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 3))
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 4))
                similarity_vec.append(get_nist_score(trans_sen, vi_sen, 1))
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 2))
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 3))
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 4))
                similarity_vec.append(get_bleu_score(trans_sen, vi_sen, 5))
                similarity_vec.append(get_ter_score(trans_sen, vi_sen))
                result.append(similarity_vec)
        result = np.array(result)
        matches = []
        pred_proba = clf_model.predict_proba(result)
        output_str = ''
        proba_match = []
        
        for i in range(0, pred_proba.shape[0], len(match_table[0])):
            proba_match.append([]) 
            for j in range(len(match_table[0])):
                proba_match[int(i/len(match_table[0]))].append((pred_proba[i + j][1], j))

        if threshold != -1: 
            for i in range(0, pred_proba.shape[0], len(match_table[0])):
                matches.append([])
                for j in range(len(match_table[0])):
                    if pred_proba[i + j][1] >= threshold: 
                        matches[i//len(match_table[0])].append((j, pred_proba[i + j][1]))
            for i, match in enumerate(matches):
                output_str += 'Original ' + str(i) + ': ' + vi_doc[i] + ' <br/> '
                output_str += 'Best matches: <br/> '
                for sen in match:
                    output_str += 'Khmer sentence ' + str(sen[0]) + ': ' + km_doc[sen[0]] + ' <br/> '
                    output_str += 'Translation ' + str(sen[0]) + ': ' + trans_doc[sen[0]] + ' <br/> '
                    output_str += 'Score: ' + str(sen[1]) + ' <br/> '
                output_str += '-'*100 + ' <br/> '
            
        else: 
            for i in range(len(proba_match)):
                proba_match[i] = sorted(proba_match[i])
                output_str += 'Original ' + str(i) + ': '  + vi_doc[i] + ' <br/> ' 
                #print('Original ' + str(i) + ': '  + vi_doc[i])
                output_str += 'Best matches: ' + str(min(num_sens, len(trans_doc))) + ' <br/> '

                #print('Best matches: ' + str(min(num_sens, len(trans_doc))))
                for j in range(min(num_sens, len(trans_doc))): 
                    print(proba_match[i][-1 - j][1])
                    print(km_doc[proba_match[i][-1 - j][1]])
                    output_str += 'Khmer sentence ' + str(proba_match[i][-1 - j][1]) + ': ' + km_doc[proba_match[i][-1 - j][1]] + ' <br/> '
                    output_str += 'Translation ' + str(proba_match[i][-1 - j][1]) + ': ' + trans_doc[proba_match[i][-1 - j][1]] + ' <br/> '
                    #print('Translation ' + str(proba_match[i][-1 - j][1]) + ': ' + trans_doc[proba_match[i][-1 - j][1]])
                    output_str += 'Score: ' + str(proba_match[i][-1 - j][0]) + ' <br/> '
                    #print('Score: ' + str(proba_match[i][-1 - j][0]))
                output_str += '-'*100 + ' <br/> '
        #         #print('-'*100)
                

        

        # for i in range(len(match_table)):
        #     max_score = 0 
        #     max_index = -1 
        #     for j in range(len(match_table[i])):
        #         if match_table[i][j] > max_score:
        #             max_score = match_table[i][j]
        #             max_index = j 
        #     result.append((max_index, max_score))
        
        # for i, sen in enumerate(result): 
        #     max_index = sen[0]
        #     max_score = sen[1]
        #     original = vi_doc[i]
        #     best_match = trans_doc[max_index]
        #     output_str += 'Original ' + str(i) + ': ' + original + ' <br/> ' +  'Translation ' + str(max_index) + ': ' + best_match + ' <br/> ' + 'Score: ' + str(max_score) + ' <br/> ' + '-'*50 + ' <br/>'
        
        return output_str
        
            
if __name__ == '__main__' :
    app.run(debug=True)