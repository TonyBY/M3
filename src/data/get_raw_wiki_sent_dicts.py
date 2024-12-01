import os
import sys
pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from src.utils.data_utils import dump_pickle, read_jsonl, read_json
import unicodedata

from typing import Union

from tqdm import tqdm
import glob


def get_all_unique_sentIds_from_fever_data(train_data_path: str, dev_data_path: str) -> set:
    train_data = read_jsonl(train_data_path)
    dev_data = read_json(dev_data_path)
    
    data_all = train_data + dev_data
    
    sentIds = set()
    for item in data_all:
        evidence_groups = item['evidence']
        for eg in evidence_groups:
            e=eg[0]
            doc_id = e[2]
            line_num = str(e[3])
            try:
                sentId = unicodedata.normalize('NFC', doc_id) + "|#SEP#|" + str(line_num)
                sentIds.add(sentId)
            except:
                continue
    return sentIds

def buld_sentenceIdx_to_sentenceLine_docLink_dicts(
                                                    input_dir: str, 
                                                    min_token: int=5, 
                                                    keep_hyper_link: bool=True,
                                                ) -> Union[dict, dict]:
    """
    paras: 
        input_dir: directory of wiki_pages.
        min_token: minimum length of a line(sentence) to be included.
    """
    wiki_line_dict = {}
    wiki_extra_line_dict = {}
    
    doc_cnt = 0
    valid_doc_cnt = 0
    for wikipage in tqdm(glob.glob(input_dir + '/*.jsonl'), 
                         total=len(glob.glob(input_dir + '/*.jsonl')), 
                         desc="Processing Wiki pages for pyserini..."):
        print(f"wikipage: {wikipage}")
        docs = read_jsonl(wikipage) # docs: [{'id': , 'text': , 'lines': }]
        cnt = 0
        for doc in docs:
            doc_cnt += 1
            if doc['id'] == "" or doc['text'] == '' or doc['lines'] == '':
                continue
            valid_doc_cnt += 1
            doc_id = doc['id']    
            lines = doc['lines']
            
            for page_line in lines.split("\n"):
                split = page_line.split("\t")
                if not split[0].isdigit():
                    continue
                line_num, line = [int(split[0]), split[1]]
                if len(line.split()) < min_token:
                    continue
                line_extra = "\t".join([x for x in split[2:]])
                
                global_sent_id = doc_id + "|#SEP#|" + str(line_num)
                wiki_line_dict[global_sent_id] = "\t".join([x for x in split[1:]]) if keep_hyper_link else line
                wiki_extra_line_dict[global_sent_id] = line_extra                             

    return wiki_line_dict, wiki_extra_line_dict

if __name__=='__main__':
    output_dir = "M3/data/pyserini/wiki_sent_dict/"

    wiki_page_dir = "M3/data/FEVER_1/wiki-pages"
    min_token = 2
    keep_hyper_link = False

    wiki_line_dict, wiki_extra_line_dict = buld_sentenceIdx_to_sentenceLine_docLink_dicts(wiki_page_dir, 
                                                                                      min_token=min_token,
                                                                                      keep_hyper_link=keep_hyper_link)

    wiki_line_dict_pkl_path = os.path.join(output_dir, f'fever_wiki_line_dict_{len(wiki_line_dict.keys())}.pkl')
    wiki_extra_line_dict_pkl_path = os.path.join(output_dir, f'fever_wiki_extra_line_dict_{len(wiki_extra_line_dict.keys())}.pkl')

    print(f"Saving wiki_line_dict to: {wiki_line_dict_pkl_path}")
    dump_pickle(wiki_line_dict, wiki_line_dict_pkl_path)
    print(f"Saving wiki_extra_line_dict to: {wiki_extra_line_dict_pkl_path}")
    dump_pickle(wiki_extra_line_dict, wiki_extra_line_dict_pkl_path)
