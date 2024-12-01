import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))

from src.utils.data_utils import process_evid, load_pickle, dump_pickle, make_directory, get_file_dir, save_jsonl

from typing import List
from tqdm import tqdm


def get_proceesed_wikiSentCorpus(wiki_line_dict: dict, 
                                 sep="|#SEP#|", 
                                 worker_id: int=0, 
                                 ADD_TITLE: bool=None, 
                                 CLEAN: bool=True, 
                                 ADD_NER: bool=False,
                                 KEEP_HYPERLINK: bool=False) -> List[dict]:
    """
    Construct a pyserini style corpus using wiki sentence dictionary.
    params:
        wiki_line_dict: a map between global sentence ids to sentence line without hyperlinks.
        worker_id: Process id for mutilprocess parallel computing.
        ADD_TITLE: Whether to add doc id in front of the sentence text.
        CLEAN: Whether to process the sepcial tokens in the fever data.
        ADD_NER: Whether to extrac named entities from the sentence text.
    output: 
        data_list: A list of processed sentence of format: {"id": str, "context": str, "NER": dict}
    """
    
    data_list = []
    for sentence_id in tqdm(wiki_line_dict.keys(), 
                  total=len(wiki_line_dict.keys()),
                  desc=f">>Worker {worker_id} is processing Wiki sentences..."):
        
        sentence_line = wiki_line_dict[sentence_id]

        title = process_evid(sentence_id.split(sep)[0])
        split = sentence_line.split('\t')
        if split[0].isdigit():
            split = split[1:]
                    
        if KEEP_HYPERLINK:
            sentence_line = "\t".join([x for x in split])
        else:
            sentence_line = split[0]

        contents = process_evid(sentence_line) if CLEAN else sentence_line
        
        contents = " . ".join([title, contents]) if ADD_TITLE else contents
        
        data_list.append({'id': sentence_id, 'contents': contents})

    return data_list

if __name__ == "__main__":
    wiki_line_dict_path = "M3/data/pyserini/fever_wiki_line_dict_min-2_wHyper_25009475.pkl"
    output_processed_data_path = "M3/data/pyserini/processed_min-2_wHyper_25009475/processed.pkl"

    KEEP_HYPERLINK=True

    print(f"Loading wiki_line_dict from: {wiki_line_dict_path}")
    wiki_line_dict = load_pickle(wiki_line_dict_path)

    processed_data = get_proceesed_wikiSentCorpus(wiki_line_dict, 
                                                    sep="|#SEP#|", 
                                                    ADD_TITLE=True, 
                                                    CLEAN=True, 
                                                    ADD_NER=False,
                                                    KEEP_HYPERLINK=KEEP_HYPERLINK)

    print(f"Saving processed_data to: {output_processed_data_path}")
    make_directory(get_file_dir(output_processed_data_path))
    dump_pickle(processed_data, output_processed_data_path)
    save_jsonl(processed_data, output_processed_data_path.replace('.pkl', '.jsonl'))
