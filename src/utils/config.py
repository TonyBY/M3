from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import argparse

from src.utils.args import ArgumentGroup

parser = argparse.ArgumentParser(__doc__)

common_g = ArgumentGroup(parser, "common", "common options.")
common_g.add_arg("num_workers",                      int,           24,
                "Number of workers of DataLoader.")
common_g.add_arg("do_train",                         bool,          True,  
                "Whether to perform training.")
common_g.add_arg("do_predict",                       bool,          False,
                "Whether to run eval on the dev set.")
common_g.add_arg("train_mode",                       str,           "JOINT",
                "Training mode, choices from ('RETRIEVE_ONLY', 'NLI_ONLY', 'JOINT')")
common_g.add_arg("use_cuda",                         bool,          True,  
                "If set, use GPU for training.")
common_g.add_arg("local_rank",                       int,           -1,
                "local_rank for distributed training on gpus")
common_g.add_arg("no_cuda",                          bool,          False,
                "Whether not to use CUDA when available")
common_g.add_arg("debug",                            bool,          False,
                "Controls log level and other logics for debugging.")

model_g = ArgumentGroup(parser, "model", "model configuration and paths.")
model_g.add_arg("single_encoder",                    bool,          False,
                "If true, single-encoder retriever will be selected, otherwise, bi-encoder retriever.")
model_g.add_arg("query_encoder_name",                str,           "facebook/dpr-question_encoder-multiset-base",
                "Name/dir of query encoder model.")
model_g.add_arg("ctx_encoder_name",                  str,           "facebook/dpr-ctx_encoder-multiset-base",
                "Name/dir of context encoder model.")
model_g.add_arg("init_checkpoint",                   bool,          False,
                "Whether to init model with checkpoint.")
model_g.add_arg("checkpoint_path",                   str,           "",
                "Path to a pretrained model.")
model_g.add_arg("freeze_ctx_encoder",                bool,          False,
                "Whether to freeze parameters in the layers of the context encoder in the bi-model contrastive learning system. This is usefuly when training a multi-hop dense retriever.")
model_g.add_arg("continue_training",                 bool,          True,
                "Whether to inherent previous best results when initiating from pretrained models.")
model_g.add_arg("shared_encoder",                    bool,          False,
                "If true, only context encoder will be used.")
model_g.add_arg("output_dir",                        str,           "M3/data/checkpoints",  
                "Directory to save checkpoints.")

data_g = ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
data_g.add_arg("train_file",                         str,           "M3/data/dpr/fever_bm25/train/singleHopOnly/dpr_train_singleHop_100.jsonl", 
                "Path to the training data.")
data_g.add_arg("development_file",                   str,           "M3/data/dpr/fever_bm25/dev/singleHopOnly/dpr_dev_singleHop_100.jsonl", 
                "M3/data/dpr/fever_bm25/dev/singleHopOnly/dpr_dev_singleHop_100.jsonl")
data_g.add_arg("max_seq_len",                        int,           512,   
                "Number of words of the longest seqence.")
data_g.add_arg("num_hard_negs",                      int,           5,    
                "Number of hard negatives per example.")


train_g = ArgumentGroup(parser, "training", "training options.")
train_g.add_arg('prefix',                            str,           "eval",
                "Run name of experiment.")
train_g.add_arg("num_train_epochs",                  int,           3,       
                "Number of epoches for fine-tuning.")
train_g.add_arg("learning_rate",                     float,         1e-5,    
                "Learning rate used to train with warmup.")
train_g.add_arg("use_adam",                          bool,          True,
                "Whether to use adam optimizer.")
train_g.add_arg("adam_epsilon",                      float,         1e-8,
                "Epsilon for Adam optimizer.")
train_g.add_arg("lr_scheduler",                      str,           "linear_warmup_decay",
                "scheduler of learning rate.", choices=['linear_warmup_decay', 'noam_decay'])
train_g.add_arg("weight_decay",                      float,         0.01,    
                "Weight decay rate for L2 regularizer.")
train_g.add_arg("warmup_ratio",                      float,         0.1,
                "Proportion of training steps to perform linear learning rate warmup for.")
train_g.add_arg("save_checkpoints_steps",            int,           10000,   
                "The steps interval to save checkpoints.")
train_g.add_arg("eval_period",                       int,           2500,   
                "The steps interval to evaluate model performance.")
train_g.add_arg("train_batch_size",                  int,           1,    
                "Total examples' number in batch for training.")
train_g.add_arg("extra_batch_size",                  int,           1,    
                "Total examples' number in batch for extra training dataset.")
train_g.add_arg("predict_batch_size",                int,           1,    
                "Total examples' number in batch for training.")
train_g.add_arg("accumulate_gradients",              int,           1,
                "Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
train_g.add_arg('gradient_accumulation_steps',       int,           1,
                "Number of updates steps to accumualte before performing a backward/update pass.")
train_g.add_arg('seed',                              int,           3,
                "Random seed for initialization")
train_g.add_arg("max_grad_norm",                     float,         2, 
                "Max gradient norm.")
train_g.add_arg("train_nli_only",                    bool,          False,
                "Train with only nli task.")
train_g.add_arg("positive_weight",                   float,         None,
                "Weight of positive examples when calculating loss.")
train_g.add_arg("use_extra_retrieve_only_train_dataset", bool,      False,
                "Whether to use an extra dataset to train the model with RETRIEVE_ONLY objective.")
train_g.add_arg("extra_retrieve_only_train_file",    str,           "",
                "Path to the extra training set.")
train_g.add_arg("weighted_sampling",                 bool,          False,
                "Whether to do over sampling for rare classes.")
train_g.add_arg("use_weighted_ce_loss",              bool,          False,
                "Whether to use weighted cross entropy loss for NLI task.")
train_g.add_arg("mix_interval",                      float ,          2.0,
                "Train with extra retrieval only data very n epochs.")
train_g.add_arg("retrieval_to_nli_weight",           float ,        1,
                "Weight of multi-task losses, i.e., retrieval loss and nli loss.")
train_g.add_arg("target_nli_distribution",           int,           [1, 1, 1],
                "Target distribution of NLI classes of (supporting : refuting : not enough info) after applying weighted_sampling or weighted loss.", 
                nargs='+')
train_g.add_arg("use_joint_best_score",              bool,          False,
                "Whether to use best joint score to decide when to cache model.")
train_g.add_arg("use_nli_best_score",                bool,          False,
                "Whether to use best nli score to decide when to cache model.")
train_g.add_arg("sim_type",                          str,           'dot',
                "Whether to use dot product or consine similarity to evalate coherence between two sentence embeddings. Option: 'dot' or 'cosine'. ")
train_g.add_arg("temp",                              float,          1.0,
                "Temperatur, a hyper-parameter that used to scale similarity score between two sentence embeddings, e.g., score = score/temp.")
train_g.add_arg("use_ce_loss",                       bool,           False,
                "Whether to use cross-entropy loss, if not, negative log likelihood loss would be used.")
train_g.add_arg("senteval_data_path",                str,            "",
                "Path to the SentEval dataset for universal sentence embedding evaluation.")
train_g.add_arg("caching_metric",                    str,            "",
                "Metric name that used to caching the best model during training. (stsb, stskr, joint_sts)")
train_g.add_arg("extra_train_mode",                  str,            "",
                "Train mode for the extra task for mix training: (RETRIEVE_ONLY, JOINT, NLI_ONLY)")

index_g = ArgumentGroup(parser, "index", "index options.")
index_g.add_arg('sentence_corpus_path',              str,           "",
                "Sentence corpus '.jsonl' file path.")
index_g.add_arg('index_dir',                         str,           "",
                "Directory to save generated index and relevant files.")
index_g.add_arg('encoding_batch_size',               int,           128,
                "Batch size for encoding sentences.")
index_g.add_arg('shard_id',                          int,           0,
                "Chunk/shard id for parrallel encoding.")
index_g.add_arg('indexing_batch_size',               int,           50000,
                "Buffer size for generating index.")
index_g.add_arg('index_type',                        str,           'HNSWFlat',
                "Faisee index type, choices from ('IndexFlatIP', 'IVF16384_HNSW32', 'HNSWFlat')")
index_g.add_arg('sentence_embedding_path',           str,           '',
                "Path to the file of generated sentence embeddings.")
index_g.add_arg('encode_only',                       bool,          False,
                "Whether to only generate sentence embeddings without doing indexing.")
index_g.add_arg('index_dim',                         int,           768,
                "Hidden size. 768 for roberta-base and bert-base, 1024 for roberta-large and bert-large.")

search_g = ArgumentGroup(parser, "search", "search options.")
search_g.add_arg('index_in_gpu',                     bool,          True,
                 "Whether to search with index in gpus.")
search_g.add_arg('topk',                             int,           100,
                 "Whether to search with index in gpus.")
search_g.add_arg('query_batch_size',                 int,           64,
                "Query batch size for searching.")
search_g.add_arg('data_path',                        str,           '',
                "Query data path."),
search_g.add_arg('output_path',                      str,           '',
                "Searching results output path."),
search_g.add_arg('multi_hop_sparse_retrieval_dir',   str,           '',
                "Output directory that saves retrieval results."),
search_g.add_arg('multi_hop_sparse_retrieval',       bool,          False,
                "Wheterh to do multi_hop_sparse_retrieval."),
search_g.add_arg('max_num_process',                  int,           1,
                "Number of process for python multiprocessing.")
search_g.add_arg("wiki_line_dict_pkl_path",          str,           '',
                "Path to a pickle file of a map between sentence id to sentence text."),
search_g.add_arg("wiki_extra_line_dict_pkl_path",    str,           '',
                "Path to a pickle file of a map between sentence id to hyperlinks(doc titles) within the sentence."),     
search_g.add_arg("docid_to_sent_idx_dict_pkl_path",  str,           '',
                "Path to a pickle file of a map between the doc ids to the sentence index of the global sentence id list."), 
search_g.add_arg('cache_searching_result',           bool,          True,
                 "Cache intermediate reults every n steps while doing searching.")

reranker_g = ArgumentGroup(parser, "rerank", "sentence rerank options.")
reranker_g.add_arg('model_type',                     str,           '',
                 "Model type for doing reranking, e.g., roberta-large.")
reranker_g.add_arg('model_path',                     str,           '',
                 "Path to the pretrained model.")  
reranker_g.add_arg('num_labels',                     int,           3,
                 "Number of labels the reranker is trained with, either 2(e.g., relevant/not relevant) or 3(e.g., supporting, refuting, not enough info).")
reranker_g.add_arg('first_hop_search_results_path',  str,           '',
                 "Path to the frist hop search results by DPR-singleHop.")                
reranker_g.add_arg('reranking_dir',                  str,           '',
                 "Reranking output directory.")
reranker_g.add_arg('fist_hop_topk',                  int,           128,
                 "Top-k sentences from the first hop search results as input to the reranker.")
reranker_g.add_arg('rerank_topk',                    int,           5,
                 "Number of sentences returned by the reranker.")
reranker_g.add_arg('retrank_batch_size',             int,           128,
                 "Batch size for reranking.")
reranker_g.add_arg('reretrieval',                    bool,          False,
                 "If this round of re-ranking is for reretrieval, i.e., evidence retrieval with evidence retrieved from the first round of retreival.")
reranker_g.add_arg('num_neg_samples',                int,           100,
                 "Number of negative samples per claim when constructing dataset to train sentence reranker.")
reranker_g.add_arg('use_mnli_labels',                bool,          True,
                 "Whether to use three labels (otherwise one lable) to train the sentence reranker.")
reranker_g.add_arg('reranking_train_file',           str,           "",
                 "Training set file path.")
reranker_g.add_arg('reranking_dev_file',             str,           "",
                 "Developement set file path.")
reranker_g.add_arg('add_multi_hop_egs',                bool,          True,
                 "Whether to include multi-hop examples in the datasets.")
reranker_g.add_arg('add_single_hop_egs',                bool,          True,
                 "Whether to include single-hop examples in the datasets.")

ir_eval_g = ArgumentGroup(parser, "ir_eval", "retrieval evaluation options.")
ir_eval_g.add_arg('retrieval_result_path',           str,           '',
                  "Path to pickl file of the retrieval results that to be evaluated.")
ir_eval_g.add_arg('submission_dir',           str,           '',
                  "Path to pickl file of the final claim classification results that to be evaluated.")

se_eval_g = ArgumentGroup(parser, "se_eval", "sentence encoding evaluation options.")
se_eval_g.add_arg('SE_dir',                          str,           '',
                  "Output path of saving evalution results.")
se_eval_g.add_arg('sentence_encoding_eval_data_path',str,           '',
                  "Dirctory contains data for sentence encoding evalution.")
se_eval_g.add_arg('encoder_type',                    str,           'query',
                  "Encoder type of the bi-encoder model. It's either 'query' or 'ctx'.")
se_eval_g.add_arg('run_name',                        str,           '',
                  "Experiment run name for log printing.")

se_eval_g = ArgumentGroup(parser, "esc_eval", "evidence sufficiency checking evaluation options.")
se_eval_g.add_arg('sufficiency_checking_dir',        str,           '',
                  "Output path of saving evalution results of evidence sufficiency checking.")
se_eval_g.add_arg('SUF_CHECK',                       bool,          False,
                  "Whther to do evidence sufficiency checking. If not, the model will be trained for sentence reranking.")
se_eval_g.add_arg('check_topk',                      int,           5,
                  "Top-k sentence reranker's predictions to check for each claim.")
se_eval_g.add_arg('sufficiency_checking_batch_size', int,           5,
                  "Batch size for evidence checking evaluation. Not benifitial when larger than args.check_topk")
se_eval_g.add_arg('sr_results_path',                 str,           '',
                  "Sentence reranker output path.")

mdr_eval_g = ArgumentGroup(parser, "multihop_doc_retrieval", "Retrieve and evaluate document-level multihop evidence.")
mdr_eval_g.add_arg('multihop_doc_retrieval_dir',     str,           '',
                  "Output directory of saving evalution results of doc-level multi-hop retrieval.")
mdr_eval_g.add_arg('sufficiency_checking_results_path', str,        '',
                  "Output file path of the sufficiency checking module, which will be used as input of this module.")
mdr_eval_g.add_arg('similarity_func_name',           str,           'jaccard',
                  "Similarity function used for document-level retrieval. ('jaccard', 'cosine', 'containment')")
mdr_eval_g.add_arg('similarity_threshold',           float,         0.5,
                  "Similarity thrshold for doc-level multihop evidence retrieval.")
mdr_eval_g.add_arg('mdr_topk',                       int,           2,
                  "topk docs to retrieve for each hyperlink. ")
mdr_eval_g.add_arg('maxhop',                         int,           2,
                  "Only do multihop retrieval for those multihop examples whose minimum evidence hop is less than or equal to 'maxhop'. ")
mdr_eval_g.add_arg('multi_hop_dense_retrieval',      bool,          False,
                  "Whether to do multi-hop dense retrieval.")
mdr_eval_g.add_arg('srr_th',                         float,         1.0,
                  "Sentnence reranking score threshold, used for multi_hop evidence selecting.")
mdr_eval_g.add_arg('sf_th',                          float,         1.0,
                  "Sufficiency checking score threshold, used for multi_hop evidence selecting.")

cc_g = ArgumentGroup(parser, "claim_classification", "claim classification options.")
cc_g.add_arg('claim_classification_dir',             str,           '',
                 "Claim classification output directory.")
cc_g.add_arg('final_retrieval_results_path',         str,           '',
                 "Path to the final search results.")
cc_g.add_arg('retrieved_evidence_feild',             str,           '',
                 "a key name of the input_data example, which is used to get the top retrieved evidence.")
cc_g.add_arg('claim_classificaton_train_file',       str,           '',
                 "Training data path of claim classification.")
cc_g.add_arg('claim_classificaton_dev_file',         str,           '',
                 "Dev data path of claim classification.")
cc_g.add_arg('shuffle_evidence_p',                   float,         0.0,
                 "Chance of shuffuling the evidence in the dataload when training a claim classifier.")
cc_g.add_arg('cc_batch_size',                        int,           1,
                 "Training batch size of the claim classifier.")
cc_g.add_arg('label_smoothing',                      float,         0.0,
                 "Label soothing rate of pytorch CrossEntropy loss.")
cc_g.add_arg('val_check_interval',                   float,         0.1,
                 "validation interval of portition of training steps.")
cc_g.add_arg('prediction_type',                      str,           'mixed',
                 "Evidence formulation type when doing claim classification evaluation, options: ('singleton', 'concat', ''mixed)")

msrr_g = ArgumentGroup(parser, "multihop_sentence_reranking_and_result_merging.", "Multi-hop reranking options.")
msrr_g.add_arg('merged_reranked_results_dir',         str,           '',
               "Output directory of merged results.")
msrr_g.add_arg('msrr_result_path',                    str,           '',
               "Input data file, the output of multihop sentence reranker.")
msrr_g.add_arg('msrr_merge_metric',                   str,           '',
               "Metric name the used to calculate scores to filter irrelevant multi-hop retrieval paths. Choices: ['path', 'sum']")
msrr_g.add_arg('mhth',                                float,          0.9,
               "Threshold of the irrelevant multi-hop retrieval paths filter.")
msrr_g.add_arg('alpha',                               float,          1.0,
               "Weight of score when merging two retriever's results.")
msrr_g.add_arg('normalization',                       bool,           True,
               "Whether to normalize ranking scores before merging two retrievers' results.")
msrr_g.add_arg('weight_on_dense',                     bool,           False,
               "Whether to apply the alpha/weight over the dense_retriever/singlehop_reranker or over the sparse_retriever/multihop_reranker.")
msrr_g.add_arg('singleHopNumbers',                    int,            0,
               "Number of single-hop examples will be included when doing multi-hop retrieval/evaluation/dataset construction for joint reranking.")
msrr_g.add_arg('multiHopNumbers',                     int,            0,
               "Number of multi-hop examples will be included when constructing dataset for joint reranking.")
msrr_g.add_arg('joint_reranking',                     bool,           False,
               "Whether to constructe dataset for a joint ranking task.")
msrr_g.add_arg('save_evi_path',                       bool,           True,
               "Whether to save previous evidence when doing multihop sentence reranking.")
msrr_g.add_arg('concat_claim',                        bool,           True,
               "Whether to concate claim before an first-hop evidence as a new claim when doing sencond-hop reranking, otherwise, only the first-hop evidence will be used as the claim.")
msrr_g.add_arg('naive_merge',                         bool,           False,
               "Whether to do naive merging when combining sing-hop and multi-hop sentence reranking results. Otherwise, complex joint reranking will be applied.")
msrr_g.add_arg('naive_merge_discount_factor',         float,          0.95,
               "Discount factor applied over the multi-hop evidence when combining single-hop and multi-hop sentence reranking results naively.")
msrr_g.add_arg('tune_params',                         bool,           True,
               "Whether to tune parames or use given parameters when doing complext joint reranking.")

xgbc_g = ArgumentGroup(parser, "xgboost_classifier", "XGBoost Classification options.")
xgbc_g.add_arg('xgbc_dir',                            str,           '',
               "Output directory of the best xgboost classifier model.")
xgbc_g.add_arg('claim_scores_path',                   str,           '',
               "Classification scores generated by the previous classifier. It's the input of xgbc.")
xgbc_g.add_arg('xgbc_model_path',                     str,           '',
               "pretrained xgboost classifier model path.")


hybrid_g = ArgumentGroup(parser, "hybrid_ranking", "Hybrid ranking options.")
hybrid_g.add_arg('hybric_search_dir',                 str,           '',
               "Output directory of the hybrid ranking results.")
hybrid_g.add_arg('dense_results_path',                str,           '',
               "Path to the retrieval results of the dense retriever.")
hybrid_g.add_arg('sparse_results_path',               str,           '',
               "Path to the retrieval results of the sparse retriever.")
