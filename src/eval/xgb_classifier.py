"""
Script for training gradient boosting classifier via XGBoost.
Expects that previous pipeline steps have been ran.
"""
import os
import sys

pwd = os.getcwd()
sys.path.append('/'.join(pwd.split('/')[:-3]))
sys.path.append('/'.join(pwd.split('/')[:-2]))
sys.path.append('/'.join(pwd.split('/')[:-1]))
os.environ['FLAGS_eager_delete_tensor_gb'] = '0'  # enable gc

import logging
import numpy as np
from typing import List
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

import transformers
transformers.logging.set_verbosity_error()

from src.utils.data_utils import make_directory, load_pickle, dump_pickle, get_label_num
from src.utils.args import prepare_logger
from src.utils.config import parser

args = parser.parse_args()
logger = logging.getLogger()

# FEVER's dev set is large enough for support fairly large XGBC models.
XGBC_PARAMS = {
    "max_depth": [1, 2],
    "lambda": [1],
    "alpha": [0],
    "n_estimators": [65, 70, 75],
    "learning_rate": [0.1, 0.2],
    "use_label_encoder": [False],
    "eval_metric": ["mlogloss"],
}

def reorder_by_score_filter_add_softmax(scores, dev_data, min_ret_score=0.0):
    """
    Reorders the score matrix in descending order. If `min_ret_score` set,
    filter out any results with ret_score `min_ret_score`.
    """
    new_scores = []
    new_labels = []
    for sample, item in zip(scores, dev_data):
        matrix_shape = sample.shape[0]
        concat_score = sample[-1].reshape((1, -1))
        sample = sample[:-1]
        label = get_label_num(item['label']) if 'label' in item else 0
        idx_order = np.argsort(sample[:, -1])
        sample = sample[idx_order]
        sample = sample[sample[:, -1] >= min_ret_score]
        sample = np.concatenate([sample, concat_score])
        nan_padded = np.ones((matrix_shape, sample.shape[1]))
        nan_padded[nan_padded == 1] = np.nan
        if sample.shape[0] >= 1:
            nan_padded[-sample.shape[0] :] = sample
        new_scores.append(nan_padded)
        new_labels.append(label)
    return np.array(new_scores), new_labels


def main(args, xgbc_params=XGBC_PARAMS):
    """
    Function for training XGBoost classifier. Use dev results to train
    an aggregation model. The dev set is used, since it isn't finetuned
    on, thus the scores are not biases.

    Uses 4 fold cross validation. Best parameter set is chosen and the
    final model is retrained with all available data.
    """
    dev_data = load_pickle(args.final_retrieval_results_path)
    dev_labels = [get_label_num(item['label']) for item in dev_data]

    dev_claim_scores = np.load(args.claim_scores_path)

    dev_claim_scores, dev_labels = reorder_by_score_filter_add_softmax(
        dev_claim_scores, dev_data
    )

    dev_claim_scores = dev_claim_scores.reshape((len(dev_claim_scores), -1))
    xgbc = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
    logger.info(f"xgbc: {xgbc}")
    logger.info(f"xgbc_params: {xgbc_params}")
    gridcv = GridSearchCV(xgbc, 
                          xgbc_params, 
                          scoring="accuracy",
                          n_jobs=10,
                          cv=4,
                          verbose=10,
                          refit=True,
                          return_train_score=True,
                        )
    logger.info("Start fitting...")
    gridcv.fit(dev_claim_scores, dev_labels)
    logger.info("Done.")
    logger.info(f"gridcv.cv_results_.keys(): {gridcv.cv_results_.keys()}")
    logger.info(f"gridcv.best_params_: {gridcv.best_params_}")
    logger.info( f"Best Mean Train Accuracy Across Folds: {gridcv.cv_results_['mean_train_score'].mean()}")
    logger.info(f"Best Mean Test Accuracy Across Folds: {gridcv.cv_results_['mean_test_score'].mean()}")
    dump_pickle(gridcv.best_estimator_, os.path.join(args.xgbc_dir, "best_xgbc.pkl"))


if __name__ == "__main__":
    make_directory(args.xgbc_dir)
    prepare_logger(logger, debug=args.debug, save_to_file=os.path.join(args.xgbc_dir, f"xgbc_train.log"))
    logger.info(args)
    main(args)
    logger.info("All Done.")
