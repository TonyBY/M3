import subprocess
import os


def index_pyserini(processed_text_dir: str, index_saving_path: str,
                   encoder_name: str = "facebook/dpr-ctx_encoder-multiset-base",
                   batch_size: int = 32, indexer_type: str = "Sparse", num_threads: int = 4,
                   save_raw: bool = False, gpu_idx: str = '1'):

    if indexer_type == "Dense":
        indexing_script = f"CUDA_VISIBLE_DEVICES={gpu_idx} python -m pyserini.encode input --corpus {processed_text_dir} " \
                          f"--fields text output  --embeddings {index_saving_path} --to-faiss encoder --encoder " \
                          f"{encoder_name} --fields text --batch {batch_size}"
    elif indexer_type == "Sparse":
        if save_raw:
            indexing_script = f"python -m pyserini.index.lucene --collection JsonCollection --input " \
                              f"{processed_text_dir} --index {index_saving_path} --generator " \
                              f"DefaultLuceneDocumentGenerator --threads {num_threads} --storePositions " \
                              f"--storeDocvectors --storeRaw"
            print(f"running following command: {indexing_script}")
        else:
            indexing_script = f"python -m pyserini.index.lucene --collection JsonCollection --input " \
                              f"{processed_text_dir} --index {index_saving_path} --generator " \
                              f"DefaultLuceneDocumentGenerator --threads {num_threads} --storePositions -" \
                              f"-storeDocvectors"
    else:
        raise Exception(f"Unknown indexer type: {indexer_type}")
    print("Start...")

    print(f"indexing_script: {indexing_script}")
    exit()
    try:
        result = subprocess.run(indexing_script, shell=True, check=True, stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE, universal_newlines=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    if not os.path.exists(index_saving_path):
        print("There is something wrong when generating index. Check if you have successfully installed pyserini.")
    else:
        print("Done")


if __name__ == "__main__":
    # indexer_type = "Sparse"
    indexer_type = "Dense"
    save_raw = False
    num_threads = 16
    batch_size = 256
    gpu_idx = "7"

    encoder_name = "facebook/dpr-ctx_encoder-multiset-base"
    processed_text_dir = "M3/data/pyserini/processed"

    if indexer_type == "Sparse":
        index_path = "M3/data/pyserini/index/sparse_term_frequency_embedding_noNER_min-2_noHyper_25009475"
    else:
        if 'single' in encoder_name.lower():
            index_path = "M3/data/pyserini/index/zero_shot_single_dpr_noNER"
        elif 'multi' in encoder_name.lower():
            index_path = "M3/data/pyserini/index/zero_shot_multi_dpr_noNER"
    index_pyserini(processed_text_dir, index_path, indexer_type=indexer_type, save_raw=save_raw,
                   num_threads=num_threads, encoder_name=encoder_name, batch_size=batch_size, gpu_idx=gpu_idx)
