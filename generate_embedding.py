from bert_serving.client import BertClient
import numpy as np


def generate_embedding(sentences, max_len, include_cls=False, verbose=True):
    try:
        # check if the server is online, to avoid possible infinite waiting
        bc = BertClient(timeout=10000)
        bc.close()
    except TimeoutError:
        raise TimeoutError("the server is not online")

    bc = BertClient()
    if verbose:
        print("Embedding will be generated with the following config:")
        config = {
            "model_dir": bc.server_status["model_dir"],
            "tuned_model_dir": bc.server_status["tuned_model_dir"],
            "max_seq_len": bc.server_status["max_seq_len"]
        }
        print(config)

    embeddings = bc.encode(sentences)
    bc.close()

    if max_len > bc.server_status["max_seq_len"] - 1:
        raise ValueError(
            "max_len should not be larger than server max_seq_len - 1")
    if include_cls:
        embeddings = embeddings[:, :max_len, :]
    else:
        embeddings = embeddings[:, 1: max_len + 1, :]
    return embeddings


def generate_embedding_to_file(sentences, max_len, save_path, include_cls=False, nb_per_job=512):
    embeddings = np.array()
    total_jobs = len(sentences) / nb_per_job - 1

    for idx in range(total_jobs):
        job_sentences = sentences[idx * nb_per_job: (idx + 1) * nb_per_job]
        np.append(embeddings, generate_embedding(
            job_sentences, max_len, include_cls, verbose=False))
    if len(sentences) % nb_per_job != 0:
        job_sentences = sentences[len(
            sentences) - len(sentences) % nb_per_job:]
        np.append(embeddings, generate_embedding(
            job_sentences, max_len, include_cls, verbose=False))

    np.save(save_path, embeddings)
    return


if __name__ == "__main__":
    embedding = generate_embedding(
        ["grand theft auto", "this is my car"], max_len=10)
    print(embedding.shape)
    print(embedding)
