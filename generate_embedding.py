from bert_serving.client import BertClient


def generate_embedding(sentences, max_len, verbose=True):
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

    if max_len > bc.server_status["max_seq_len"]:
        raise ValueError(
            "max_len should not be larger than server max_seq_len")
    embeddings = embeddings[:, :max_len, :]
    return embeddings


if __name__ == "__main__":
    embedding = generate_embedding(
        ["grand theft auto", "this is my car"], max_len=10)
    print(embedding.shape)
    print(embedding)
