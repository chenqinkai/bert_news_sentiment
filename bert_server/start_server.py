import sys

from bert_serving.server import BertServer
from bert_serving.server.helper import get_run_args


if __name__ == '__main__':
    args = get_run_args()
    server = BertServer(args)
    server.start()
    server.join()

# to start:
# python start_server.py -model_dir D:/data/nlp/bert/uncased_L-12_H-768_A-12 -num_worker=1
