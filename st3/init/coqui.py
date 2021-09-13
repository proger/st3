from dataclasses import dataclass
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import torch


@dataclass
class Coqui:
    stt_pb: str = 'coqui-stt-0.9.3-models.pb'
    sample_rate: int = 16000
    feature_win_len: int = 32
    feature_win_step: int = 20
    audio_window_samples: int = int(sample_rate * feature_win_len / 1000)
    audio_step_samples: int = int(sample_rate * feature_win_step / 1000)
    magnitude_squared: bool = True
    n_input: int = 26
    n_left_context: int = 9
    n_right_context: int = 9

    blank_index = 28
    alphabet = "abcdefghijklmnopqrstuvwxyz 'ε"


def load_pb(path_to_pb=Coqui.stt_pb):
    with gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def, name='')
    return graph_def


def get_tensor_value(node) -> torch.Tensor:
    return torch.tensor(tensor_util.MakeNdarray(node.attr['value'].tensor))


def from_linear(nodes, name='layer_1'):
    return {f'{name}.weight': get_tensor_value(nodes[f'{name}/weights']).T,
            f'{name}.bias': get_tensor_value(nodes[f'{name}/bias'])}


def from_lstm_block_cell(nodes, name='cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell', output_name='lstm'):
    # https://www.tensorflow.org/api_docs/python/tf/raw_ops/LSTMBlockCell

    weight_ih, weight_hh = get_tensor_value(nodes[f'{name}/kernel']).T.chunk(2, axis=1)
    bias = get_tensor_value(nodes[f'{name}/bias']).T

    # TF 1.x igfo -> ifgo reordering hint: https://discuss.tensorflow.org/t/tensorflow-keras-lstm-vs-tf-contrib-rnn-lstmblockfusedcell/1107
    i,g,f,o = weight_ih.chunk(4, axis=0)
    weight_ih = torch.cat([i,f,g,o])
    i,g,f,o = weight_hh.chunk(4, axis=0)
    weight_hh = torch.cat([i,f,g,o])
    i,g,f,o = bias.chunk(4)
    bias = torch.cat([i,f,g,o])

    return {f'{output_name}.weight_ih': weight_ih,
            f'{output_name}.weight_hh': weight_hh,
            f'{output_name}.bias_ih': torch.zeros_like(bias),
            f'{output_name}.bias_hh': bias}


def state_dict(path_to_pb=Coqui.stt_pb):
    graph_def = load_pb(path_to_pb)
    nodes = {n.name: n for n in graph_def.node}

    return (from_linear(nodes, 'layer_1')
            | from_linear(nodes, 'layer_2')
            | from_linear(nodes, 'layer_3')
            | from_lstm_block_cell(nodes, 'cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell')
            | from_linear(nodes, 'layer_5')
            | from_linear(nodes, 'layer_6'))


def mfccs(input_samples):
    with tf.compat.v1.Session() as sess:
        mfccs = sess.run(
            fetches=(sess.graph.get_tensor_by_name('mfccs:0')),
            feed_dict={sess.graph.get_tensor_by_name('input_samples:0'): input_samples})
    return mfccs


if __name__ == '__main__':
    graph_def = load_pb(Coqui.stt_pb)

    #mfccs()