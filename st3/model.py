import numpy as np
import torch
import torch.nn as nn


class ClippedReLU(nn.Module):
    def forward(self, x):
        return torch.nn.functional.threshold(x, 0, 20)


class DeepSpeech(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear((9+1+9)*26, 2048)
        self.relu_1 = ClippedReLU()
        self.layer_2 = nn.Linear(2048, 2048)
        self.relu_2 = ClippedReLU()
        self.layer_3 = nn.Linear(2048, 2048)
        self.relu_3 = ClippedReLU()
        self.lstm = nn.LSTM(2048, 2048, batch_first=True)
        self.layer_5 = nn.Linear(2048, 2048)
        self.relu_5 = ClippedReLU()
        self.layer_6 = nn.Linear(2048, 29)

    def forward(self, x, previous_states: tuple[torch.Tensor, torch.Tensor]):
        previous_state_h, previous_state_c = previous_states
        x = self.layer_1(x)
        x = self.relu_1(x)
        x = self.layer_2(x)
        x = self.relu_2(x)
        x = self.layer_3(x)
        x = self.relu_3(x)
        x, (new_state_h, new_state_c) = self.lstm(x, (previous_state_h.unsqueeze(0), previous_state_c.unsqueeze(0)))
        x = self.layer_5(x)
        x = self.relu_5(x)
        x = self.layer_6(x)
        x = x.log_softmax(dim=-1)
        return x, (new_state_h.squeeze(0), new_state_c.squeeze(0))


if __name__ == '__main__':
    import argparse
    import st3.init.coqui

    parser = argparse.ArgumentParser()
    parser.add_argument('output_checkpoint')
    args = parser.parse_args()

    model = DeepSpeech()
    model.load_state_dict(st3.init.coqui.state_dict())
    model.eval()

    import tensorflow as tf

    with tf.compat.v1.Session() as sess:
        tf_logits, tf_new_state_h, tf_new_state_c = sess.run(
            fetches=(sess.graph.get_tensor_by_name('logits:0'),
                     sess.graph.get_tensor_by_name('new_state_h:0'),
                     sess.graph.get_tensor_by_name('new_state_c:0')),
            feed_dict={sess.graph.get_tensor_by_name('input_node:0'): np.ones((1,16,19,26)),
                       sess.graph.get_tensor_by_name('input_lengths:0'): [1],
                       sess.graph.get_tensor_by_name('previous_state_h:0'): np.zeros((1,2048)),
                       sess.graph.get_tensor_by_name('previous_state_c:0'): np.zeros((1,2048))})

    with torch.no_grad():
        x, (new_state_h, new_state_c) = model(torch.ones(1, 1, 494), (torch.zeros(1,2048), torch.zeros(1,2048)))
        print(torch.nn.functional.normalize(new_state_c) @ torch.nn.functional.normalize(torch.tensor(tf_new_state_c)).T)
        assert torch.allclose(new_state_c[0], torch.tensor(tf_new_state_c[0]), atol=1e-3)

        assert torch.allclose(new_state_h[0], torch.tensor(tf_new_state_h[0]), atol=1e-3)

        x = x[0].exp()
        print(torch.nn.functional.normalize(x, dim=0) @ torch.nn.functional.normalize(torch.tensor(tf_logits[0])).T)
        assert torch.allclose(x, torch.tensor(tf_logits[0]), atol=1e-6)

    torch.save(model, args.output_checkpoint)