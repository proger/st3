import torch
import torch.nn as nn
import torchaudio

from st3.init.coqui import Coqui


class MFCC(nn.Module):
    def __init__(self):
        super().__init__()

        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=Coqui.sample_rate,
            n_mfcc=Coqui.n_input,
            log_mels=True,
            melkwargs=dict(n_fft=Coqui.audio_window_samples,
                           # https://github.com/tensorflow/tensorflow/blob/8d72537c6abf5a44103b57b9c2e22c14f5f49698/tensorflow/core/kernels/mfcc.cc#L27-L28
                           n_mels=Coqui.n_input, # TODO: recheck: tf uses 40?
                           win_length=Coqui.audio_window_samples,
                           hop_length=Coqui.audio_step_samples,
                           f_min=20.,
                           f_max=Coqui.sample_rate / 2.,
                           # https://github.com/tensorflow/tensorflow/blob/8d72537c6abf5a44103b57b9c2e22c14f5f49698/tensorflow/core/kernels/spectrogram.cc#L29
                           window_fn=torch.hann_window,
                           pad=0,
                           power=2.,
                           center=False,
                           mel_scale='htk'))

    def forward(self, waveform):
        return self.mfcc(waveform).T


mfcc = MFCC()


if __name__ == '__main__':
    import sys
    import st3.init.mfcc_tf

    waveform, sr = torchaudio.load(sys.argv[1], channels_first=False, normalize=True)
    assert sr == 16000
    features = torch.tensor(st3.init.mfcc_tf.mfcc(waveform.numpy()).numpy())
    waveform = waveform.squeeze()

    #print((features - mfcc(waveform)).abs())

    import matplotlib.pyplot as plt
    import torchvision.utils as tu
    plt.imshow(tu.make_grid([features.unsqueeze(0),
                             mfcc(waveform).unsqueeze(0),
                             (features - mfcc(waveform)).pow(2).unsqueeze(0)],
               nrow=1, normalize=True, padding=0).permute(2,1,0))
    plt.show()
