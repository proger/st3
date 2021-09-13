import tensorflow as tf
from tensorflow.python.ops import gen_audio_ops as contrib_audio

from st3.init.coqui import Coqui

def mfcc(audio):
    """
    Adapted from https://github.com/coqui-ai/STT/blob/f94d16bcc3d1476a20799f11572cf9077b893173/training/coqui_stt_training/util/feeding.py#L51
    """
    spectrogram = contrib_audio.audio_spectrogram(
        audio,
        window_size=Coqui.audio_window_samples,
        stride=Coqui.audio_step_samples,
        magnitude_squared=True,
    )

    features = contrib_audio.mfcc(
        spectrogram=spectrogram,
        sample_rate=Coqui.sample_rate,
        dct_coefficient_count=Coqui.n_input,
        upper_frequency_limit=Coqui.sample_rate // 2,
    )
    features = tf.reshape(features, [-1, Coqui.n_input])
    return features
