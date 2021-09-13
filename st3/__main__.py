import argparse
from pathlib import Path

import torch
import torchaudio

from st3.decode import decoder
import st3.init.coqui

coqui = st3.init.coqui.Coqui()
pt_path = Path(coqui.stt_pb).with_suffix('.pt')

parser = argparse.ArgumentParser(description=f'ST3 Decoder for {pt_path}')
parser.add_argument('audios', metavar='FILE', type=Path, nargs='*',
                    help='audio filenames to recognize')
parser.add_argument('--export', action='store_true',
                    help=f'read {coqui.stt_pb} and export TorchScript-based decoder as {pt_path}')

args = parser.parse_args()

if args.export:
    decoder.model.load_state_dict(st3.init.coqui.state_dict())
    torch.jit.save(decoder, pt_path)
else:
    decoder = torch.jit.load(pt_path)

for filename in args.audios:
    waveform, sr = torchaudio.load(filename, channels_first=False, normalize=True)
    waveform = waveform.squeeze()
    assert sr == coqui.sample_rate

    print(decoder(waveform))
