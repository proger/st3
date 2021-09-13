import torch
import torch.nn as nn

import st3.mfcc
from st3.model import DeepSpeech


@torch.jit.script_if_tracing
def beams(ctc_probs, beam_size: int, alphabet: str, blank_index: int):
    """
    Return top-beam_size beams using CTC prefix beam search,
    adapted from https://github.com/wenet-e2e/wenet/blob/829d2c85e0495094636c2e7ab7a24c29818e1eff/wenet/transformer/asr_model.py#L329
    """

    # cur_hyps: (prefix, (blank_ending_score, non_blank_ending_score))
    cur_hyps: list[tuple[str, tuple[float, float]]] = [("", (float(0.0), -float('inf')))]

    for logp in ctc_probs:
        # key: prefix, value (pb, pnb), default value(-inf, -inf)
        next_hyps: dict[str, tuple[float, float]] = {}
        neginf = (-float('inf'), -float('inf'))

        _, top_k_indices = logp.topk(beam_size)
        for s in top_k_indices:
            s = int(s.item())
            ps = logp[s].item()

            for prefix, (pb, pnb) in cur_hyps:
                last = prefix[-1] if len(prefix) > 0 else ""
                if s == blank_index:
                    n_pb, n_pnb = next_hyps.get(prefix, neginf)
                    n_pb = torch.tensor([n_pb, float(pb + ps), float(pnb + ps)]).logsumexp(dim=0).item()
                    next_hyps[prefix] = (n_pb, n_pnb)
                elif alphabet[s] == last:
                    # Update *ss -> *s
                    n_pb, n_pnb = next_hyps.get(prefix, neginf)
                    n_pnb = torch.tensor([n_pnb, float(pnb + ps)]).logsumexp(dim=0).item()
                    next_hyps[prefix] = (n_pb, n_pnb)

                    # Update *sεs -> *ss
                    n_prefix = prefix + alphabet[s]
                    n_pb, n_pnb = next_hyps.get(n_prefix, neginf)
                    n_pnb = torch.tensor([n_pnb, float(pb + ps)]).logsumexp(dim=0).item()
                    next_hyps[n_prefix] = (n_pb, n_pnb)
                else:
                    n_prefix = prefix + alphabet[s]
                    n_pb, n_pnb = next_hyps.get(n_prefix, neginf)
                    n_pnb = torch.tensor([n_pnb, float(pb + ps), float(pnb + ps)]).logsumexp(dim=0).item()
                    next_hyps[n_prefix] = (n_pb, n_pnb)


        next_hyps_list: list[tuple[str, tuple[float, float]]] = list(next_hyps.items())
        _, indices = torch.tensor([float(torch.tensor([float(hyp[1][0]), float(hyp[1][1])]).logsumexp(dim=0))
                                   for hyp in next_hyps_list]).topk(beam_size)
        cur_hyps = [next_hyps_list[index] for index in indices]

    return cur_hyps


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(19, 26), padding=(9, 0))
        self.frontend = st3.mfcc.MFCC()
        model = DeepSpeech()
        self.hidden_size = model.lstm.hidden_size
        self.model = torch.jit.trace(model, (torch.randn(1, 19*26), (torch.randn(1, self.hidden_size), torch.randn(1, self.hidden_size))))

        self.beam_size = 10
        self.alphabet = " abcdefghijklmnopqrstuvwxyz'ε"
        self.blank_index = self.alphabet.find('ε')
        assert self.blank_index >= 0
        assert len(self.alphabet) == model.layer_6.out_features

    def forward(self, waveform):
        state = (torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size))
        frames = self.frontend(waveform).unsqueeze(0)
        windows = self.unfold(frames.unsqueeze(0)).permute(2,0,1)

        outputs = []
        for window in windows:
            out, state = self.model(window.reshape(1,-1), state)
            outputs.append(out)

        return beams(torch.stack(outputs).squeeze(), self.beam_size, self.alphabet, self.blank_index)


decoder = torch.jit.script(Decoder())