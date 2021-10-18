import argparse
from pathlib import Path
from loguru import logger


import bitsandbytes as bnb
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, one_hot
from torch.utils.tensorboard import SummaryWriter
import torchaudio


parser = argparse.ArgumentParser()
parser.add_argument('--init', type=Path, required=False, help='initial checkpoint file, get using python -m st3.model init.pth')
parser.add_argument('--epochs', default=1000, help='how many epochs to repeat')
parser.add_argument('--comment', type=str, default='', required=False)
parser.add_argument('output', type=Path)
args = parser.parse_args()

device = 'cuda'

writer = SummaryWriter(args.comment)

from st3.model import DeepSpeech, ClippedReLU  # for pickle
#model = DeepSpeech()

class BLSTMP(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=26, out_channels=13, kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=13, out_channels=6, kernel_size=3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=6, out_channels=13, kernel_size=3, stride=1)

        self.lstm = nn.LSTM(input_size=13, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True, proj_size=3)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x[:,:,:3] + x[:,:,3:]
        return x.log_softmax(dim=-1)

model = BLSTMP()
model = model.to(device)

if args.init:
    model.load_state_dict(torch.load(args.init)['model'])
else:
    logger.debug('random init')
    def weights_init_(m):
        print(type(m))
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in',
                                    nonlinearity='relu')

    model.apply(weights_init_)

from st3.init.coqui import Coqui

alphabet = {c: i for i, c in enumerate(Coqui.alphabet)}

def tokenize(string):
    return torch.tensor([alphabet[char] for char in string.lower() if char in alphabet])

def collate(batch):
    (sources, _, targets, _, _, _) = zip(*batch)
    sources = [source.squeeze() for source in sources]
    targets = [tokenize(target) for target in targets]
    source_lengths = torch.tensor([len(source) for source in sources])
    target_lengths = torch.tensor([len(target) for target in targets])
    source = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=0.0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=Coqui.blank_index)
    return source, targets, source_lengths, target_lengths

#upsample = torchaudio.transforms.Resample(8000,16000)

def yesno_collate(batch):
    sources, _, targets = zip(*batch)

    #sources = [upsample(source.squeeze()) for source in sources]
    sources = [source.squeeze() for source in sources]
    #targets = [torch.tensor([2] + t + [2])+1 for t in targets]
    targets = [torch.tensor(t)+1 for t in targets]
    source_lengths = torch.tensor([len(source) for source in sources])
    target_lengths = torch.tensor([len(target) for target in targets])
    source = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=0.0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
    return source, targets, source_lengths, target_lengths

#dataset = torchaudio.datasets.TEDLIUM(root='/tank/tedlium', subset='dev')
#dataset = torchaudio.datasets.LIBRISPEECH('/tank/librispeech', 'train-clean-100')
dataset = torchaudio.datasets.YESNO('yesno', download=True)
data = torch.utils.data.DataLoader([dataset[i] for i in range(56)], batch_size=4, shuffle=True, collate_fn=yesno_collate, pin_memory=True, num_workers=8)
test_data = torch.utils.data.DataLoader([dataset[i] for i in range(50,60)], batch_size=4, pin_memory=True, collate_fn=yesno_collate)

#print(next(iter(data)))

model = model.to(device)
#optimizer = bnb.optim.Adam8bit(model.parameters(), lr=0.00005, betas=(0.9, 0.995), weight_decay=1e-5)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.995))

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-2)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001)# , weight_decay=1e-2)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=1e-5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
# 	max_lr=0.01,
# 	steps_per_epoch=len(data),
# 	epochs=args.epochs,
# 	anneal_strategy='linear')


#scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
scaler = None

model.train()

ctc = nn.CTCLoss(blank=0)

import st3.mfcc
frontend = st3.mfcc.MFCC()
unfold = nn.Unfold(kernel_size=(19, 26), padding=(9, 0))

# 49 frames per second
assert unfold(frontend(torch.randn(4,Coqui.sample_rate)).permute(2,0,1).unsqueeze(1)).permute(0,2,1).shape == torch.Size([4, 49, 494]) # (N,T,C)

frontend = frontend.to(device)
unfold = unfold.to(device)

from torch.profiler import profile, record_function, ProfilerActivity

import itertools as it
steps = it.count()


import Levenshtein as Lev

def decode_argmax(ctc_argmax):
    blank = str(ctc_argmax[0]) == '0'
    s1 = '' if blank else str(ctc_argmax[0])
    for c in (str(c) for c in ctc_argmax[1:]):
        if blank:
            if c == '0': # also blank
                blank = True
            else:
                s1 += c
                blank = False
        else:
            if c == '0':
                blank = True
            elif s1[-1] == c:
                blank = False
            else:
                s1 += c
                blank = False

    return s1

def cer(s1, s2):
    return Lev.distance(s1, s2)


def evaluate(test_data):
    loss = 0.
    total_cer = 0.
    for source, targets, source_lengths, target_lengths in test_data:
        with torch.cuda.amp.autocast(scaler is not None):
            spec = frontend(source.to(device)).permute(2,0,1)
            windows = spec.detach()

            #windows = unfold(spec.unsqueeze(1)).permute(0,2,1)
            #windows = windows.detach()

            output = model(windows)

            print(output.logsumexp(dim=0).logsumexp(dim=0))

            output_lengths = torch.div(source_lengths, Coqui.audio_step_samples, rounding_mode='floor')-1
            # subsampling
            output_lengths = torch.div(output_lengths - 3, 3, rounding_mode='floor')+1
            output_lengths = torch.div(output_lengths - 3, 3, rounding_mode='floor')+1
            output_lengths = torch.div(output_lengths - 3, 1, rounding_mode='floor')+1

            loss += ctc(output.permute(1,0,2), targets.to(device),
                        output_lengths, target_lengths).item()

            for a,b in zip([decode_argmax(m) for m in torch.topk(output, k=1, dim=-1).indices.squeeze().cpu().numpy()], [''.join(str(c.item()) for c in target) for target in targets]):
                print(a,b)
                total_cer += cer(a,b)
    return dict(avg_loss=loss / len(data), avg_cer=total_cer / len(data))

freq_mask = torchaudio.transforms.FrequencyMasking(2)
time_mask = torchaudio.transforms.TimeMasking(2)

for _ in range(args.epochs):
    for source, targets, source_lengths, target_lengths in data:
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(scaler is not None):
            # source is (N,S)

            spec = frontend(source.to(device))
            spec = time_mask(freq_mask(time_mask(freq_mask(spec.permute(2,1,0))))).permute(2,1,0)
            #windows = unfold(spec.permute(2,0,1).unsqueeze(1)).permute(0,2,1)

            #windows = nn.functional.conv1d(windows.permute(0,2,1), torch.ones(494,494,1,dtype=torch.half,device=device), stride=4).permute(0,2,1)
            #windows = windows.detach()

            windows = spec.permute(2,0,1).detach()

            # with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            #         profile_memory=True, record_shapes=True) as prof:
            #     output = model(windows)

            # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

            output = model(windows)

            targets = targets.to(device)
            output_lengths = torch.div(source_lengths, Coqui.audio_step_samples, rounding_mode='floor')-1

            # subsampling
            output_lengths = torch.div(output_lengths - 3, 3, rounding_mode='floor')+1
            output_lengths = torch.div(output_lengths - 3, 3, rounding_mode='floor')+1
            output_lengths = torch.div(output_lengths - 3, 1, rounding_mode='floor')+1
            #print(output_lengths)

            ctc_loss = ctc(output.permute(1,0,2), targets,
                           output_lengths, target_lengths)

            # q = output.logsumexp(dim=1) - torch.log(torch.tensor(output.shape[1], device=device))
            # p = torch.nn.functional.one_hot(torch.cat([targets, torch.zeros(targets.shape[0],2,dtype=torch.long,device=device)], dim=1), num_classes=3).sum(dim=1) / 10
            # ce = -torch.sum(p*q)

            #loss = 0.
            loss = ctc_loss # + 0.1*ce

            # ce = 0
            # for i, output_length in enumerate(output_lengths):
            #     oh = (one_hot(targets[i][None,:])*2 + 0.1).softmax(dim=-1)
            #     ce += (interpolate(oh.permute(0,2,1).float(), size=output_length).permute(0,2,1)[0] * output[i,:output_length,:]).sum()/output_length

            # loss += -0.3*ce

            # for o,t in zip(output.chunk(8,dim=1), targets.chunk(8,dim=1)):
            #     loss += 0.1*ctc(o.permute(1,0,2), t,
            #                 o.shape[1]*torch.ones(o.shape[0],dtype=torch.int32), torch.ones(t.shape[0],1, dtype=torch.int32))

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        #scheduler.step()
        step = next(steps)
        logger.info('step={} loss={} lr={} windows={}', step, loss.item(), [p['lr'] for p in optimizer.param_groups], windows.shape)
        writer.add_scalar('loss', loss.item(), global_step=step)
        #writer.add_scalar('loss/ce', ce.item(), global_step=step)
        writer.add_scalar('loss/ctc', ctc_loss.item(), global_step=step)

    logger.info('eval {}', evaluate(test_data))

torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    #'lr_sched': lr_sched.state_dict(),
}, args.output)