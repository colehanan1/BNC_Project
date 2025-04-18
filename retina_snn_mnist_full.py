#!/usr/bin/env python3
"""
retina_snn_mnist_conv_fixed.py

Spiking CNN + R‑STDP for MNIST, no inhibition:
 - Retina pooling (28→7) + Poisson spikes
 - Conv2d input→hidden (dynamic hidden size)
 - Hidden LIF neurons + placeholder STDP
 - Reward‑modulated STDP on output
 - Logs every 10 train steps and every 100 test steps
"""

import argparse, random
import torch, torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

device = torch.device("mps") if torch.backends.mps.is_available() \
         else torch.device("cuda") if torch.cuda.is_available() \
         else torch.device("cpu")
print("Running on", device)

def encode_image_to_spikes(img, T=250, rate=100):
    x = img.unsqueeze(0)                    # (1,1,28,28)
    x = F.avg_pool2d(x,2,2)                 # (1,1,14,14)
    x = F.avg_pool2d(x,2,2)                 # (1,1,7,7)
    inten = x.squeeze()                     # (7,7)
    p = torch.clamp(inten.flatten()*(rate/1000),0,1)
    p = p.view(-1,1).expand(-1,T)           # (49,T)
    return (torch.rand(p.shape,device=img.device)<p).float(), inten

class SpikingNetwork:
    def __init__(self,
                 maps=8, kernel=5, stride=2,
                 tau_m=20., tau_ref=2,
                 A_plus=0.02, A_minus=0.01, w_max=0.5,
                 eta=0.01, alpha_e=0.95):
        # R‑STDP & STDP parameters
        self.A_plus, self.A_minus, self.w_max = A_plus, A_minus, w_max
        self.tau_trace = 50.; self.alpha_trace = np.exp(-1/self.tau_trace)
        self.eta_rstdp = eta; self.alpha_e = alpha_e
        # Conv input→hidden
        self.conv = torch.nn.Conv2d(1, maps, kernel, stride=stride, bias=False).to(device)
        torch.nn.init.uniform_(self.conv.weight,0,0.1)
        # infer hidden size
        with torch.no_grad():
            d = torch.zeros(1,1,7,7,device=device)
            o = self.conv(d)
            self.N_exc = int(np.prod(o.shape[1:]))
        # output dim
        self.N_out = 10
        # LIF params
        self.v_thr=1.0; self.v_reset=0.0
        self.tau_m, self.tau_ref = tau_m, tau_ref
        self.alpha_v = np.exp(-1/self.tau_m)
        # state & traces
        self.reset_state()
        self.trace_pre  = torch.zeros(self.N_exc,device=device)
        self.trace_post = torch.zeros(self.N_exc,device=device)
        self.e_trace    = torch.zeros(self.N_exc,self.N_out,device=device)
        # output weights
        self.W_out = torch.randn(self.N_exc,self.N_out,device=device)*0.1

    def reset_state(self):
        self.v = torch.zeros(self.N_exc,device=device)
        self.ref = torch.zeros(self.N_exc,device=device,dtype=torch.int32)

    def simulate_step(self, inp_spikes):
        # conv→hidden drive
        maps = inp_spikes.view(1,1,7,7)
        o = self.conv(maps)                   # (1, C, H, W)
        I = o.view(-1)                        # flatten to (N_exc,)
        # membrane update
        self.v = self.alpha_v*self.v + I
        # refractory clamp
        self.v[self.ref>0] = self.v_reset
        # spiking
        s = (self.v>=self.v_thr)&(self.ref==0)
        self.v[s] = self.v_reset
        self.ref[s] = int(self.tau_ref)
        # decay refractory counters
        self.ref[self.ref>0] -= 1
        return s.float()

    def stdp_update(self, pre, post):
        # placeholder: here you’d update conv weights
        self.trace_pre  = self.trace_pre*self.alpha_trace + pre
        self.trace_post = self.trace_post*self.alpha_trace + post
        # (full STDP on conv omitted for brevity)

    def update_eligibility(self, hidden_spk, out_spk):
        self.e_trace = self.alpha_e*self.e_trace + hidden_spk.unsqueeze(1)*out_spk.unsqueeze(0)

    def apply_rstdp(self, r):
        self.W_out += self.eta_rstdp * r * self.e_trace

    def train(self, ds, N=10000, T=250, rate=100):
        for i in range(N):
            img, lbl = ds[i]
            spikes, _ = encode_image_to_spikes(img.to(device), T, rate)
            self.reset_state()
            self.trace_pre.zero_(); self.trace_post.zero_(); self.e_trace.zero_()
            total = torch.zeros(self.N_exc,device=device)
            for t in range(T):
                s = self.simulate_step(spikes[:,t])
                self.stdp_update(s,s)
                total += s
                # R‑STDP eligibility
                o_current = self.W_out.t() @ s
                o_spk     = (o_current>=self.v_thr).float()
                self.update_eligibility(s,o_spk)
            # reward
            votes = (total.unsqueeze(1)*self.W_out).sum(0)
            p     = votes.argmax().item()
            r     = 1.0 if p==lbl else -1.0
            self.apply_rstdp(r)
            if (i+1)%10==0:
                print(f"[Train] processed {i+1}/{N}")
        print("Training complete")

    def evaluate(self, ds, N=2000, T=250, rate=100):
        correct=0
        true, pred = [],[]
        for i in range(N):
            img,lbl=ds[i]
            spikes,_=encode_image_to_spikes(img.to(device),T,rate)
            self.reset_state()
            total=torch.zeros(self.N_exc,device=device)
            for t in range(T):
                s=self.simulate_step(spikes[:,t])
                total+=s
            votes=(total.unsqueeze(1)*self.W_out).sum(0)
            p=votes.argmax().item()
            true.append(lbl); pred.append(p)
            if p==lbl: correct+=1
            if (i+1)%100==0:
                print(f"[Test] accuracy: {correct/(i+1)*100:.2f}%")
        final=correct/N*100
        print(f"[Test] final accuracy: {final:.2f}%")
        cm=confusion_matrix(true,pred)
        disp=ConfusionMatrixDisplay(cm,display_labels=list(range(self.N_out)))
        disp.plot(cmap="Blues"); plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples",  type=int, default=2000)
    args = parser.parse_args()

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST("./data",train=True,download=True,transform=tf)
    test_ds  = torchvision.datasets.MNIST("./data",train=False,download=True,transform=tf)

    snn = SpikingNetwork()
    snn.train(train_ds, N=args.train_samples)
    snn.evaluate(test_ds, N=args.test_samples)

if __name__=="__main__":
    main()
