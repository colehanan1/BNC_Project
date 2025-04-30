import time, torch, argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from snn_mnist_optuna_poission import SNN, poisson_encode   # your SNN file

# ---------- helpers -----------------------------------------------------------
def load_mnist(flatten=False):
    tf = transforms.ToTensor()
    tr = datasets.MNIST('.', train=True,  download=True, transform=tf)
    te = datasets.MNIST('.', train=False, download=True, transform=tf)
    if flatten:
        return ((tr.data.view(-1,784).float()/255, tr.targets),
                (te.data.view(-1,784).float()/255, te.targets))
    return tr, te

def bench_logreg():
    (Xtr,ytr),(Xte,yte)=load_mnist(flatten=True)
    lr = LogisticRegression(solver='saga',C=50,max_iter=10_000,n_jobs=5)
    tic=time.perf_counter(); lr.fit(Xtr.numpy(),ytr.numpy())
    train_t=time.perf_counter()-tic
    tic=time.perf_counter(); lr.predict(Xte.numpy())
    pred_t=(time.perf_counter()-tic)/len(yte)
    print(f"LR ▸ train {train_t:6.1f}s | predict {pred_t*1e3:6.2f} ms/img | "
          f"acc {lr.score(Xte.numpy(),yte.numpy()):.4f}")

def bench_snn(device,T=300,epochs=20):
    tr,te=load_mnist(); tl=DataLoader(tr,64,True); vl=DataLoader(te,64)
    # drop-in Optuna winners from the report
    model=SNN(49,203,10,torch.tensor(0.904),torch.tensor(0.904)).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=3e-3)
    tic=time.perf_counter()
    for _ in range(epochs):
        for x,y in tl:
            s=poisson_encode(x.to(device),T)
            opt.zero_grad()
            out=model(s,T)[2].sum(0)
            torch.nn.functional.cross_entropy(out,y.to(device)).backward()
            opt.step()
    train_t=time.perf_counter()-tic
    tic=time.perf_counter()
    with torch.no_grad():
        for x,_ in vl:
            _=model(poisson_encode(x.to(device),T),T)
    pred_t=(time.perf_counter()-tic)/len(te)
    print(f"SNN▸ train {train_t/60:6.1f} min | predict {pred_t*1e3:6.2f} ms/img")

# ---------- main --------------------------------------------------------------
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cpu")
    args=p.parse_args(); dev=torch.device(args.device)
    bench_logreg(); print('-'*60); bench_snn(dev)
