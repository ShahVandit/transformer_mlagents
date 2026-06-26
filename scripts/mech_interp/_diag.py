import torch, numpy as np
from policy_backend import TransformerPolicyBackend

# 1. histogram of saved relative_norms
d = torch.load(r"..\..\results\mech_interp\crosscoder\crosscoder_encoding.pt", map_location="cpu")
r = d["relative_norms"].numpy()
print("relative_norms: min=%.3f  p5=%.3f  median=%.3f  p95=%.3f  max=%.3f" %
      (r.min(), np.percentile(r,5), np.median(r), np.percentile(r,95), r.max()))
hist,edges = np.histogram(r, bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
print("bins :", " ".join(f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(hist))))
print("count:", " ".join(f"{h:6d}" for h in hist))

# 2. how similar ARE the two models' encodings on the same obs?
dev = "cuda" if torch.cuda.is_available() else "cpu"
obs = torch.load(r"..\..\results\mech_interp\captures\task2_v5\activations.pt", map_location="cpu")["obs"].float()[:8000]
b1 = TransformerPolicyBackend(r"..\..\results\task1_v9\Drone\checkpoint.pt", n_head=4, device=dev)
b2 = TransformerPolicyBackend(r"..\..\results\task2_v5\Drone\checkpoint.pt", n_head=4, device=dev)
for layer in ("resid.0","resid.1","resid.2","encoding"):
    e1 = b1.capture(obs)[layer].float(); e2 = b2.capture(obs)[layer].float()
    cos = torch.nn.functional.cosine_similarity(e1, e2, dim=-1)
    rel = (e1-e2).norm(dim=-1) / (e1.norm(dim=-1)+1e-8)
    print(f"{layer:9s}  cos(enc1,enc2)= {cos.mean():.4f}  ||d||/||e||= {rel.mean():.4f}")
