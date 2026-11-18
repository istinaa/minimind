"""
实验 3：去掉残差连接会怎样 —— 配合 docs/notes/02-残差连接与hidden_state.md 阅读

在同一个随机初始化的 MiniMind 上对比两种 Block：
  有残差：h = x + Attn(Norm(x));  x' = h + FFN(Norm(h))      （原版）
  无残差：h =     Attn(Norm(x));  x' =     FFN(Norm(h))

逐层记录三个量：
  (a) 与第 0 层输入的余弦相似度   —— 原始 token 信息还剩多少
  (b) 同一句中不同位置之间的平均余弦相似度 —— 各 token 的表示是否"糊成一团"
  (c) 原 token 可恢复率：用 embedding 矩阵对 x_l 打分，最高分是不是输入 token 本身
  (d) loss 对每层输入 hidden_state 的梯度范数 —— 梯度能否传回底层

用法：
  python experiments/exp3_residual.py              # 默认 8 层
  python experiments/exp3_residual.py --layers 32  # 更深的网络
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model_minimind import MiniMindBlock, MiniMindConfig, MiniMindForCausalLM  # noqa: E402

_orig_forward = MiniMindBlock.forward


def no_residual_forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    h, present = self.self_attn(self.input_layernorm(hidden_states), position_embeddings,
                                past_key_value, use_cache, attention_mask)
    return self.mlp(self.post_attention_layernorm(h)), present


def run(residual: bool, layers: int):
    MiniMindBlock.forward = _orig_forward if residual else no_residual_forward
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniMindForCausalLM(MiniMindConfig(num_hidden_layers=layers)).to(device)
    ids = torch.randint(0, 6400, (4, 64), device=device)

    states = []
    hooks = [layer.register_forward_pre_hook(lambda m, args: (args[0].retain_grad(), states.append(args[0])) and None)
             for layer in model.model.layers]
    hooks.append(model.model.norm.register_forward_pre_hook(lambda m, args: states.append(args[0]) and None))
    out = model(ids, labels=ids)
    out.loss.backward()
    for h in hooks:
        h.remove()

    x0 = states[0].detach().float()
    E = model.model.embed_tokens.weight.detach().float()     # [V, 768]
    rows = []
    for l, s in enumerate(states):
        s_ = s.detach().float()
        keep = F.cosine_similarity(s_, x0, dim=-1).mean().item()
        n = F.normalize(s_, dim=-1)
        sim = n @ n.transpose(1, 2)                          # [B, T, T]
        T = sim.shape[-1]
        pair = ((sim.sum((-1, -2)) - T) / (T * (T - 1))).mean().item()
        recover = ((s_ @ E.T).argmax(-1) == ids).float().mean().item()
        grad = s.grad.float().norm(dim=-1).mean().item() if s.retains_grad else float("nan")
        rows.append((l, keep, pair, recover, grad))
    MiniMindBlock.forward = _orig_forward
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8)
    args = ap.parse_args()
    L = args.layers
    res, nores = run(True, L), run(False, L)
    show = sorted(set(list(range(0, L + 1, max(1, L // 8))) + [L]))
    print(f"{L} 层。l 表示第 l 个 Block 的输入（l={L} 即最后一层输出）。每格左为有残差，右为无残差\n")
    print(f"{'l':>4} | {'与x0相似度':>13} | {'位置间相似度':>11} | {'原token可恢复率':>11} | {'梯度范数':>19}")
    for l in show:
        (_, k1, p1, r1, g1), (_, k2, p2, r2, g2) = res[l], nores[l]
        print(f"{l:>4} | {k1:>7.3f} {k2:>7.3f} | {p1:>6.3f} {p2:>6.3f} | {r1:>6.1%} {r2:>6.1%} | {g1:>9.2e} {g2:>9.2e}")


if __name__ == "__main__":
    main()
