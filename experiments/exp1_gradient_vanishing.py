"""
实验 1：10 层网络中，有/无归一化时的激活标准差与梯度变化（梯度消失可视化）

对比三种结构（每种都是 10 个 block，权重初始化相同）：
  1. No Norm        : x = tanh(W x)                       —— 纯堆叠，无归一化
  2. RMSNorm        : x = tanh(RMSNorm(W x))              —— 每层加归一化
  3. Pre-Norm + Res : x = x + tanh(W · RMSNorm(x))        —— MiniMind 的 Pre-LN 残差结构

记录每一层的三个量：
  (a) 前向激活标准差       std(x_l)
  (b) 对激活的梯度范数     ||dL/dx_l||   —— 反向传播的“信号强度”，最能直接反映梯度消失
  (c) 对权重的梯度范数     ||dL/dW_l||   —— 优化器实际拿到的更新量

用法：
  python experiments/exp1_gradient_vanishing.py
  python experiments/exp1_gradient_vanishing.py --init_std 0.05 --act tanh --layers 10
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


# 与 model/model_minimind.py 中的 RMSNorm 完全一致（拷贝过来是为了不依赖 transformers）
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


ACTS = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": F.relu, "gelu": F.gelu}


class Block(nn.Module):
    def __init__(self, dim, mode, act):
        super().__init__()
        self.mode, self.act = mode, ACTS[act]
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(dim) if mode != "none" else nn.Identity()

    def forward(self, x):
        if self.mode == "none":       # x = act(Wx)
            return self.act(self.linear(x))
        if self.mode == "norm":       # x = act(Norm(Wx))
            return self.act(self.norm(self.linear(x)))
        if self.mode == "prenorm":    # x = x + act(W·Norm(x))   (Pre-LN 残差)
            return x + self.act(self.linear(self.norm(x)))
        raise ValueError(self.mode)


class DeepNet(nn.Module):
    def __init__(self, dim, n_layers, mode, act, init_std):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, mode, act) for _ in range(n_layers)])
        # Pre-LN 结构末尾跟一个最终 Norm（与 MiniMindModel.norm 对应）
        self.final_norm = RMSNorm(dim) if mode == "prenorm" else nn.Identity()
        self.head = nn.Linear(dim, dim, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)

    def forward(self, x):
        acts = []
        for blk in self.blocks:
            x = blk(x)
            x.retain_grad()           # 保留中间激活的梯度，便于统计 dL/dx_l
            acts.append(x)
        return self.head(self.final_norm(x)), acts


def run_one(mode, args):
    torch.manual_seed(args.seed)      # 同一种子 → 三种结构的输入、目标、权重初始值完全相同
    model = DeepNet(args.dim, args.layers, mode, args.act, args.init_std)
    x = torch.randn(args.batch, args.dim)
    y = torch.randn(args.batch, args.dim)

    out, acts = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()

    return {
        "act_std": [a.detach().std().item() for a in acts],
        "act_grad": [a.grad.norm().item() for a in acts],
        "w_grad": [b.linear.weight.grad.norm().item() for b in model.blocks],
    }


# 配色：dataviz 参考调色板的前三个分类色（蓝/橙/青，已通过色盲校验）；另用不同 marker 做二次编码
STYLES = {
    "none":    dict(label="No Norm",         color="#2a78d6", marker="o"),
    "norm":    dict(label="RMSNorm",         color="#eb6834", marker="s"),
    "prenorm": dict(label="Pre-Norm + Res",  color="#1baf7a", marker="^"),
}
INK, MUTED, GRID = "#1f2328", "#59636e", "#e6e8eb"


def plot(results, args, path):
    panels = [
        ("act_std",  "(a) Activation std per layer",            "std(x_l)"),
        ("act_grad", "(b) Gradient w.r.t. activation",          "||dL/dx_l||"),
        ("w_grad",   "(c) Gradient w.r.t. weight",              "||dL/dW_l||"),
    ]
    layers = list(range(1, args.layers + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    for ax, (key, title, ylabel) in zip(axes, panels):
        for mode, st in STYLES.items():
            vals = results[mode][key]
            ax.plot(layers, vals, color=st["color"], marker=st["marker"],
                    linewidth=2, markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                    label=st["label"])
            # 只在一个端点做直接标注，避免每个点都写数字；
            # 激活梯度面板标在第 1 层（梯度消失体现在靠近输入端，且末层数值挤在一起）
            i = 0 if key == "act_grad" else -1
            ax.annotate(f"{vals[i]:.1e}", (layers[i], vals[i]), xytext=(-6 if i == 0 else 6, 0),
                        textcoords="offset points", va="center", ha="right" if i == 0 else "left",
                        fontsize=8, color=MUTED)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Layer (1 = closest to input)", color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.set_xticks(layers)
        ax.set_xlim(-0.8 if key == "act_grad" else 0.5,
                    args.layers + (0.5 if key == "act_grad" else 1.8))
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{args.layers}-layer net, act={args.act}, dim={args.dim}, init N(0, {args.init_std}²)",
                 y=0.925, fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    print(f"图已保存: {path}")


def print_table(results, args):
    for key, name in [("act_std", "激活标准差 std(x_l)"), ("act_grad", "激活梯度 ||dL/dx_l||"),
                      ("w_grad", "权重梯度 ||dL/dW_l||")]:
        print(f"\n== {name} ==")
        print("layer " + "".join(f"{STYLES[m]['label']:>17}" for m in STYLES))
        for i in range(args.layers):
            print(f"{i + 1:>5} " + "".join(f"{results[m][key][i]:>17.3e}" for m in STYLES))
    print("\n首层/末层 激活梯度比（越小 → 梯度消失越严重）:")
    for m, st in STYLES.items():
        g = results[m]["act_grad"]
        print(f"  {st['label']:<16} {g[0] / g[-1]:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--act", type=str, default="tanh", choices=list(ACTS))
    p.add_argument("--init_std", type=float, default=0.02, help="权重初始化标准差（LLM 常用 0.02）")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = p.parse_args()

    results = {mode: run_one(mode, args) for mode in STYLES}
    print_table(results, args)
    os.makedirs(args.out, exist_ok=True)
    plot(results, args, os.path.join(args.out, f"exp1_{args.act}_std{args.init_std}.png"))


if __name__ == "__main__":
    main()
