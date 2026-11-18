"""
实验 2：MiniMind 模型解剖 —— 配合 docs/notes/01-从组件到完整模型.md 阅读

用 hook 观察一次完整前向传播，并验证几个结构性质：
  1. 各组件参数量
  2. 张量在 embedding → block → lm_head 中的形状变化
  3. 残差流：每层主干范数，以及注意力 / FFN 各往主干写入了多少
  4. 初始 loss ≈ ln(vocab_size)
  5. 因果性：改动后面的 token，不影响前面位置的输出
  6. KV Cache：逐 token 增量推理与整句一次性前向的结果一致

注意：仓库里没有训练好的权重，这里是随机初始化的模型。
      形状、参数量、因果性、KV Cache 这些结构性质与训练无关；残差流的数值则只反映初始化状态。

用法：
  python experiments/exp2_model_anatomy.py
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def n_params(module):
    return sum(p.numel() for p in module.parameters())


def section(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MiniMindConfig()
    model = MiniMindForCausalLM(cfg).to(device).eval()
    B, T = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # ---------------- 1. 参数量 ----------------
    section("1. 参数量")
    blk = model.model.layers[0]
    total = n_params(model)  # 共享权重的参数只算一次
    emb = n_params(model.model.embed_tokens)
    attn, ffn = n_params(blk.self_attn), n_params(blk.mlp)
    norms = n_params(blk.input_layernorm) + n_params(blk.post_attention_layernorm)
    L = cfg.num_hidden_layers
    print(f"embed_tokens (= lm_head, 共享)  {emb / 1e6:6.2f}M  {emb / total:5.1%}")
    print(f"Attention × {L}                  {L * attn / 1e6:6.2f}M  {L * attn / total:5.1%}")
    print(f"FFN × {L}                        {L * ffn / 1e6:6.2f}M  {L * ffn / total:5.1%}")
    print(f"RMSNorm × {2 * L + 1}                    {(L * norms + cfg.hidden_size) / 1e3:6.1f}K")
    print(f"总计                           {total / 1e6:6.2f}M")
    print(f"lm_head 与 embed 是同一个张量: {model.lm_head.weight is model.model.embed_tokens.weight}")

    # ---------------- 2 & 3. 形状与残差流 ----------------
    shapes, stream = [], []

    def shape_hook(name):
        def fn(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            shapes.append((name, tuple(inp[0].shape), tuple(o.shape)))
        return fn

    hooks = [model.model.embed_tokens.register_forward_hook(shape_hook("embed_tokens"))]
    a = blk.self_attn
    for name, mod in [("input_layernorm", blk.input_layernorm), ("q_proj", a.q_proj), ("k_proj", a.k_proj),
                      ("v_proj", a.v_proj), ("o_proj", a.o_proj), ("self_attn", a),
                      ("post_attention_layernorm", blk.post_attention_layernorm),
                      ("mlp.gate_proj", blk.mlp.gate_proj), ("mlp.up_proj", blk.mlp.up_proj),
                      ("mlp.down_proj", blk.mlp.down_proj), ("block 输出", blk)]:
        hooks.append(mod.register_forward_hook(shape_hook(f"layer0.{name}")))
    hooks += [model.model.norm.register_forward_hook(shape_hook("最终 norm")),
              model.lm_head.register_forward_hook(shape_hook("lm_head"))]

    # 残差流：记录每层输入范数、注意力写入范数、FFN 写入范数（按 token 取 L2 范数再求平均）
    rms = lambda t: t.float().norm(dim=-1).mean().item()
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.register_forward_pre_hook(
            lambda m, args, i=i: stream.append({"layer": i, "in": rms(args[0])})))
        hooks.append(layer.self_attn.register_forward_hook(
            lambda m, inp, out: stream[-1].__setitem__("attn", rms(out[0]))))
        hooks.append(layer.mlp.register_forward_hook(
            lambda m, inp, out: stream[-1].__setitem__("ffn", rms(out))))
        hooks.append(layer.register_forward_hook(
            lambda m, inp, out: stream[-1].__setitem__("out", rms(out[0]))))

    with torch.no_grad():
        out = model(ids, labels=ids)
    for h in hooks:
        h.remove()

    section(f"2. 张量形状（B={B}, T={T}，只展示第 0 层 block 内部）")
    for name, i, o in shapes:
        print(f"{name:<34} {str(i):<18} -> {o}")
    print(f"注：q/k/v 投影后会 view 成 [B, T, 头数, {cfg.head_dim}]：Q {cfg.num_attention_heads} 头，"
          f"K/V {cfg.num_key_value_heads} 头（GQA），再 repeat_kv 对齐")

    section("3. 残差流（每 token 向量的 L2 范数，取平均；随机初始化）")
    print(f"{'layer':>5} {'主干输入':>10} {'Attn写入':>10} {'FFN写入':>10} {'主干输出':>10}")
    for s in stream:
        print(f"{s['layer']:>5} {s['in']:>10.3f} {s['attn']:>10.3f} {s['ffn']:>10.3f} {s['out']:>10.3f}")

    # ---------------- 4. 初始 loss ----------------
    section("4. 初始 loss")
    print(f"loss = {out.loss.item():.4f}   ln(vocab_size) = ln({cfg.vocab_size}) = {math.log(cfg.vocab_size):.4f}")

    # ---------------- 5. 因果性 ----------------
    section("5. 因果性")
    ids2 = ids.clone()
    ids2[:, 10:] = torch.randint(0, cfg.vocab_size, (B, T - 10), device=device)
    with torch.no_grad():
        l1, l2 = model(ids).logits, model(ids2).logits
    print(f"改动位置 10 之后的 token：")
    print(f"  位置 0-9 logits 最大差异  {(l1[:, :10] - l2[:, :10]).abs().max().item():.2e}  (应 ≈ 0)")
    print(f"  位置 10-15 logits 最大差异 {(l1[:, 10:] - l2[:, 10:]).abs().max().item():.2e}  (应明显 > 0)")

    # ---------------- 6. KV Cache ----------------
    section("6. KV Cache 增量推理 vs 整句前向")
    with torch.no_grad():
        full = model(ids).logits
        past, steps = None, []
        for t in range(T):
            o = model(ids[:, t:t + 1], past_key_values=past, use_cache=True)
            past = o.past_key_values
            steps.append(o.logits[:, -1])
        inc = torch.stack(steps, dim=1)
    print(f"逐 token 增量推理与整句前向的 logits 最大差异: {(full - inc).abs().max().item():.2e}")
    print(f"第 0 层 KV Cache 形状: K {tuple(past[0][0].shape)}  V {tuple(past[0][1].shape)}"
          f"   [B, 已缓存长度, KV头数, head_dim]")


if __name__ == "__main__":
    main()
