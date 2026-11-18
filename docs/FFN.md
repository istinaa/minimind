二、结构：先升维、再激活、最后降维

经典 FFN 只有两层：
$$
\text{FFN}(x) = W_2,\sigma(W_1 x),\qquad x\in\mathbb{R}^{768}\ \xrightarrow{W_1}\ \mathbb{R}^{2432}\ \xrightarrow{\sigma}\ \xrightarrow{W_2}\ \mathbb{R}^{768}
$$

先把 768 维扩展到约 3 倍宽，经过非线性激活，再压回 768 维。

为什么要先变宽？中间层的每一维可以看作一个**"神经元"或"模式探测器"**。维度越宽，能同时探测的模式就越多。768 维的向量只是"接口宽度"，2432 维的中间层才是"工作空间"。

三、最有用的理解方式：FFN 是一个键值记忆库

Geva et al. 2021（Transformer Feed-Forward Layers Are Key-Value Memories）提供了一个非常直观的视角。把两个矩阵拆开来看：

$$
\text{FFN}(x)=\sum_{i=1}^{2432}\ \underbrace{\sigma(,k_i\cdot x,)}{\text{第 } i \text{ 条记忆的匹配程度}}\ \cdot\ \underbrace{v_i}{\text{第 } i \text{ 条记忆的内容}}
$$

- $W_1$ 的每一行 $k_i$ 是一把**"键"**，代表某种输入模式；
- $k_i\cdot x$ 计算当前向量与这把键的匹配程度，激活函数把匹配程度低的抑制掉；
- $W_2$ 的每一列 $v_i$ 是对应的**"值"**，匹配上之后，就把这个值按匹配强度加到输出里。

这和注意力的结构非常相似：注意力是"query 和当前句子里其他词的 key 做匹配"，FFN 是"x 和训练时存进权重的 2432 条记忆做匹配"。

▎ 注意力：从上下文中检索信息（短期、动态）
▎ FFN：从参数中检索信息（长期、固定）

举个例子：输入"法国的首都是"。注意力先把"法国""首都"的信息汇集到最后一个位置。然后，FFN 中某个神经元的键恰好匹配"某国 + 首都"这个模式，被强烈激活，于是把指向"巴黎"的值写入向量。

这不只是比喻。Meng et al. 2022（ROME）定位到事实知识主要存储在中间层的 FFN，并且只修改其中一个 FFN 的权重，就能把模型的"知识"改成"埃菲尔铁塔在罗马"。这也解释了为什么 FFN 占了 70% 的参数：参数量大致就是模型的记忆容量。

四、MiniMind 的实现：SwiGLU 门控 FFN

model/model_minimind.py:136-146：

```python
self.gate_proj = nn.Linear(768, 2432, bias=False)
self.up_proj   = nn.Linear(768, 2432, bias=False)
self.down_proj = nn.Linear(2432, 768, bias=False)

def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

$$
\text{FFN}(x)=W_{down}\big(,\text{SiLU}(W_{gate},x)\ \odot\ W_{up},x,\big)
$$

和经典 FFN 相比，它有三个矩阵，多出来一个 gate_proj：
- up_proj(x)：计算"要写入什么内容"；
- SiLU(gate_proj(x))：一个 0 到 1 左右的门，逐个神经元决定"这条通路开多大"；
- 两者逐元素相乘（*），再由 down_proj 压回 768 维。

换成键值记忆的视角：经典 FFN 里，一条记忆是否被激活和激活多强，都由同一个 $k_i\cdot x$ 决定；SwiGLU 则把这两件事拆给了两组参数，"是否激活"由门决定，"写入多少"由 up 决定，表达能力更强。这种结构来自 Shazeer 2020（GLU Variants Improve Transformer），LLaMA、Qwen、DeepSeek 都在用。

为什么中间维度是 2432？ 配置里写的是（:26）：
intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64   # 768×π ≈ 2412.7 → 向上取到 64 的倍数 2432

- 经典 FFN 用 2 个矩阵，中间维度 4d，参数量 $8d^2$；
- SwiGLU 有 3 个矩阵，如果要保持参数量相同，中间维度应为 $\tfrac{8}{3}d\approx 2.67d$（LLaMA 的做法）；
- MiniMind 用 $\pi d\approx 3.14d$，比 LLaMA 稍宽一些，给记忆库多留了一点容量；
- 取 64 的倍数，是为了让 GPU 的矩阵乘法对齐，计算更高效。

SiLU 激活函数：$\text{SiLU}(z)=z\cdot\text{sigmoid}(z)$。它比 ReLU 更平滑：负半轴不是一刀切为 0，梯度更连续，训练更稳定。

五、MoE：把一个大 FFN 拆成多个"专家"

理解了"FFN 是记忆库"，就能理解 MiniMind 的 MoE 版本（MOEFeedForward，:148）为什么只替换 FFN，不动注意力：

scores = F.softmax(self.gate(x_flat), dim=-1)               # 路由器：给 4 个专家打分
topk_weight, topk_idx = torch.topk(scores, k=1, dim=-1)     # 每个 token 只选 1 个专家

- 把 FFN 复制成 4 个专家（num_experts=4），每个 token 由路由器挑选 1 个专家来处理（num_experts_per_tok=1）；
- 总记忆容量约为 4 倍，但每个 token 只经过 1 个专家，计算量基本不变；
- 这之所以可行，是因为 FFN 本来就是逐 token 独立计算的，不同 token 走不同的专家，互不影响。注意力需要跨 token 交互，就没法这样拆。
- aux_loss（:176-178）是负载均衡损失，防止所有 token 都挤到同一个专家，导致其他专家闲置。

六、总结

▎ FFN 是 Transformer 中逐 token 独立运行的"记忆库 + 非线性加工器"。
▎ 注意力负责从上下文中收集信息，FFN 负责用参数里存储的知识去加工这些信息：用 $W_1$ 的行去匹配模式，用 $W_2$ 的列写入知识。
▎ 它占了 MiniMind 70% 的参数，是模型记忆容量的主体。MiniMind 用 SwiGLU 门控结构提升了表达能力，并在 MoE 版本中把它扩展为多个专家，在计算量基本不变的情况下扩大了记忆容量。

一个形象的说法：注意力让每个词**"知道句子里还有什么"，FFN 让每个词"想起自己学过的东西"**。一个 block 做一轮"交流 + 思考"，8 个 block 就迭代 8 轮，每一轮都在残差主干上逐步补充和修正信息。