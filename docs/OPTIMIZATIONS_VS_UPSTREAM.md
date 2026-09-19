# 相对 antirez/ds4 上游做了哪些优化

本仓库是 [antirez/ds4](https://github.com/antirez/ds4) 的 fork（`supermy/ds41f`）。
上游的主战场是 Metal（Apple Silicon）与 DGX Spark，而**这一支只做一件事：
把 CUDA 单卡上的 SSD 流式推理在 16 GiB 显存 + 96 GB 内存的消费级机器上推到带宽上限**。

所有优化都落在同一条路径上：`--cuda --ssd-streaming` 下路由专家（routed experts）的读取。
本文记录每一项优化的**目标、做法、实测幅度**，以及三个模型各自的完整提升链条。

- 每一步的推导、失败方案与调试过程：[OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md)
- 读透缓存的设计笔记与实现偏差：[PLAN-host-expert-cache.md](PLAN-host-expert-cache.md)
- 逐次改动的原始记录：`changelog.md`

## 测试环境

| 项 | 值 |
|---|---|
| GPU | RTX 5060 Ti 16 GiB（可用 15.48 GiB，sm_120） |
| 内存 | **96 GB 级**：`MemTotal` 97959540 kB ≈ **93.4 GiB**，`MemAvailable` 约 90 GiB |
| 存储 | NVMe（Fanxiang S910Pro 2TB），`/data` 为 ext4；裸盘 O_DIRECT 顺序读 8.7 GB/s |
| CUDA | 13.3 |
| 测量口径 | 同一 prompt、`--temp 0 -n 128`、`--nothink`，报告生成阶段的 t/s |

**基线怎么定的**：上游没有这里加的任何一项，所以基线用 env 把三件事全部关掉来近似
——`DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1`（退回串行 pread）+
`DS4_RAM_RESIDENT_EXPERTS=0`（不装常驻池）+ `DS4_HOST_EXPERT_CACHE=off`（不装读透缓存）。
**基线不是编译上游代码测出来的**，是同一份二进制关掉开关后的读数。

## 总览

| # | 优化 | 目标 | 做法 | 影响哪个模型 | 幅度（解码 t/s） |
|---|---|---|---|---|---|
| 1 | `make cuda` 自动探测架构 | 不再手动传 `CUDA_ARCH` | `nvidia-smi` 探测，失败回退 `native` | 构建，本机 sm_120 | — |
| 2 | 路由专家预取并行化（滑动窗口） | 把盘的实际供给榨出来 | 每 reader 一个 staging buffer/event/槽位，谁回来就立刻上拷并补下一个请求，队列深度默认 8 | **全部三个** | V4 3.71→7.14、V4.1 1.78→3.85、GLM 1.57→3.82 |
| 3 | `--ram-resident-experts` 全量常驻池 | 装得下就把盘踢出解码路径 | 启动时把全部路由专家 pread 进 pinned 主机内存，此后变成 H2D 拷贝 | **V4 Flash**（72.56 GiB 装得下） | V4 7.14→13.46 |
| 4 | `--host-offload-token-embd` | 腾显存 | `token_embd` 放进 pinned 主机内存，按行 gather 走 PCIe | 全部（默认关闭） | 显存 −1.2 GiB，速度不变 |
| 5 | 读透式主机专家缓存 | 装不下时的降级路径 | 读过的专家留在 pinned 内存，命中走 H2D；CLOCK 淘汰、引用计数 | **V4.1 / GLM** | V4.1 3.85→6.31、GLM 3.82→6.07 |
| 6 | `DS4_RAM_RESIDENT_RESERVE_MB` | 差几个 GiB 的也让它装下 | 覆盖主机预留（默认总内存 1/8） | **GLM**（只差 2.6 GiB） | GLM 6.07→7.20 |
| 7 | 专家槽调优 | 把显存槽推到卡的极限 | V4 Flash 用 960 槽（1200+ 会 OOM） | **V4 Flash** | V4 13.46→15.16 |

## 三个模型的完整提升链条

同一 prompt、`--temp 0 -n 128`：

### V4 Flash IQ2XXS（80.76 GiB，路由专家 72.56 GiB）

| 阶段 | 解码 t/s | prefill t/s | 相对上一步 |
|---|---|---|---|
| 上游基线（串行 + 无池 + 无缓存） | 3.71 | 2.34 | — |
| + 并行预取 | 7.14 | 4.38 | **1.93×** |
| + 全量常驻池（默认 512 槽 `-c 4096`） | 13.46 | 6.39 | **1.89×** |
| + 960 槽 `-c 2048`（本机最优） | **15.16** | 6.47 | **1.13×** |

**合计 3.71 → 15.16 = 4.09×**。落盘字节 132 → 79 GiB。
上限：每 token 1.83 GB ÷ 28.6 GB/s = 64 ms → 15.6 t/s，实测已到 **97%**，基本贴死 PCIe。

### V4.1 Flash Q2（340.60 GiB，路由专家 142.38 GiB）

| 阶段 | 解码 t/s | prefill t/s | 相对上一步 |
|---|---|---|---|
| 上游基线 | 1.78 | 2.60 | — |
| + 并行预取 | 3.85 | 5.89 | **2.16×** |
| + 读透缓存（默认 26 GiB 预算，41 槽） | **6.31** | 5.77 | **1.64×** |

**合计 1.78 → 6.31 = 3.54×**。落盘字节 317 → 89 GiB（**−72%**），命中率 73.7%。
142.38 GiB 的专家在本机**数学上装不下**（需要可用 158.4 GiB），所以只能走读透缓存这一档。
按 PCIe 推算它的天花板是 10.8 t/s，**未验证**（本机做不到）。

### GLM 5.3 Flash Q2（89.88 GiB，路由专家 81.63 GiB）

模型参数：46 层、每层 288 专家、每 token 约 2.61 GB（46 层中 43 层是均匀层，可进常驻池）。
全部四档都实测过，同一 prompt、`--temp 0 -n 128`：

| 阶段 | 解码 t/s | prefill t/s | 落盘 GiB | 命中率 | 相对上一步 |
|---|---|---|---|---|---|
| 上游基线 | 1.57 | 2.64 | 310 | — | — |
| + 并行预取 | 3.82 | 5.58 | 310 | — | **2.43×** |
| + 读透缓存（默认 26 GiB 预算） | 6.07 | 5.43 | 98 | 70.2% | **1.59×** |
| + 压低预留使其全量常驻（`RESERVE_MB=6144`） | **7.20** | 7.51 | 89 | — | **1.19×** |

**合计 1.57 → 7.20 = 4.59×**，落盘字节 310 → 89 GiB（−71%）。（重复运行落在 6.06–6.07，
噪声约 ±0.02。）

三个从数字里读出来的事实：

1. **并行预取那一档落盘字节一点没少**（310 GiB 和基线相同），速度却 2.43×。
   这跟 V4.1 上看到的是同一件事：瓶颈先是**读队列深度**，不是盘的带宽。
   盘一直在以同样的字节量供给，只是不再空等。
2. **读透缓存削掉的是字节**（310 → 98 GiB，−68%），命中率 70.2%，
   212.63 GiB 从内存供给、90.18 GiB 仍走盘。
3. **全常驻后剩下的 89 GiB 里绝大部分是启动那次预载**（81.63 GiB，11.8 s @ 7.5 GB/s），
   解码阶段几乎不再碰盘。

**为什么全常驻只到 7.20，而不是按 PCIe 算的 11.0**：`2.61 GB ÷ 28.6 GB/s = 91 ms`
只算了搬字节，实测 139 ms/token，多出的 48 ms 有据可查——

- 46 层里只有 **43 层进池**，3 层是混合精度层，不进 slab 缓存也就进不了池，
  它们的流量（约 6.5% ≈ 0.17 GB/token）仍要走盘，约 17 ms。
- GLM 走 `full-attention argmax generation path`（日志里写明），不是 DeepSeek 的压缩 KV 路径，
  attention 侧的 GPU 开销更大，约 30 ms。

也就是说**盘踢出去之后，GLM 的瓶颈是它自己的 attention 计算**，不是搬运。
同样的全常驻待遇，V4 Flash 能贴到 PCIe 的 97%，GLM 只到 65%。

**复现**（GLM 在本机有两个坑，见下）：

```sh
G=<GLM53-Q2.gguf>; P="Explain what mmap is, briefly."
BASE='DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1 DS4_RAM_RESIDENT_EXPERTS=0 DS4_HOST_EXPERT_CACHE=off'
PAR='DS4_RAM_RESIDENT_EXPERTS=0 DS4_HOST_EXPERT_CACHE=off'
COMMON=(-m "$G" --cuda --ssd-streaming -c 2048 --nothink --temp 0 -n 128 -p "$P")

bash -c "$BASE DS4_GLM_MEMORY_GUARD=0 ./ds4 \"\${@}\"" _ "${COMMON[@]}"   # 1.57
bash -c "$PAR  DS4_GLM_MEMORY_GUARD=0 ./ds4 \"\${@}\"" _ "${COMMON[@]}"   # 3.82
DS4_GLM_MEMORY_GUARD=0 ./ds4 "${COMMON[@]}"                              # 6.07
DS4_GLM_MEMORY_GUARD=0 DS4_RAM_RESIDENT_RESERVE_MB=6144 ./ds4 "${COMMON[@]}"   # 7.20
```

**GLM 在本机的两个坑**

- 不加 `DS4_GLM_MEMORY_GUARD=0` 会被守卫拒掉：它要预留 32 GiB，16 GiB 卡上预算直接算成 0
  （`GLM memory guard refused ctx=2048 ... reserve 32.00 GiB`）。
- **不接受 `--prefill-chunk`**（"GLM uses graph-selected prefill chunks"）；`-c` 大小对速度无影响
  （512 与 2048 都是 6.06–6.09 t/s）。

**压预留的代价**：pin 完 81.63 GiB 后系统只剩约 9 GiB 给页缓存和其他进程。
`ds4` 把自己的 `oom_score_adj` 设成 1000，真到 OOM 时先被杀的是它。
预留 6 GiB 与 8 GiB 都能装下且速度相同（都是 7.20 t/s），具体留多少看机器上还要跑什么。

## 各项优化的目标与为什么有效

### 1. 专家预取并行化：瓶颈先是队列深度，不是盘带宽

**目标**：每 token 要搬 40 层 × 6 专家 × 9.49 MiB ≈ 2.4 GiB，而单个 `pread` 只有约 5 GB/s，
整条 token 时间被单个 IO 占满。

**做法**：分批栅栏改成滑动窗口——谁回来就立刻上拷、立刻补下一个请求，读队列始终填满。
默认并发 8（扫过 4–16，8 是峰值）。

**修正了一个判断**：原先以为"NVMe 5 GB/s 即墙"。实测单个 pread 约 5 GB/s，
但多个重叠 pread 在同一块盘上能到约 13 GB/s——**真正的墙是队列深度**，深度给够才碰到设备带宽。

### 2. 全量常驻池：把盘整个踢出去

**目标**：V4 Flash 的 72.56 GiB 路由专家装得下 90 GiB 的可用内存，那就别每 token 读一遍。

**做法**：按文件偏移排序、8 MiB 块、16 线程并行 pread 一次性预载（实测 7.8 GB/s，
达裸盘的 90%）；此后这些层的"读取"是 pinned→显存的 `cudaMemcpyAsync`。
语义是**全量或不装**——不可换出的大块内存换不可预测的命中率不划算。

**两条读路径都必须挂钩**：前台逐专家的 `cuda_stream_copy_worker` 和后台整层预取，
漏掉任何一处都白改。

### 3. 读透缓存：装不下时的降级

**目标**：解决"装不下就完全没有加速"的断层。

**动手前先做了零代码对照实验**（用内核页缓存冒充读透缓存）：同一 prompt 从 3.81 → 6.54 t/s，
而落盘读为 **0**。判据是"提升 ≥20% 才写这段代码"，实测 **+72%**，证明专家访问高度偏斜——
每 token 读 2.4 GiB，但 128 token 读掉的 316 GiB 里唯一字节远少于此。

**做法**：三个 arena（gate/up/down）按文件偏移索引，CLOCK 淘汰，命中取引用、
在该批拷贝落地后统一释放。填充只发生在"一个 token 自己的路由"这一档——
判据是**请求里的专家数 ≤ 8 而不是入口函数**，因为 DeepSeek 的 decode 与 prefill
走的是同一个 `begin_selected_load`（实测 slot 数：6×1240 次是解码，12/48 是 prefill），
按入口区分会让 prefill 把后面 token 要用的热专家冲掉。

**默认预算不是能吃多少吃多少**：预算扫描 8/16/32/78 GiB → 5.23/6.23/6.30/6.17 t/s，
32 GiB 已 75.7% 命中，78 GiB 只多 1.3 个百分点反而略慢（更多 pinned 内存挤压页缓存）。
所以默认取可用内存的 1/3，上限 32 GiB。

### 4. `DS4_RAM_RESIDENT_RESERVE_MB`：差一点点的也让它装下

主机预留是总内存的 1/8（本机 11.68 GiB），于是 GLM 需要 93.31 GiB 而本机只有约 90 GiB。
pinned 内存不可换出，但**预留该留多少是用户的取舍**：压到 6 GiB 后
usable = 84.7 GiB > 81.63 GiB，GLM 就全常驻了（6.07 → 7.20 t/s）。
代价是 pin 完系统只剩约 9 GiB；`ds4` 把自己的 `oom_score_adj` 设成 1000，OOM 时先杀它。

### 5. `--host-offload-token-embd`：唯一没带来加速的一项

腾出约 1.2 GiB 显存，速度不变（噪声内），默认关闭。记录它是因为结论有用：
**只有按行/列 gather 的张量适合卸载**；`engram_kv` 这种每层 300 MiB 的稠密投影，
卸载等于每 token 多走约 600 MiB PCIe。而且省下的 1.2 GiB **不会变成更多专家槽**
（规划器按显存总量算槽位），实际价值是上下文/prefill/OOM 余量。

## 评估后否决的方向

| 方向 | 关键数字 | 结论 |
|---|---|---|
| SSD→内存跨层预取 | 解码已在盘带宽上限（9.07 GB/s）；跨层预读放大 62 倍 | 增益 ≈ 0 |
| MoE 放 CPU | ds4 CPU 后端 0.06 t/s；内存带宽实测 34 GB/s | 至多 2 倍，要写 llama.cpp 级内核 |
| 权重无损压缩 | iq2_xxs 熵 7.94（上限 1.01x） | GPU 熵解码比不压更慢 |
| 扩大专家显存缓存 | 工作集约 15360 专家，逐 token 复用 ≈ 0 | 无收益 |
| 更大的读队列深度 | 16/24/32 → 3.78/2.98/2.96 t/s | 已到平台期 |

完整推导见 [OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md) 第 5、6 节。

## 下一步目标

1. **批处理解码**：同批 token 复用同一层专家权重（batch 8 时每 token 专家数 6 → 5.5）。
   这是 V4 Flash 现在唯一的路——它已经贴死 PCIe，只能削字节。
2. **更小检查点**：IQ2XXS 相对 Q2 每 token 流量少约 7.5 倍，始终是最直接的杠杆。
3. **把读透缓存接到 prefill**：目前 prefill 只查不填，长 prompt 仍全额落盘读。
   动工前要先量清楚它会不会把解码的热专家冲掉。

## 复现这些数字

```sh
make cuda && make tests/test_cuda_ssd_cache    # 改完 ds4_cuda.cu 必须单独重建测试
P="Explain what mmap is, briefly."
BASE='DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1 DS4_RAM_RESIDENT_EXPERTS=0 DS4_HOST_EXPERT_CACHE=off'
PAR='DS4_RAM_RESIDENT_EXPERTS=0 DS4_HOST_EXPERT_CACHE=off'

# V4 Flash IQ2XXS：3.71 → 7.14 → 13.46 → 15.16
bash -c "$BASE ./ds4 -m <IQ2XXS> --cuda --ssd-streaming --ssd-streaming-cache-experts 512 -c 4096 --prefill-chunk 512 --nothink --temp 0 -n 128 -p \"$P\""
bash -c "$PAR  ./ds4 -m <IQ2XXS> ...同上..."
./ds4 -m <IQ2XXS> ...同上...                                    # 默认：常驻池 + 512 槽
./ds4 -m <IQ2XXS> --cuda --ssd-streaming --ssd-streaming-cache-experts 960 -c 2048 ...   # 最优

# V4.1 Flash Q2：1.78 → 3.85 → 6.31（41 是槽位数，不是 GB）
bash -c "$BASE ./ds4 -m <V41Q2> --cuda --ssd-streaming --ssd-streaming-cache-experts 41 -c 2048 --prefill-chunk 512 --nothink --temp 0 -n 128 -p \"$P\""
bash -c "$PAR  ./ds4 -m <V41Q2> ...同上..."
./ds4 -m <V41Q2> ...同上...

# GLM 5.3 Flash Q2：1.57 → 3.82 → 6.07 → 7.20（必须绕守卫，且不能给 --prefill-chunk）
bash -c "$BASE DS4_GLM_MEMORY_GUARD=0 ./ds4 -m <GLM53> --cuda --ssd-streaming -c 2048 --nothink --temp 0 -n 128 -p \"$P\""
bash -c "$PAR  DS4_GLM_MEMORY_GUARD=0 ./ds4 -m <GLM53> ...同上..."
DS4_GLM_MEMORY_GUARD=0 ./ds4 -m <GLM53> ...同上...
DS4_GLM_MEMORY_GUARD=0 DS4_RAM_RESIDENT_RESERVE_MB=6144 ./ds4 -m <GLM53> ...同上...
```

判据与坑：

- 只有 `--temp 0` 的前后对比才有意义（默认采样不确定）。
- 同时采样 `/proc/<pid>/io` 的 `read_bytes` 得到真实落盘字节，并用 md5 校验
  任何档位的输出都与基线**逐字一致**。
- `ds4` 有单实例锁（`/tmp/ds4.lock`），连着跑多组要用 `flock -n /tmp/ds4.lock` 先拿锁。
- 本机 PATH 里的 `env` 是不转发参数的 shim，环境变量一律用 shell 前缀赋值。
