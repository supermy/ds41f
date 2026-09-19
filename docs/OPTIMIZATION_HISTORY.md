# 历次优化记录与目标

本文档按时间顺序记录在 CUDA 单卡（SSD 流式）路径上做过的每一次优化：
**目标 → 做法 → 实测结果 → 结论**，以及一批"评估后被否决"的方案。
本机基线环境：RTX 5060 Ti（16 GiB，可用 15.48 GiB，sm_120）、93 GB 内存、
NVMe（Fanxiang S910Pro 2TB，`/data` 为 ext4 on nvme1n1p6）、CUDA 13.3。

文档索引

- **相对 antirez 上游做了哪些优化、逐模型的实测幅度**：
  [OPTIMIZATIONS_VS_UPSTREAM.md](OPTIMIZATIONS_VS_UPSTREAM.md)
- 读透式主机专家缓存的方案与偏差：[PLAN-host-expert-cache.md](PLAN-host-expert-cache.md)（已实现）
- 调优后的实用命令：仓库根目录 `优化步骤教程.md`、[中文 README](../README_CN.md) 的
  「SSD 流式调优」章节、[英文 README](../README.md) 的 "SSD streaming tuning" 章节
- 流式机制本身：[SSD_STREAMING.md](SSD_STREAMING.md)

## 0. 目标

**总目标**：在不改变输出正确性的前提下，把 V4 / V4.1 Flash 系列在「模型远大于显存、
专家绝大部分躺在盘上」这种配置下的解码吞吐，推到介质供给带宽允许的上限。

**当前目标清单（按优先级）**

1. **批处理解码**：同批 token 复用同一层的专家权重——GPU 侧纯吞吐提升，不需要写新内核
   （batch 8 时每 token 专家数 6 → 5.5，batch 32 → 4.25）。
2. **更小检查点**：IQ2XXS 相对 Q2 每 token 流量少约 7.5 倍，是最直接的杠杆。
3. **把读透缓存也接到 prefill**：目前 prefill 只查不填（第 4 节），
   长 prompt 的 prefill 仍在全额落盘读；若要动，先量清楚它会不会把解码的热专家冲掉。

**已完成并固化的目标**

- 串行 pread → 滑动窗口并发预取：V4.1 Q2 解码 1.93 → 3.80 t/s（第 1 节）。
- 每 token 落盘读 → pinned 主机内存直拷：V4 Flash IQ2XXS 解码 7.22 → 13.02 → **15.19 t/s**
  （装上池之后再把专家槽推到 960，第 3 节与第 9 节）。
- 装不进内存时的读透式主机缓存：V4.1 Q2 解码 3.85 → 6.30 t/s，GLM 5.3 Flash 3.82 → 6.06 t/s
  （第 4 节与第 9 节）。
- 压低主机预留让 GLM 5.3 也全量常驻：6.06 → **7.20 t/s**（第 9 节）。

**三条路径现在是同一个天花板**：V4 Flash 已贴住 PCIe（15.19 / 15.6 = 97%），
V4.1 与 GLM 的有效带宽只有它的一半（数据要绕"盘→staging→H2D"两趟而不是一趟
pinned→device）。剩下能动的只有两件事：削字节（batch 解码 / 更小检查点），
或者加内存让后两者也走全常驻（GLM 只差 3.2 GiB，第 9 节）。

**下一个目标**（[计划文档](PLAN-host-expert-cache.md) 已转为实现记录）：批处理解码，
同批 token 复用同一层的专家权重；以及更小检查点这一最直接的杠杆。

## 一句话结论（判据）

流式解码的吞吐 ≈ **每 token 必须穿过的字节数 ÷ 供给带宽**。
所以：

1. 先算 `每 token 字节 × 实测 t/s`，看是否逼近盘带宽；
   逼近了就说明该**削字节**（更小检查点 / 批处理摊薄 / 命中更快的介质），
   而不是加管道（更大并发、更深的预取）。
2. 也不要指望"<｜hy_place▁holder▁no▁813｜>计算位置"：把 MoE 挪到 CPU 在 decoded 稳态是净亏损
   （见第 6 节），因为没有一个新的字节来源。

## 时间线总览

| 日期 | 目标 | 手段 | 结果（解码 t/s） | 状态 |
|---|---|---|---|---|
| 2026-09-17 | 把 SSD 专家读取跑满 | 路由专家预取并行化 + 滑动窗口补位 | V4.1 Q2：1.93 → **3.80** | 已实现 |
| 2026-09-17 | （评估）能否压缩权重省流量 | Q2 张量熵 / 编解码器评估 | 最多省 ~7%，GPU 解压反而更慢 | 否决 |
| 2026-09-17 | （评估）MoE 放 CPU 计算 | ds4 CPU 后端 / llama.cpp / 带宽实测 | CPU 后端 0.06 t/s，上限约 2 倍 | 否决 |
| 2026-09-18 | 腾出显存 | `--host-offload-token-embd` | 显存 −1.2 GiB，速度不变 | 已实现 |
| 2026-09-18 | 消灭整块落盘读 | `--ram-resident-experts`（全量常驻 pinned 池） | V4 Flash IQ2XXS：7.22 → **13.02** | 已实现 |
| 2026-09-19 | 覆盖「装不进内存」的模型 | 读透式主机专家缓存 | V4.1 Q2：3.85 → **6.30**；GLM 5.3：3.82 → **6.06** | 已实现 |
| 2026-09-19 | 把显存专家槽推到卡的极限 | 960 槽（1200+ 会 OOM） | V4 Flash IQ2XXS：13.55 → **15.19** | 已实现 |
| 2026-09-19 | 让 GLM 也全量常驻 | `DS4_RAM_RESIDENT_RESERVE_MB` 压低预留 | GLM 5.3：6.06 → **7.20** | 已实现 |
| 2026-09-18 | （评估）SSD→内存跨层预取 | 盘带宽 / 放大倍数核算 | 增益 ≈ 0（解码已在盘带宽上限） | 否决 |

## 1. 2026-09-17 — 路由专家预取并行化（V4.1 Flash Q2）

**目标**：每 token 要搬 40 层 × 6 专家 × 9.49 MiB = 2.28 GB，单个 `pread` 只有约 5 GB/s，
把整条 token 时间占满。想把盘的实际供给能力榨出来。

**做法**

- 每个 reader 一个 staging buffer / 完成事件 / 槽位，请求按 gate→up→down 逐张量连续下发。
- 分批栅栏改为**滑动窗口**：谁回来就立刻上拷并立刻补下一个请求，读队列始终填满。
- 默认并发 8（扫过 4/6/8/10/12/16，8 为峰值）；`DS4_CUDA_EXPERT_READ_DEPTH` 覆盖，上限 32；
  `DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1` 退回串行（**也是所有后续对比的基线**）。
- 缓存 reserve 从固定 8 GiB 改为随显存缩放（1/8 显存，夹在 1–8 GiB 且 ≤ 1/4 显存）。
- `make cuda` 未给 `CUDA_ARCH` 时用 `nvidia-smi` 自动探测（本机 sm_120）。

**实测**

| 专家暂存方式 | prefill t/s | 解码 t/s |
|---|---|---|
| 串行 pread（基线） | 2.78 | 1.93 |
| 分批栅栏 @16 | 4.43 | 3.21 |
| 滑动窗口 @8（当前默认） | **6.01** | **3.80** |

**结论**：机制判据被修正——单个 pread 约 5 GB/s，但多个重叠 pread 在同一块盘上可到约 13 GB/s；
真正的"墙"先是队列深度，深度给足后才碰到设备带宽。>8 并发不再有增益，更多并发只是多占 pinned 内存。

## 2. 2026-09-18 — `--host-offload-token-embd`

**目标**：token_embd（1.23 GiB）每 token 只做一次约 10 KB 的行 gather，却整块占着显存；
把它搬到 pinned 主机内存应能腾出约 1.2 GiB。

**做法**：显式 `cudaHostAlloc` + `cudaHostGetDevicePointer`（UVA 指针）。
本机对模型 mmap 做 `cudaHostRegister` 的 zero-copy 通道**静默失败**并回落成 `cudaMalloc`+拷贝
（`DS4_CUDA_NO_FD_CACHE=1` 时所有 range 都打印 `CUDA cached`，无 `CUDA mapped`），所以不能复用它。

**实测**：显存 7.86 → 6.87 GiB；解码 8.31 → 8.21（噪声内）。`--temp 0` 输出逐 token 一致。

**结论与边界**

- 只有**行/列 gather**的张量适合这样卸载；被 GEMV 整块读的权重不行（`engram_kv` 每层 300 MiB，
  卸载等于每 token 多走约 600 MiB PCIe，约耗 8% 的 token 预算）。
- 释放出来的 1.2 GiB **不会变成更多专家槽**（规划器按显存总量估算），实际价值是上下文/prefill/OOM 余量。

## 3. 2026-09-18 — `--ram-resident-experts`：内存常驻 pinned 专家池

**目标**：只要可用主机内存装得下**全部**路由专家权重，就把它们一次性 pread 进 pinned 主机内存池，
此后这些层的专家"读取"变成 pinned→显存的 `cudaMemcpyAsync`，完全不碰盘。

**做法**

- 判断口径：`MemAvailable − CmaFree` 减去预留（总内存 1/8，夹在 4..16 GiB）≥ 全部路由专家
  （gate/up/down）——**装不下就整个忽略**，不做部分常驻（不可换出的大块内存换不可预测的命中率不划算）。
- 池按文件偏移索引（排序 + 二分），独立于 `g_host_ranges`；分配用 `cudaHostAlloc`
  **不带 `cudaHostAllocMapped`**（否则产生 device 别名，内核会逐字节走 PCIe）。
- 预载按偏移排序、8 MiB 块、16 线程并行 pread；GGUF 专家张量偏移只有 512 字节对齐，
  未对齐的头/尾必须走 page cache，否则 O_DIRECT 全部失效（这一步把预载从 6.5 提到 7.7 GB/s）。
- 挂钩点：`cuda_stream_copy_worker`（前台逐专家，命中 → `payload` 直接给出，跳过 pread）
  与 `cuda_stream_prefetch_read`（后台整层预取，命中 → 直拷 H2D）。

**实测**（同一 prompt，`--temp 0 -n 96`）

| 模型 / 配置 | 解码 t/s |
|---|---|
| V4 Flash IQ2XXS，512 槽，池关闭 | 7.22 |
| V4 Flash IQ2XXS，512 槽，池自动启用 | **13.02** |
| V4.1 Flash Q2，41 槽（专家 142.38 GiB 装不下，忽略） | 3.86（与关闭时一致） |

预载 72.56 GiB 用 10.0 s（7.7 GB/s），已达裸盘 O_DIRECT 顺序读 8.7 GB/s 的约 90%；
线程数 8/16/32 → 11.9 / 10.0 / 10.3 s，默认 16。理论上限 17 t/s（H2D 28.6 GB/s），实测 13.02。

**勘误**：早期的 changelog 记录称"后台整层预取是解码速度的主要来源"是错的。
`ds4_gpu_stream_expert_cache_prefetch` 只在 **prefill** 中被调用，且要求 `total_count >= 2048`
（`ds4.c:41713`，注释写明低于 2K 时无用读取超过收益）；解码真正受益的是
`cuda_stream_copy_worker` 里的 host-pool 命中直拷。

## 4. 2026-09-19 — 读透式主机专家缓存（V4.1 Flash Q2）

**目标**：`--ram-resident-experts` 是「全量或不装」，Q2 的 142.38 GiB 路由专家装不进约 78 GiB
可用内存，于是被整包跳过、停在 3.85 t/s。让这一档降级为读透缓存：读过的专家留在 pinned 内存，
后续命中用 H2D 代替 pread。

**第 0 步：先证明"复用"真的存在（零代码，用内核页缓存冒充）**

```sh
# 本机 PATH 里的 env 是不转发参数的 shim，环境变量一律用 shell 前缀赋值
export DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1
./ds4 -m <Q2.gguf> --cuda --ssd-streaming --ssd-streaming-cache-experts 41 \
      -c 2048 --prefill-chunk 512 --nothink --temp 0 -n 128 -p "<同一 prompt>"
# 同时后台采样 /proc/<pid>/io 的 read_bytes → 每 token 实际落盘字节
```

| 组 | 落盘读 | prefill t/s | 解码 t/s |
|---|---|---|---|
| A 当前行为（O_DIRECT + 丢弃模型页） | 316.4 GiB | 5.90 | 3.81 |
| B1 冷页缓存第一遍 | 65.3 GiB | 4.55 | 4.97 |
| B2 / B3 页缓存已热 = 复用上限 | 0.0 GiB | 10.33 | **6.54** |

B2 相对 A **+72%**（判据是 ≥20% 才动手）。关键数字：每 token 约 2.4 GiB，但 128 token 读掉的
316 GiB 里唯一字节远少于此——**专家访问是高度偏斜的**，不是均匀分布。

**做法**

- gate/up/down 各一个 arena，条目按文件偏移索引，`cudaHostAlloc` 按约 1 GiB 分块
  （几万次小 pinned 分配会先撞 `vm.max_map_count`），CLOCK 二次机会淘汰。
- 命中取引用，引用在**唯一无条件执行点**释放：前台路径是
  `cuda_stream_copy_requests` 末尾那次 `cudaStreamSynchronize` 之后（含中止批次），
  预读路径是 `prefetch_finish` 里 join 之后。漏收一处引用，条目就永久不可淘汰，
  缓存会慢慢退化成只读。
- 填充先独占条目（`state=filling`）→ 锁外 memcpy → 再发布，并发查到的必是完整字节。
- **填充只发生在"一个 token 自己的路由"这一档**，判据是请求里的专家数 ≤ 8
  （`DS4_HOST_EXPERT_CACHE_SLOTS` 可调），不是入口函数：DeepSeek 的 decode 与 prefill 走同一个
  `begin_selected_load`，实测 slot 数分布为 6 × 1240 次（decode）与 12/48（prefill）。
  prefill 一次扫上千专家，让它填充会把后面 token 要用的热专家冲掉。
- 预读（整层 look-ahead）只查不填：look-ahead 不是 reuse 的证据。

**实测**（同一 prompt、`--temp 0 -n 128`，Q2，41 槽）

| 缓存预算 | 落盘读 | 解码 t/s |
|---|---|---|
| 关 | 316 GiB | 3.86 / 3.84 |
| 8 GiB | 168 GiB | 5.23 |
| 16 GiB | 118 GiB | 6.23 |
| 32 GiB | 84 GiB | 6.30 |
| 78 GiB（可用内存全给） | 80 GiB | 6.17 |
| **默认：可用内存的 1/3 = 26.18 GiB** | 89 GiB | **6.29 / 6.32** |

默认预算**不是**能吃多少吃多少：曲线 16–32 GiB 到平台，32 GiB 已 75.7% 命中，
78 GiB 只多 1.3 个百分点反而略慢（更多 pinned 内存挤压页缓存）。命中率 73.7% 时
227 GiB 来自内存、81 GiB 仍走盘。四次运行（开/关各两遍）的正文 md5 逐字一致。

**结论**：这条路径的价值全在"削字节"而不是"加管道"——落盘字节 316 → 89 GiB（−72%），
解码 3.85 → 6.30 t/s（+64%）。剩下 81 GiB 仍走盘，是因为 26 GiB 装不下全部热专家；
继续加大预算收益极小，要再上一个台阶得靠更小检查点或批处理。

## 5. 2026-09-18 —（评估否决）SSD→内存跨层异步预取

提议："GPU 计算时把路由权重先搬到主机内存"。核算：

- Q2 每 token 2.39 GB，解码 3.80 t/s → **9.07 GB/s**，已等于本盘实际供给（裸盘 8.7 GB/s，池预载 7.7 GB/s）。
  盘没有空转时间 → 提前搬不会产生新字节，收益 ±5% 以内。
- GPU 每 token 实际计算仅约 20 ms（7.07 GiB 常驻 @ ~450 GB/s ≈ 16 ms），对比 263 ms 的 token 时间 → 瓶颈 100% 在盘。
- 跨层预取不可行：L+1 的 top-6 依赖 L 的残差输出；整层预读是 **62 倍放大**（3.56 GiB/层 vs 需要 57 MiB），
  一层就要约 420 ms > 263 ms 的 token 预算。
- 已佐证：读深度 16/24/32 → 3.78 / 2.98 / 2.96 t/s，深度不是瓶颈。

## 6. 被评估否决的其他方向

| 方向 | 关键数字 | 结论 |
|---|---|---|
| MoE 放 CPU（llama.cpp 风格） | ds4 CPU 后端同模型 0.06 t/s；内存带宽实测 34 GB/s；现实上限约 9–11 t/s（vs 当前 4.76） | 至多 2 倍，要写 llama.cpp 级内核；保留 SSD 读取只挪计算是净亏损（210 ms → 260–290 ms/token） |
| 权重复化/无损压缩 | iq2_xxs 熵 7.94（上限 1.01x）、q2_k 7.81（1.03x）、q8_0 7.68（1.05x）；全部常驻非路由压到熵极限只省约 1 GiB ≈ +0.6% 流量 | 否决；GPU 熵解码几十 GB/s vs HBM 450 GB/s，比不压更慢 |
| 扩大专家显存缓存 | 工作集约 15360 个专家，逐 token 复用率 ≈ 0 | 无收益 |
| `--mtp` / `--dspark` | V4.1 CUDA 直接拒绝 | 不适用 |
| 更大的并发/更深的暂存队列 | 见第 1 节与第 5 节 | 已到平台期，只是多占 pinned 内存 |

## 7. 下一步目标

见第 0 节的目标清单。第 0 节里凡是标着「已实现」的，对应的就是
[PLAN-host-expert-cache.md](PLAN-host-expert-cache.md) 这份计划文档，它已转为实现记录。

## 8. 实用命令速查

```sh
make cuda                                   # 不给 CUDA_ARCH 时自动探测（本机 sm_120）
make tests/test_cuda_ssd_cache              # 改了 ds4_cuda.cu 必须单独重建测试，否则在跑旧代码

# V4 Flash IQ2XXS：本机最快 15.19 t/s（960 槽 -c 2048；长 prompt 退回 512 槽 -c 4096）
./ds4 --cuda -m <IQ2XXS.gguf> --ssd-streaming --ssd-streaming-cache-experts 960 \
      -c 2048 --prefill-chunk 512 --nothink --temp 0 -n 128 -p "<prompt>"

# V4.1 Flash Q2：6.30 t/s（41 是槽位数，不是 GB）
./ds4 --cuda -m <Q2.gguf> --ssd-streaming --ssd-streaming-cache-experts 41 \
      -c 2048 --prefill-chunk 512 --nothink --temp 0 -n 128 -p "<prompt>"

# GLM 5.3 Flash Q2：7.20 t/s（必须绕开守卫，且不能给 --prefill-chunk）
DS4_GLM_MEMORY_GUARD=0 DS4_RAM_RESIDENT_RESERVE_MB=6144 ./ds4 --cuda -m <GLM53-Q2.gguf> \
      --ssd-streaming -c 2048 --nothink --temp 0 -n 128 -p "<prompt>"

# 三模型横评、槽数扫描与上限核算见第 9 节

# 限制/关闭内存常驻专家池与读透缓存
DS4_RAM_RESIDENT_EXPERTS=0 ./ds4 --cuda ...          # 等价于 --ram-resident-experts off
./ds4 --cuda ... --ram-resident-experts 64GB          # 给池/缓存设上限（低于可用额度时才生效）

# 读透式主机专家缓存（路由专家装不进内存时自动启用）
DS4_HOST_EXPERT_CACHE=off ./ds4 --cuda ...            # 关掉，退回纯 SSD
DS4_HOST_EXPERT_CACHE=48 ./ds4 --cuda ...             # 预算 48 GiB（不能超出可用额度）
DS4_HOST_EXPERT_CACHE_STATS=1 ./ds4 --cuda ...        # 退出时打印命中率与内存/磁盘字节
DS4_HOST_EXPERT_CACHE_SLOTS=8 ./ds4 --cuda ...        # 允许填充的"单 token 路由专家数"上限

# 把 token_embd 挪到 pinned 主机内存，腾出约 1.2 GiB 显存（行 gather 类张量专用）
./ds4 --cuda ... --host-offload-token-embd

# 串行基线（任何改动都跟它比）
DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1 ./ds4 --cuda ...
# 扫一次读队列深度，确认本机峰值（本机 8）
DS4_CUDA_EXPERT_READ_DEPTH=8 ./ds4 --cuda ...
```

**已知坑**

- `--ssd-streaming-cache-experts` 写 `NGB` 会被解析成"再预留两个完整 prefill 层"，
  结果只剩 1 个槽并报 `CUDA SSD cache cannot stage ... experts with system headroom`；要写**槽位数**。
- 长 prompt 会为"每层 × 每个唯一路由专家"占槽：IQ2XXS 上 799 token 的 prompt 最多只能用 512 槽，
  960 槽会 `gpu layer 33 ffn batch encode failed`；短 prompt 才能推到 960。
  （9.2 t/s 是常驻池之前的旧上限；装上池之后 960 槽是 15.19 t/s，见第 9 节。）
- 默认采样不确定：只有 `--temp 0` 的前后对比才有意义。
- `tests/test_cuda_ssd_cache` 在预取池出错时表现为**卡死**而不是报错，约 5 秒的回归比基准更早发现问题。
- `ds4` 有单实例锁（`/tmp/ds4.lock`），脚本里连着跑多组要用 `flock -n /tmp/ds4.lock` 先拿到锁。

## 9. 2026-09-19 — 三个模型的横评与上限

**问法**：同一台机器、同一 prompt、`--temp 0 -n 128`，三个模型各自能跑多快，瓶颈分别在哪。

**实测**（RTX 5060 Ti 16 GiB，93 GiB 内存）

| 模型 | GGUF | 路由专家 | 走的路径 | 缓存 off | **最快** | 每 token 字节 |
|---|---|---|---|---|---|---|
| V4 Flash IQ2XXS | 80.76 GiB | 72.56 GiB | 全量常驻池 | — | **15.19 t/s** | 1.83 GB |
| V4.1 Flash Q2 | 340.60 GiB | 142.38 GiB | 读透缓存（装不下） | 3.85 | **6.30 t/s** | 2.65 GB |
| GLM 5.3 Flash Q2 | 89.88 GiB | 81.63 GiB | 读透缓存 → **全常驻**（见下） | 3.82 | **7.20 t/s** | 2.61 GB |

GLM 的两档都实测过：默认预留下它差 2.6 GiB 装不下，走读透缓存是 6.06 t/s；
把预留压到 6 GiB（`DS4_RAM_RESIDENT_RESERVE_MB=6144`）后 81.63 GiB 全量 pin 进主机内存，
7.20 t/s，prefill 也从 5.67 提到 7.82 t/s。

**V4 Flash 的专家槽扫描**（常驻池已装下全部 72.56 GiB，盘已出局）

| 槽数 | 显存 | 解码 t/s |
|---|---|---|
| 256 | 1.69 GiB | 11.76 |
| 512 | 3.38 GiB | 13.55 |
| **960** | 6.32 GiB | **15.19** |
| 1200 / 1500 / 2000 | — | OOM（`q8_hc_expand` arena 分配失败） |

槽越多越快：相邻层和相邻 token 会选到同一批专家，槽里已有的就不必再走一次 PCIe。
960 是这张卡的极限。

**上限核算**：每 token 字节 ÷ 搬运带宽

| 模型 | 每 token | 实测 | 有效带宽 | 全常驻时的 PCIe 天花板 |
|---|---|---|---|---|
| V4 Flash | 1.83 GB | 15.19 t/s | **27.8 GB/s** | 15.6 t/s（已达 97%） |
| V4.1 Flash | 2.65 GB | 6.30 t/s（读透） | 15.8 GB/s | 10.8 t/s |
| GLM 5.3 Flash | 2.61 GB | 6.06（读透）/ **7.20（全常驻）** t/s | 15.8 / 18.8 GB/s | 11.0 t/s（**实测只到 7.20**，见下） |

- **V4 Flash 已贴死 PCIe**：`1.83 GB ÷ 28.6 GB/s = 64 ms` → 15.6 t/s，实测 15.19 是它的 97%。
  再快只能削字节（batch 解码）或加卡做 TP。
- **V4.1 / GLM 都只有 15.8 GB/s，而它们的落盘只剩 0.7–0.8 GiB/token（折算约 5 GB/s，
  远低于盘上限）——盘已经不是它们的瓶颈**。慢的原因是数据要绕两趟：
  未命中 pread→staging→H2D，命中 pinned→H2D，两条路串行叠加，有效带宽只有 PCIe 的一半。
- 所以差距不在模型，在内存：V4.1/GLM 每 token 字节只比 V4 Flash 多 45%，速度却只有 40%。

**GLM 全常驻实测 7.20 t/s，比 PCIe 天花板 11.0 低 3.8 t/s —— 之前"加内存就能到 11"的预测偏乐观**

`2.61 GB ÷ 28.6 GB/s = 91 ms` 只算了搬字节，实测是 139 ms/token，多出来的 48 ms 有据可查：

- 池里是 **43 of 46 层**，剩下 3 层是混合精度层，不进 slab 缓存也就进不了池，
  它们的每 token 流量（约 6.5% ≈ 0.17 GB）仍要走盘，约 17 ms。
- GLM 走的是 `full-attention argmax generation path`（日志里明确），
  不是 DeepSeek 的压缩 KV/MLA 路径，attention 侧的 GPU 开销更大，约 30 ms。

也就是说：**把盘踢出去之后，GLM 剩下的不是搬运瓶颈，是它自己的 attention 计算**。
同样的"全常驻"待遇，V4 Flash 能贴到 PCIe 的 97%，GLM 只能到 65%。
（V4.1 的 10.8 t/s 仍是纯推算，本机装不下，没法验证。）

**加内存能换到什么 / 本机还能怎么挤**

| 模型 | 全常驻所需可用内存 | 怎么做到 | 速度 |
|---|---|---|---|
| GLM 5.3 Flash | 81.63 GiB（+ 预留） | **本机就行**：把预留压到 6–8 GiB | 6.06 → **7.20 t/s**（实测） |
| GLM 5.3 Flash | 81.63 + 11.68 = 93.31 GiB | 换 128 GB 内存，用默认预留 | 7.20 t/s（实测，不会更高） |
| V4.1 Flash | 142.38 + 16 = 158.4 GiB | 192 GB 级 | 6.30 → ~10.8 t/s（**纯推算，未验证**） |

GLM 默认预留（总内存的 1/8 = 11.68 GiB）下需要 93.31 GiB 可用，而本机
`MemAvailable` 只有约 90 GiB，所以自动走了读透缓存。**但把预留压到 6 GiB 就够了**：
90.7 − 6 = 84.7 GiB > 81.63 GiB。新增 `DS4_RAM_RESIDENT_RESERVE_MB` 就是为了这个——
pinned 内存不可换出，预留该留多少是用户的取舍，不是引擎该替他决定的：

```sh
DS4_GLM_MEMORY_GUARD=0 DS4_RAM_RESIDENT_RESERVE_MB=6144 ./ds4 --cuda -m <GLM53-Q2.gguf> \
      --ssd-streaming -c 2048 --nothink --temp 0 -n 128 -p "<prompt>"
# → RAM-resident experts: 81.63 GiB resident in 11.8 s (7.5 GB/s)
#   43 of 46 routed layers pinned ... generation: 7.20 t/s
```

代价：pin 完 81.63 GiB 后系统只剩约 9 GiB 给页缓存和其他进程。
`ds4` 把自己的 `oom_score_adj` 设成 1000，真到 OOM 时先被杀的是它，不是桌面。
预留 8 GiB 也够（实测同为 7.20 t/s），留 6–8 GiB 之间按机器上还要跑什么来定。

**本机推荐参数**

```sh
# V4 Flash IQ2XXS：15.19 t/s（960 槽；要扛长 prompt 就退回 512 槽 + -c 4096）
./ds4 --cuda -m <IQ2XXS.gguf> --ssd-streaming --ssd-streaming-cache-experts 960 \
      -c 2048 --prefill-chunk 512 --nothink --temp 0 -n 128 -p "<prompt>"

# V4.1 Flash Q2：6.30 t/s（41 是槽位数，不是 GB）
./ds4 --cuda -m <V4.1-Q2.gguf> --ssd-streaming --ssd-streaming-cache-experts 41 \
      -c 2048 --prefill-chunk 512 --nothink --temp 0 -n 128 -p "<prompt>"

# GLM 5.3 Flash Q2：7.20 t/s（全常驻；不想动预留就去掉第二行，落回 6.06 t/s 的读透档）
DS4_GLM_MEMORY_GUARD=0 DS4_RAM_RESIDENT_RESERVE_MB=6144 ./ds4 --cuda -m <GLM53-Q2.gguf> \
      --ssd-streaming -c 2048 --nothink --temp 0 -n 128 -p "<prompt>"
```

**GLM 在本机的两个坑**

- 不加 `DS4_GLM_MEMORY_GUARD=0` 会被守卫拒掉：它要预留 32 GiB，16 GiB 卡上预算直接算成 0
  （`GLM memory guard refused ctx=2048 ... reserve 32.00 GiB`）。
- **不接受 `--prefill-chunk`**（"GLM uses graph-selected prefill chunks"）；`-c` 大小对速度无影响
  （512 与 2048 都是 6.06–6.09 t/s）。
