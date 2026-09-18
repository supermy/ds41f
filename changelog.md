# Changelog

## 2026-09-18 — 文档：调优使用方法进入 README，新增中文 README，并存档下一步优化计划

本次**没有改任何推理代码**，只动文档。

**改动**

- `README.md` 新增 "SSD streaming tuning (CUDA)" 一节：构建与重建测试的注意事项、
  两个模型的推荐命令、`--ssd-streaming-cache-experts` / `--ram-resident-experts` /
  `--host-offload-token-embd` 与 `DS4_CUDA_EXPERT_READ_DEPTH` / `DS4_EXPERT_POOL_READERS` /
  `DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ` 的取值与使用场景、RTX 5060 Ti 上的实测对照表、
  以及已知的坑。详细指南里补上优化记录与计划的入口，开头加中文文档索引。
- 新增 `README_CN.md`：README 的完整中文版，其中「SSD 流式调优」章节比英文版更细
  （多了实测带宽、一次性预载耗时、评判方法）。`README.md` 与中文文档互相指向。
- 新增 `docs/OPTIMIZATION_HISTORY.md`：历次优化的目标 → 做法 → 实测 → 结论，
  以及被评估否决的方向。新增第 0 节集中列出**总目标、当前目标清单、已完成并固化的目标**，
  原「下一步目标」一节改为指向第 0 节，避免两处各有一份清单而漂移。
- 根目录 `优化步骤教程.md`：第 8 节「剩下的杠杆」第一条改为指向已存档的读透缓存计划，
  并把「不要再试的」补写实到 SSD→内存跨层预取与 CPU MoE；新增第 9 节列出本机默认已开启的
  三项优化（滑动窗口并发、`--ram-resident-experts auto`、`--host-offload-token-embd`）
  及各自的实测收益。
- `docs/PLAN-host-expert-cache.md`（读透式主机专家缓存）**按"保存计划、先不实施"要求冻结存档**：
  头部写明状态、冻结日期、解冻条件，以及开工前必须先跑第 0 节零代码对照实验
  （B2 相对 A 提升 < 20% 就回来讨论）。**本次不含任何实现代码。**

**改动位置**：`README.md`、`README_CN.md`（新）、`docs/OPTIMIZATION_HISTORY.md`（新）、
`docs/PLAN-host-expert-cache.md`（新）、根目录 `优化步骤教程.md`（「剩下的杠杆」改为指向已存档计划，
并补一节「已经做完的」）、`changelog.md`。

**待办（未实施）**：读透式主机专家缓存本身，见上述计划文档。

---

## 2026-09-18 — 内存常驻路由专家池：内存装得下就一次性载入，专家读取不再走 SSD（CUDA）

**改动**

- 新开关 `--ram-resident-experts off|auto|NGB`（默认 `auto`，env `DS4_RAM_RESIDENT_EXPERTS=0` 可关）。启动时若可用主机内存（`MemAvailable − CmaFree`）减去预留（总内存 1/8，夹在 4..16 GiB）仍装得下**全部**路由专家权重（gate/up/down 三个张量），就把它们一次性 pread 进一块 pinned 主机内存池；此后这些层的专家"读取"变成 pinned→显存的 `cudaMemcpyAsync`，完全不碰磁盘。**装不下就整个忽略**（不做部分常驻：不可换出的大块内存换不可预测的命中率不划算），并打印原因。
- 池按文件偏移索引（排序 + 二分），独立于 `g_host_ranges`：后者会把命中的张量从显存 span 里剔除，专家绝不能被卷进去。分配用 `cudaHostAlloc`（**不带 `cudaHostAllocMapped`**，不产生 device 别名，避免内核逐字节走 PCIe）。
- 两条读路径都挂钩：`cuda_stream_copy_worker`（前台逐专家）命中池时直接给出 payload 跳过 pread；`cuda_stream_prefetch_read`（后台整层预取，一次 256/384 个专家）命中池时直接 H2D。**后者漏改就白改**——它是解码速度的主要来源且不经过前者的调度器。命中池时跳过 staging 相关的 event 等待与 `drop/discard pages`。
- 预载按文件偏移排序、8 MiB 块、16 个线程（`DS4_EXPERT_POOL_READERS` 可覆盖）并行 pread；GGUF 的专家张量偏移只到 512 字节对齐，所以未对齐的头/尾走 page cache、主体保持 O_DIRECT 合法，buffered 回退逐块 `posix_fadvise(DONTNEED)`。
- 启动日志新增：`ds4: RAM-resident experts: ...`（预载 GiB、耗时、GB/s、常驻层数）与 `ds4: expert cache: N slots x X MiB = Y GiB VRAM (Z% of ... cacheable experts)`——槽数与占用显存此前在启动摘要里完全没有。

**改动位置**：`ds4_cuda.cu`（池、查找、预载线程、两处挂钩、cleanup）、`ds4_gpu.h`、`ds4.c`（逐层字节统计 + 预算决策 + 触发 + 日志）、`ds4.h`、`ds4_cli.c`、`ds4_help.c`、`tests/test_cuda_ssd_cache.c`。

**实测（RTX 5060 Ti 16 GiB，93 GB 内存，同一 prompt，`--temp 0 -n 96`）**

| 模型 / 配置 | 解码 t/s |
|---|---|
| V4 Flash IQ2XXS，512 槽，池关闭 | 7.22 |
| V4 Flash IQ2XXS，512 槽，池自动启用 | **13.02** |
| V4.1 Flash Q2，41 槽（专家 142.38 GiB 装不下，自动忽略） | 3.86（与关闭时一致） |

- 预载 72.56 GiB 用 10.0 s（7.7 GB/s）；本机 `/data`（ext4 on nvme1n1p6）裸盘 O_DIRECT 顺序读实测 8.7 GB/s，即预载已达盘速约 90%。8/16/32 线程分别 11.9 / 10.0 / 10.3 s，默认取 16。
- `--temp 0` 下开启前后的 96 token 输出逐字一致。
- 960 槽 + 池与 960 槽 + 关闭都以 `gpu layer 33 ffn batch encode failed` 失败，属既有行为（大缓存 + `-c 4096` prefill 的 arena 不足），与本次改动无关；推荐仍是 512 槽。

**测试**：`tests/test_cuda_ssd_cache.c` 新增用例——建池后把源文件内容清空（`ftruncate`），按需加载与整层预取两条路径都必须复现参考输出，否则只能说明它其实读了盘。

---

## 2026-09-18 — `--host-offload-token-embd`：把 token_embd 放进 pinned 主机内存（CUDA）

**改动**

- 新开关 `--host-offload-token-embd`（仅 CUDA，仅 `ds4` CLI）：把 `token_embd.weight` 拷进 `cudaHostAlloc` 的 pinned 主机内存，用 `cudaHostGetDevicePointer` 得到的 UVA 指针作为内核可见指针。原本设想的路径是复用 `cuda_model_range_ptr` 里对模型 mmap 做 `cudaHostRegister` 的 zero-copy 通道，但实测它在本机静默失败并回落到 `cudaMalloc`+拷贝（`DS4_CUDA_NO_FD_CACHE=1` 时全部 range 都打印 `CUDA cached`，无一条 `CUDA mapped`），所以改成显式 pin + 拷贝。
- 该范围不再进入设备侧的 span 安装（`accelerator_prepare_model_tensor_spans` 跳过），也不再计入静态常驻字节（`weights_streaming_non_routed_bytes`）。

**改动位置**：`ds4_cuda.cu`（`g_host_ranges` + `ds4_gpu_set_host_resident_range` / `ds4_gpu_range_is_host_resident`，hook 在 `cuda_model_range_ptr` 与 `cuda_model_range_is_cached` 的最前面）、`ds4.c`、`ds4.h`、`ds4_cli.c`、`ds4_help.c`、`ds4_gpu.h`。

**实测（RTX 5060 Ti 16 GiB）**

| 模型 | 常驻在设备内存的张量范围 | 解码 t/s |
|---|---|---|
| V4 Flash IQ2XXS（`--ssd-streaming-cache-experts 8GB`，`-n 96`） | 7.86 → 6.87 GiB（−1010 MiB） | 8.31 → 8.21 |
| V4.1 Flash Q2（`--ssd-streaming-cache-experts 41`，`-n 32`） | 首 Span 1262.50 MiB 消失 | 3.75 → 3.79 |

`--temp 0` 下两个模型的输出逐 token 一致。解码差异落在同一配置重复运行的噪声之内。

**为什么不含 `engram_kv`**：它不是按行访问的表，而是每层 `[6144, 25600]` 的 F16 稠密投影（300 MiB/层，layer 1 与 14），decode 每 token 整块读完。真正在做 264 B 行 gather 的是 `engram_embd`（189 GiB，本来就只落在磁盘上）。把它卸载到主机内存相当于每 token 多走约 600 MiB PCIe（约 21 ms），在 Q2 的 263 ms/token 预算里约 −8%。

**局限**：释放出来的 1.2 GiB 在实测过的配置里**不会变成更多专家槽**（自动模式 6 槽、显式 8GB 目标 701 槽，开关前后都一样）——规划器按显存总量和请求目标算槽位，不看这段增量；它带来的实际余量是上下文/prefill/OOM 空间。

---

## 2026-09-17 — CUDA 路由专家预取并行化（V4.1 Flash Q2 / RTX 5060 Ti 16 GiB）

**性能（同一模型、同一 prompt、`-n 64`）**

| 专家预取方式 | prefill t/s | 解码 t/s |
|---|---|---|
| 串行 pread（基线） | 2.78 | 1.93 |
| 分批栅栏 @16 | 4.43 | 3.21 |
| 滑动窗口 @8（当前默认） | **6.01** | **3.80** |

解码 1.93 → 3.80 t/s（+97%），prefill 2.78 → 6.01 t/s（+116%）。

**改动**

- 专家入缓存时不再逐个 `pread`：每个 reader 一个 staging buffer / 一个完成事件 / 一个槽位，请求按 gate→up→down 逐张量连续下发。
- 分批栅栏改为滑动窗口：哪个请求回来就立刻上拷并给该 reader 补下一个请求，读队列始终填满，不再被一批里最慢的 pread 拖住。
- 默认并发 reader 数 8（扫过 4/6/8/10/12/16，8 为峰值）。`DS4_CUDA_EXPERT_READ_DEPTH` 覆盖，上限 32；`DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1` 退回串行。
- 缓存 reserve 从固定 8 GiB 改为随显存缩放（1/8 显存，夹在 1–8 GiB，且不超过 1/4 显存），`DS4_CUDA_STREAM_EXPERT_RESERVE_MB` 仍可覆盖。
- `make cuda` 未给 `CUDA_ARCH` 时用 `nvidia-smi` 自动探测（探测不到回退 `native`）。

**修复**

- `cuda_stream_copy_release()` 设了 `quitting` 不复位，`ds4_gpu_cleanup()` 后再次初始化时新 reader 立即退出 → 回归测试卡死。现在释放后清空池状态，池可复用。
- 派活轮次号每次调用从 1 重开，与常驻计数撞号时 worker 认为没有新活 → 死锁。改为延续 worker 的票据序列。
- 所有 reader 共用一个条件变量，`pthread_cond_signal` 可能唤醒没被派活的 worker，唤醒被吞掉 → 请求无人读取、进程挂死。改为 `pthread_cond_broadcast`。
- 中断退出的批次会等待仍在读取的 reader，避免在读过程中释放 staging buffer / 描述符。

**结论修正**：此前“NVMe 5 GB/s 即墙”的机制判断有误——单个 pread 只有 ~5 GB/s，多个重叠 pread 在同一块盘上可到 ~13 GB/s；真正的墙是读队列深度，深度给足后才碰到设备带宽。
