# 计划：读透式主机专家缓存（已存档，尚未实施）

> **状态：计划文档，未写任何实现代码。**
> 首次落盘 2026-09-18；2026-09-18 **冻结存档**：按用户要求"保存计划，先不实施"。
> 本文档只是把方案、挂钩点、验收标准固定下来，供后续开工时对照执行。
> 实际动工前必须先做第 0 节的零代码对照实验。
>
> 解冻条件：有人明确要求开工。动工时按本文档 0→1→2→3→4→5 顺序执行，
> 完成标准写在第 4、5 节（测试通过 + Q2/V4 Flash 实测 + changelog 更新）。
> 第 0 节实验若显示 B2 相对 A 提升 < 20%，回头讨论后再决定是否继续。

## 目标

解决"路由专家装不进内存就完全没有加速"这一断层。当前 `--ram-resident-experts`
的语义是**全量或不装**：V4 Flash IQ2XXS（专家 72.56 GiB）能整包常驻，解码 7.22 → 13.02 t/s；
V4.1 Flash Q2（专家 142.38 GiB，可用内存约 78 GiB）装不下，被自动忽略，停在 3.86 t/s。

本方案在"装不下"时降级为**读透式（read-through）主机缓存**：已经读过的专家留在 pinned 主机内存里，
后续命中用 H2D 拷贝代替 pread，从而削掉一部分落盘字节。

收益估算（均匀分布）：常驻比例 f = 预算 / 专家总量，命中率 ≈ f，解码 ≈ 3.80 / (1 − f)。
本机 f ≈ 0.45 → 约 6.9 t/s。若专家访问有偏斜（`ds4_streaming_hotlist.inc` 按 hits/weight 排序，
说明确实有偏斜），命中率会高于 f。上界是 PCIe：2.39 GB/token ÷ 28.6 GB/s ≈ 12 t/s，需要 f ≥ 0.70
（≥ 100 GiB），本机达不到。

## 0. 先做零代码对照实验（动手之前，先确认"复用"到底值多少）

用内核页缓存冒充读透缓存：`DS4_CUDA_NO_DIRECT_IO=1`（`ds4_cuda.cu:4816` 关掉 O_DIRECT）
+ `DS4_CUDA_KEEP_MODEL_PAGES=1`（`ds4_cuda.cu:2123/2142` 不再 fadvise/madvise DONTNEED）。

```sh
cd /data/ai/ds41
M=/data/ai/models/gguf/DeepSeek-V4.1-Flash-Q2.gguf
COMMON="--cuda --ssd-streaming --ssd-streaming-cache-experts 41 -c 2048 --prefill-chunk 512 -p <同一 prompt> -n 128"
./ds4 -m $M $COMMON                                                              # A 基线（当前行为）
DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 ./ds4 -m $M $COMMON          # B1 冷页缓存
DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1 ./ds4 -m $M $COMMON          # B2 页缓存已热 = 复用上限
```

每组同时后台采样 `/proc/<pid>/io` 的 `read_bytes` → 得到每 token 实际落盘字节。

**判据：B2 相对 A 提升 ≥ 20% 才实现；明显没提升就先回来讨论**
（说明专家复用不足，改投"扩大静态常驻池 / 换更小检查点"更划算）。
实验结论同时决定缓存预算默认值是否要收敛。

## 1. 参数（不新增命令行开关）

扩展 `--ram-resident-experts off|auto|NGB`（默认 `auto`）：

- 装得下全部路由专家 → 维持现在的静态池（原语义不变）
- **装不下 → 自动降级为读透缓存**，预算 = `MemAvailable − reserve`
  （沿用 `ds4.c:70151-70167` 现有公式，本机约 78 GiB）
- `NGB` 同时是池上限与缓存预算；`off` 两者都关

改动点：`ds4_cli.c:2065`（解析不变）、`ds4_help.c:176`（帮助文本补一句降级行为）、
`ds4.c:70228` 的 env `DS4_RAM_RESIDENT_EXPERTS`（现只认 0/非空，同步 off/auto）。
新增调试 env：`DS4_HOST_EXPERT_CACHE=off|NGB`（覆盖预算）、`DS4_HOST_EXPERT_CACHE_STATS=1`（退出时打统计）。

## 2. CUDA 实现（ds4_cuda.cu，放在 `g_expert_pool` 之后，约 2528 行处）

**三个 arena**（与设备端 `gate_ptr/up_ptr/down_ptr` 同构，零空间浪费）：
gate（slot = `gate_expert_bytes`）、up、down，每个 arena 条目数
`n = budget / (gate_per + up_per + down_per)`。
分配：`cudaHostAlloc` 按 1 GiB 大块切分（避免几万次小 pinned 分配打爆 `vm.max_map_count`），
槽长按 `g_model_direct_align` 取整，`cuda_align_ptr` 对齐（照抄 2414/27671 写法）。

**条目**：`{offset, state(empty/filling/valid), refs, ref_bit, stamp, host}`；
`std::unordered_map<offset, entry>` 索引 + 全局 `pthread_mutex_t`；
淘汰用 CLOCK 二次机会（每 arena 一个 hand），跳过 `refs>0` 与 `filling` 的条目，
扫两圈仍无可用槽就本次不填充。

**API**（`ds4_gpu.h`，与 `ds4_gpu_expert_pool_install` 同一 `#if` 块内）：

```c
int  ds4_gpu_host_cache_install(const void *map, uint64_t model_size,
                                uint64_t gate_per, uint64_t up_per, uint64_t down_per,
                                uint64_t budget_bytes);
void ds4_gpu_host_cache_release(void);
void ds4_gpu_host_cache_stats(uint64_t *hits, uint64_t *hit_bytes, uint64_t *fills,
                              uint64_t *fill_bytes, uint64_t *pread_bytes, uint64_t *bytes);
```

**挂钩点（漏一处就白改）**

1. `cuda_stream_copy_worker` `ds4_cuda.cu:27607` —— 先查静态池，再查缓存；命中 →
   `payload = 槽指针`、`from_pool=1`（27629，命中不 drop/discard pages）、`refs++` 并把
   (arena, entry) 记进 `pool.host_ref[i]`；未命中且允许填充 → pread 之后 `memcpy` 进预留槽并 publish
   （**必须经 staging 中转**：张量偏移只有 512 对齐，O_DIRECT 直读会失败）。
2. `cuda_stream_copy_requests` `ds4_cuda.cu:27691` —— 每个已应答槽（含 `read_ok==0` 提前 continue 的分支）
   把 `host_ref[i]` 收进本地列表，**统一在 27834 `cudaStreamSynchronize` 之后释放**
   （唯一无条件执行点，避免每批最后一条 / MemcpyAsync 失败 / break 路径泄漏引用，泄漏会让槽永久不可淘汰）。
3. `ds4_gpu_stream_expert_cache_prefetch` `ds4_cuda.cu:28335` —— 建 copy 时按 (part, expert) 查缓存，
   命中就在 `cuda_stream_prefetch_copy` 上记 `host` 指针（并在命中/未命中边界处禁止合并 28340-28342），
   建表时即 `refs++`；`cuda_stream_prefetch_read` `28153` 命中走 H2D，**不填充**
   （look-ahead 不是 reuse 的证据）。引用在 `28195` `cudaStreamSynchronize` 之后统一释放（cancel/break 都会走到这里）。
4. 顺序回退 `cuda_model_copy_to_device_streamed`（`28044` 调用）只查不填：它是同步 `cudaMemcpy`，无引用问题。
5. `ds4_gpu_cleanup` `ds4_cuda.cu:3254` —— 调 `cuda_host_cache_release()`（env 打开时打统计）；
   `ds4_gpu_set_model_map` / `ds4_gpu_set_model_fd_for_map`（`4806` 附近）与 `ds4_gpu_expert_pool_release`
   都要 invalidate，否则换模型 / 测试重写文件会吐陈旧字节。

**只在解码路径填充**：`ds4_gpu_stream_expert_cache_begin_selected_load`（`34354`，解码）置允许填充；
`ds4_gpu_stream_expert_cache_prepare_selected_batch`（`34586`，prefill）置禁止
（prefill 一次扫上千专家，填充会把解码的热专家冲掉，且它本身几乎不会命中）。

**请求粒度**：`cuda_stream_copy_request` / `cuda_stream_prefetch_copy` 各加一个 `uint32_t part`，
直接由 `28015` / `28335` 的循环变量填，arena 选择不再靠 bytes 猜。
请求字节数与 arena 槽长不等时（混合精度层）跳过缓存。

## 3. ds4.c（安装与预算）

`ds4_engine_install_expert_host_pool()` `ds4.c:70108`：现有逻辑算完 `total` 与 `usable` 之后，
装不下时改为调 `ds4_gpu_host_cache_install(map, size, gate_per, up_per, down_per, usable)`，
其中 `*_per = max(各 uniform 层 tensor->bytes) / DS4_N_EXPERT`（`ds4.c:934`）。
日志：`ds4: host expert cache: ... GiB, N experts/arena (x3), 专家总量 X GiB → 覆盖率 Y%`。
**兜底**：条目数不足一层专家数就不装，并说明原因。

## 4. 测试（扩 `tests/test_cuda_ssd_cache.c`，`make test-cuda-ssd-cache`）

沿用该文件现有套路（合成模型 + `mkstemp` + `ds4_gpu_set_model_fd_for_map`）：

- 装一个小缓存（只够 2~3 个专家），同一批 routed MoE 跑两遍，逐字节比对关闭缓存时的参考输出；
- 第二遍必须出现 `hits > 0`；
- 借现有"清空源文件内容"（`ftruncate`）的技巧验证命中真的没读盘；
- 淘汰正确性：缓存容量 < 本批所需专家数时结果仍正确。

## 5. 文档与实测

`docs/SSD_STREAMING.md` 补一节；`changelog.md` 追加一条（照现有格式：改动 / 改动位置 / 实测表 / 测试）；
`docs/OPTIMIZATION_HISTORY.md` 更新状态为"已实现"。

实测：Q2 上 A（当前）vs 读透缓存的 t/s 与每 token 落盘字节，
`--temp 0` 下输出逐字一致为通过条件；V4 Flash（池已全覆盖）回归一次确认无退化。

## 关键文件

`ds4_cuda.cu`（核心）、`ds4_gpu.h`、`ds4.c`、`ds4.h`、`ds4_cli.c`、`ds4_help.c`、
`tests/test_cuda_ssd_cache.c`、`docs/SSD_STREAMING.md`、`changelog.md`
