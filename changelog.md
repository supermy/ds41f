# Changelog

## 2026-09-19 — 修复 `make cpu` 链接失败，并新增开发环境教程

**修复**：`make cpu`（CPU-only 构建，`DS4_NO_GPU`）链接失败：

```
undefined reference to `ds4_gpu_stream_expert_cache_configured_count'   # ds4.c:42475
```

是我加"expert cache: N slots"那条启动日志时引入的回归——该函数只有 CUDA 与 Metal 后端实现，
`DS4_NO_GPU` 下没有定义，而 `ds4.c` 在这个模式下**根本不 include `ds4_gpu.h`**（`ds4.c:139`），
所以头文件里加桩也不起作用。

按项目既有约定修复，而不是新增桩：`ds4.c:387` 的注释写明 CPU-only 构建**不提供 `ds4_gpu_*` 桩**，
"every callsite is inside !DS4_NO_GPU"——每个调用点自己包条件编译。
因此把这段日志连同它的数据一起放进 `#ifndef DS4_NO_GPU`。

**验证**：`make cpu` 编过，`./ds4 --help` 正常输出；`make cuda` 与
`tests/test_cuda_ssd_cache` 52 项不受影响（GPU 路径本就不走这段代码）。

**新增**：`docs/DEV_ENV_SETUP.md` —— 面向新手的环境教程：
预期管理（模型 45–340 GiB，手机内存跑不动）、四条路线选择（Debian+NVIDIA / 纯 CPU /
Termux / Mac）、逐步命令、CodeBuddy 用法、成功检查清单、排错表、下一步读什么。

**Termux 的定位已更正为"移动办公的登录端"**：手机/平板的 Termux **不用来构建**
（45 GiB 起的模型与 8–16 GB 内存凑不到一起），而是 SSH 回 Debian 台式机干活。
教程相应重写：Termux 装 openssh/mosh，台式机开 `sshd` 并装 mosh/tmux，
密钥登录 + `~/.ssh/config` 别名，长任务一律放 tmux（断网重连后 `tmux attach` 还在），
移动网络用 mosh（UDP 60000–61000），不在同一局域网时走 Tailscale/WireGuard 而不是裸暴露 SSH。
台式机侧已在这台机器上确认：`openssh-server` 已装、`sshd` 为 active、局域网 IP `192.168.0.168`
（无线网卡 DHCP，文档里提醒去路由器绑静态 IP，否则重启后会变）。

**Termux 自身那段仍未验证**：手上没有 Android 设备。SSH/mosh/tmux 的用法是通用的，
但"熄屏被杀、电池优化白名单、termux-wake-lock"这类来自它的常见行为，文档里已标注。

**CodeBuddy 是实际写代码的那个**：教程的定位已按真实工作流写——Termux 登录 → 台式机上跑
CodeBuddy → 它改代码、跑构建和测试 → 人下需求、批方案、验收。第 5 节新增
"在 tmux 会话里跑 CodeBuddy"（它改代码+构建+测试要几分钟到几十分钟，不需要保持连接，
挂回 `tmux attach` 就行），并把验收标准写死：看 `git diff`、测试、实测 t/s 与落盘字节，
以及 `--temp 0` 正文 md5 是否与改动前**逐字一致**——"更快但输出变了"是 bug 不是收益。

**改动位置**：`ds4.c`、`docs/DEV_ENV_SETUP.md`（新）、`README.md`、`README_CN.md`、`changelog.md`。

---

## 2026-09-19 — README 补测试环境，并把推荐命令写成各模型的最优参数

**改动**：`README.md` 与 `README_CN.md` 的调优章节。

- **新增测试环境表**：GPU（RTX 5060 Ti 16 GiB，sm_120）、主机内存（96 GB，
  `MemTotal` ≈ 93.4 GiB）、模型所在盘（Fanxiang S910Pro 2TB，`/data` ext4）、
  盘速（裸盘 O_DIRECT 8.7 GB/s，重叠读约 13 GB/s）、系统内核（Ubuntu 24.04.2，6.8.0-139）、
  CUDA 13.3 / gcc 13.3.0、三个 checkpoint 的体积。
  并点明这台机器决定所有数字的三个特性——16 GiB 显存、约 90 GiB 可用主机内存、
  单块 8.7 GB/s NVMe——**换机器时按这三条重新推算，不要直接抄数值**。
- **"推荐命令"改成"最优命令与参数"**：每个模型给实测最快的那一组，并注明参数为什么这么设：
  V4 Flash 的 960 槽是这张卡的极限（1200 分配失败，长 prompt 要退回 512 槽 + `-c 4096`）；
  V4.1 的 `41` 是槽位数不是字节预算（写 `NGB` 会塌成 1 个槽）；
  GLM 必须给 `DS4_GLM_MEMORY_GUARD=0` 且不接受 `--prefill-chunk`。
- **补基线对照命令与测量口径**：基线是同一份二进制关掉三项新增（无需重新编译）；
  口径是固定 prompt、`--nothink --temp 0 -n 128`、**正文 md5 各配置必须逐字一致**
  （"更快"但改了输出的配置是 bug 不是收益）。
- 中文版小节重排为：1 构建 / 2 最优命令与参数 / 3 测试环境 / 4 开关 / 5 实测 / 6 坑。

**改动位置**：`README.md`、`README_CN.md`、`changelog.md`。无代码改动。

---

## 2026-09-19 — README 实测部分补上 GLM 5.3 Flash，并换成四阶段优化表

**改动**：`README.md` 与 `README_CN.md` 的「SSD 流式调优 / SSD streaming tuning」章节。

- 实测表从"每个模型几行零散配置"换成**一张四阶段表**，一行一个模型：
  基线 / +并行预取 / +读透缓存 / +全常驻，并说明后两列**为什么互斥**
  （装得进内存就整包 pin、读透缓存根本不建；装不进就只剩读透缓存），所以空格不是漏测。
- 补上 **GLM 5.3 Flash** 的完整数据：1.57 → 3.82 → 6.07 → **7.20**（**4.59×**），
  prefill 2.64 → 7.51，落盘 310 → 89 GiB。
- 推荐命令加 GLM（含守卫与压预留两个 env），并把 V4 Flash 改成实测最快的组合
  （960 槽 `-c 2048 -n 128`）而不是之前保守的 512 槽 `-n 96`。
- 坑里补 GLM 相关的三条：16 GiB 卡必须有 `DS4_GLM_MEMORY_GUARD=0`、
  不接受 `--prefill-chunk` 且 `-c` 不影响速度、压预留换全常驻的代价（系统只剩约 9 GiB）。
- 口径统一为 96 GB 内存（此前写 93 GB）。

| 模型 | 基线 | +并行预取 | +读透缓存 | +全常驻 | 总幅度 |
|---|---|---|---|---|---|
| V4 Flash IQ2XXS | 3.71 | 7.14 | — | **15.16** | **4.09×** |
| V4.1 Flash Q2 | 1.78 | 3.85 | **6.31** | — | **3.54×** |
| GLM 5.3 Flash Q2 | 1.57 | 3.82 | 6.07 | **7.20** | **4.59×** |

**改动位置**：`README.md`、`README_CN.md`、`changelog.md`。无代码改动。

---

## 2026-09-19 — GLM 5.3 Flash 的四档测试结果入文档

**改动**：`docs/OPTIMIZATIONS_VS_UPSTREAM.md` 的 GLM 小节从"只有一张链条表"扩成完整的测试结果——
四档的解码/prefill/落盘/命中率、模型参数、三个从数字里读出来的事实、全常驻为何没到按 PCIe
推算的 11.0 t/s、四档复现命令、本机的两个坑、压预留的代价。
`docs/OPTIMIZATION_HISTORY.md` 第 9 节的 GLM 读数统一到最终测量值并指向该文档。

**GLM 5.3 Flash Q2 的完整结果**（89.88 GiB GGUF，路由专家 81.63 GiB，46 层 × 288 专家，
每 token 约 2.61 GB；同一 prompt、`--temp 0 -n 128`）：

| 阶段 | 解码 t/s | prefill t/s | 落盘 GiB | 命中率 |
|---|---|---|---|---|
| 上游基线（三件事全关） | 1.57 | 2.64 | 310 | — |
| + 并行预取 | 3.82 | 5.58 | 310 | — |
| + 读透缓存（默认 26 GiB 预算） | 6.07 | 5.43 | 98 | 70.2% |
| + 压低预留使其全量常驻（`RESERVE_MB=6144`） | **7.20** | 7.51 | 89 | — |

合计 **4.59×**，落盘字节 310 → 89 GiB（−71%）。

本次补测了此前缺的一格：**并行预取档的落盘字节是 310 GiB，与基线完全相同，
速度却是 2.43×**。这直接印证了本项目在这里的核心判断——瓶颈先是**读队列深度**，
不是盘的带宽：盘一直在以同样的字节量供给，只是不再空等。

另外记录清楚：全常驻后 89 GiB 落盘里绝大部分是启动那一次预载（81.63 GiB，
11.8 s @ 7.5 GB/s），解码阶段几乎不再碰盘；而它没到按 PCIe 算的 11.0 t/s，
是因为 46 层里只有 43 层进池（3 层混合精度仍走盘，约 17 ms）加上 GLM 自己的
`full-attention argmax` 路径（约 30 ms）——**盘踢出去后瓶颈是 attention 计算，不是搬运**。

**改动位置**：`docs/OPTIMIZATIONS_VS_UPSTREAM.md`、`docs/OPTIMIZATION_HISTORY.md`、`changelog.md`。
无代码改动。

---

## 2026-09-19 — 0.6：GitHub Actions 自动构建与发布

**改动**：新增 `.github/workflows/release.yml`。tag（`v*`）推送即构建并发布 release；
PR 只构建不发布；`workflow_dispatch` 可手动指定 CUDA 架构重新出包。

| job | 环境 | 命令 | 产物 |
|---|---|---|---|
| `linux-cuda` | ubuntu-latest + CUDA toolkit 12.8 | `make cuda CUDA_ARCH=sm_120a` | 带本 fork 全部 CUDA SSD 流式优化 |
| `linux-cpu` | ubuntu-latest | `make cpu`（`DS4_NO_GPU`） | 无 GPU 依赖的通用版 |
| `windows-cpu` | windows-latest + MSYS2/MinGW-w64 | `make cpu` | **实验性，允许失败**（见下） |

发布产物统一用 `-march=x86-64-v2 -mtune=generic` 而不是默认的 `-march=native`：
核心代码不直接用 SIMD intrinsics（靠编译器自动向量化），降档后仍然正确，
但要保证编译出来的二进制能在别的机器上跑。CUDA job 用 `-j2`，nvcc 编这几个大文件
单个就要几 GB，runner 只有 16 GB。

**Windows 构建状态：实验性，这一版很可能没有 `windows` 产物**

`Makefile` 只认 Darwin 与 Linux，源码没有 Windows 分支，缺一个 POSIX 兼容层。
这个 job 设了 `continue-on-error`，让 CI 持续暴露真实的编译错误，作为后续移植的输入：

| 需要的 | 用在哪 | 说明 |
|---|---|---|
| `sys/mman.h`（`mmap`/`munmap`/`madvise`） | `ds4.c` 模型加载：mmap 1 处、munmap 3 处、madvise 1 处 | 用 `CreateFileMapping`/`MapViewOfFile` 封装，`madvise` 可 no-op |
| `flock` | `ds4.c` 单实例锁，1 处 | 换 `LockFileEx`，或改成 `O_EXCL` 建锁文件 |
| `sys/socket.h`、`netinet/in.h`、`netinet/tcp.h`、`poll.h`、`arpa/inet.h`、`sys/wait.h` | `ds4_distributed.c`、`ds4_tp.c`、`ds4_web.c`、`ds4_server.c` | Winsock2 适配：`close()→closesocket()`、`errno→WSAGetLastError()`，还要 `WSAStartup` |
| `O_DIRECT` | SSD 直读路径 | 换 `FILE_FLAG_NO_BUFFERING`，或退化到 buffered |
| `pthread`（约 262 处） | 各模块 | MinGW 的 winpthreads 可直接用 |
| `pread`（约 15 处） | `ds4.c` 等 | mingw-w64 提供 |

**改动位置**：`.github/workflows/release.yml`（新）、`changelog.md`。本次无推理代码改动。

**发布**：`v0.6`。

---

## 2026-09-19 — 新增《相对上游做了哪些优化》，并把三模型的基线→最优链条测完整

**改动**：新增 `docs/OPTIMIZATIONS_VS_UPSTREAM.md`，汇总本 fork 相对 antirez/ds4 的全部优化
（目标 / 做法 / 影响哪个模型 / 幅度），并给出 V4 Flash、V4.1 Flash、GLM 5.3 Flash
三个模型从"上游基线"到"本机最优"的逐步链条与复现命令。`README.md`、`README_CN.md`
详细指南与 `docs/OPTIMIZATION_HISTORY.md` 文档索引各加一条链接。

**实测**（96 GB 级内存，RTX 5060 Ti 16 GiB，同一 prompt，`--temp 0 -n 128`）

基线是同一份二进制关掉三个开关后的读数，不是编译上游代码测的：
`DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1 DS4_RAM_RESIDENT_EXPERTS=0 DS4_HOST_EXPERT_CACHE=off`。

| 模型 | 上游基线 | +并行预取 | +常驻池/读透缓存 | +调优 | **总计** |
|---|---|---|---|---|---|
| V4 Flash IQ2XXS | 3.71 | 7.14 | 13.46 | **15.16**（960 槽） | **4.09×** |
| V4.1 Flash Q2 | 1.78 | 3.85 | **6.31**（读透缓存） | — | **3.54×** |
| GLM 5.3 Flash Q2 | 1.57 | 3.82 | 6.07 | **7.20**（压预留全常驻） | **4.59×** |

prefill 同口径：V4 2.34 → 6.47（2.77×）、V4.1 2.60 → 5.77（2.22×）、GLM 2.64 → 7.51（2.84×）。
落盘字节：V4 132 → 79 GiB、V4.1 317 → 89 GiB（−72%）、GLM 310 → 89 GiB（−71%）。

**改动位置**：`docs/OPTIMIZATIONS_VS_UPSTREAM.md`（新）、`README.md`、`README_CN.md`、
`docs/OPTIMIZATION_HISTORY.md`、`changelog.md`。本次无代码改动。

---

## 2026-09-19 — `DS4_RAM_RESIDENT_RESERVE_MB`：让差几个 GiB 的 checkpoint 也能全量常驻

**动机**：主机预留固定为总内存的 1/8（本机 11.68 GiB），于是 GLM 5.3 Flash 的
81.63 GiB 路由专家需要 93.31 GiB 可用，而本机 `MemAvailable` 只有约 90 GiB —— 差 2.6 GiB，
只能走读透缓存。这个预留是为了系统上其他东西（pinned 内存不可换出），
但"差一点点就装得下"时，这个取舍应该由用户决定。

**改动**：新增 env `DS4_RAM_RESIDENT_RESERVE_MB`（MiB，0..65536），覆盖默认的
`总内存/8`（夹在 4..16 GiB）。不设时行为与之前完全一致。
它同时影响常驻池的判定和读透缓存的预算（两者共用 `usable`）。

**改动位置**：`ds4.c`（`ds4_engine_install_expert_host_pool` 的预留计算）。

**实测**（GLM 5.3 Flash Q2，同一 prompt，`--temp 0 -n 128`，16 GiB 卡 + 93 GiB 内存）

| 配置 | 预留下 usable | 走的路径 | 解码 t/s | prefill t/s |
|---|---|---|---|---|
| 默认（11.68 GiB 预留） | 78.4 GiB < 81.63 | 读透缓存 | 6.06 | 5.67 |
| `RESERVE_MB=6144` | 84.7 GiB | **全量常驻**（81.63 GiB pinned） | **7.20** | **7.82** |
| `RESERVE_MB=8192` | 82.7 GiB | **全量常驻** | 7.20 | 7.76 |
| 缓存全关 | — | 纯 SSD | 3.82 | 5.43 |

预载 81.63 GiB 用 11.8 s（7.5 GB/s），43 of 46 层进池（3 层是混合精度层，不进 slab 缓存）。
6 GiB 与 8 GiB 结果相同，说明 8 GiB 预留也够，按机器上还要跑什么在 6–8 之间选。
代价是 pin 完系统只剩约 9 GiB；`ds4` 的 `oom_score_adj=1000` 保证 OOM 时先杀它。

**上限的修正**：此前按 `2.61 GB ÷ 28.6 GB/s` 推算 GLM 全常驻能到 ~11 t/s，
实测只有 7.20（139 ms/token）。差的那 48 ms 有据可查：3 个非均匀层仍走盘（约 17 ms），
GLM 走 `full-attention argmax generation path` 而非压缩 KV 路径（约 30 ms）。
即**搬字节不再是它的瓶颈，attention 计算才是**——同样全常驻，V4 Flash 能贴到 PCIe 的
97%，GLM 只到 65%。

**测试**：`tests/test_cuda_ssd_cache` 52 项全 PASS（本次改动只是一个 env 覆盖，默认路径不变）。

---

## 2026-09-19 — 三个模型的横评、槽数扫描与上限核算（文档）

**改动**：`docs/OPTIMIZATION_HISTORY.md` 新增第 9 节「三个模型的横评与上限」，
并把第 0 节的目标、时间线总览和第 8 节的推荐命令同步到实测结果；
第 8 节的已知坑里补上单实例锁，并更正"960 槽约 9.2 t/s"这条常驻池之前的旧数字。

**实测**（RTX 5060 Ti 16 GiB，93 GiB 内存，同一 prompt，`--temp 0 -n 128`）

| 模型 | 路由专家 | 走的路径 | 缓存 off | 最快 | 每 token 字节 |
|---|---|---|---|---|---|
| V4 Flash IQ2XXS | 72.56 GiB | 全量常驻池 | — | **15.19 t/s** | 1.83 GB |
| V4.1 Flash Q2 | 142.38 GiB | 读透缓存 | 3.85 | **6.30 t/s** | 2.65 GB |
| GLM 5.3 Flash Q2 | 81.63 GiB | 读透缓存（差 3.2 GiB 装不下） | 3.82 | **6.06 t/s** | 2.61 GB |

- V4 Flash 的专家槽扫描：256 槽 11.76、512 槽 13.55、**960 槽 15.19**，1200 及以上
  `q8_hc_expand` arena 分配失败。槽越多越快，是因为相邻层/相邻 token 会选到同一批专家。
- 上限核算：V4 Flash 的有效带宽 27.8 GB/s，已贴死 PCIe 天花板 15.6 t/s 的 97%；
  V4.1 与 GLM 只有 15.8 GB/s——它们的落盘只剩 0.7–0.8 GiB/token（约 5 GB/s，盘已不是瓶颈），
  慢在数据要绕"pread→staging→H2D"和"pinned→H2D"两趟。
- 加内存的预测：GLM 只差 3.2 GiB 就能全常驻（128 GB 机器，预计 6.06 → ~11 t/s）；
  V4.1 需要 158.4 GiB 可用（192 GB 级，预计 ~10.8 t/s）。本机 GLM 数学上不可能：
  需要 93.31 GiB 而 `MemAvailable` 最大 90 GiB。
- GLM 在本机必须 `DS4_GLM_MEMORY_GUARD=0`（否则守卫要预留 32 GiB，16 GiB 卡上预算为 0），
  且不接受 `--prefill-chunk`；`-c` 512 与 2048 速度相同。

**改动位置**：`docs/OPTIMIZATION_HISTORY.md`、`changelog.md`。

---

## 2026-09-19 — 读透式主机专家缓存：路由专家装不进内存时的降级路径（CUDA）

**目标**：`--ram-resident-experts` 的语义是"全量或不装"——V4.1 Flash Q2 的 142.38 GiB 路由专家
装不进约 78 GiB 的可用内存，于是整条路径退回每 token 全额落盘读（3.85 t/s）。
让"装不下"这一档降级为**读透式（read-through）主机缓存**：已经读过的专家留在 pinned 主机内存里，
后续命中用 H2D 拷贝代替 pread。

**先做的零代码对照实验**（计划第 0 节，决定值不值得写这段代码）：用内核页缓存冒充读透缓存
（`DS4_CUDA_NO_DIRECT_IO=1` + `DS4_CUDA_KEEP_MODEL_PAGES=1`），同一 prompt、`--temp 0 -n 128`：

| 组 | 落盘读 | prefill t/s | 解码 t/s |
|---|---|---|---|
| A 当前行为（O_DIRECT + 丢弃模型页） | 316.4 GiB | 5.90 | 3.81 |
| B1 冷页缓存第一遍 | 65.3 GiB | 4.55 | 4.97 |
| B2/B3 页缓存已热（= 复用上限） | 0.0 GiB | 10.33 | 6.54 |

B2 相对 A **+72%**，远超"≥20% 才动手"的判据：每个 token 读约 2.4 GiB，但 128 token 读掉的
316 GiB 里，唯一字节远少于此——专家复用是真实存在的。

**改动**

- `ds4_cuda.cu` 新增读透式主机专家缓存（在 `g_expert_pool` 之后）：gate/up/down 各一个 arena，
  条目按文件偏移索引（`unordered_map` + 向量），`cudaHostAlloc` 按约 1 GiB 分块分配
  （几万次小 pinned 分配会先撞上 `vm.max_map_count`），淘汰用 CLOCK 二次机会，跳过 `refs>0`
  与正在填充的条目。
- 命中在 `cuda_stream_copy_worker`（前台逐专家）里返回池内指针并**取引用**，引用在
  `cuda_stream_copy_requests` 末尾那次无条件 `cudaStreamSynchronize` 之后统一释放；
  中止的批次也要收走引用，否则条目会永久不可淘汰，缓存慢慢变成只读。
- 填充只在"一个 token 自己的路由"这一档发生。判据不是入口函数而是**请求里的专家数**
  （≤ 8，可用 `DS4_HOST_EXPERT_CACHE_SLOTS` 调）：DeepSeek 的 decode 与 prefill 走的是同一个
  `begin_selected_load`（实测 slot 数分布：6 × 1240 次 = decode，12/48 = prefill），
  按入口区分是错的。prefill 一次扫上千专家，让它填充会把后面 token 要用的热专家冲掉。
- `cuda_stream_prefetch_read`（整层预读）只查不填：look-ahead 不是 reuse 的证据。
  它的引用在 `ds4_gpu_stream_expert_cache_prefetch_finish` 里、join 之后统一释放。
- 填充路径先独占条目（`state=filling`）、锁外 memcpy、再发布：并发查到的一定是完整字节。
- 拆卸：`ds4_gpu_cleanup`、`ds4_gpu_set_model_map`、`ds4_gpu_set_model_fd_for_map`
  （测试会在同一个 mapping 下重写文件）三处都释放缓存。
- `ds4.c`：路由专家装不下时改调 `ds4_gpu_host_cache_install`；`DS4_HOST_EXPERT_CACHE=off|NGB`
  覆盖预算，`DS4_HOST_EXPERT_CACHE_STATS=1` 退出时打命中率。

**改动位置**：`ds4_cuda.cu`、`ds4_gpu.h`、`ds4.c`、`ds4.h`、`ds4_help.c`、`tests/test_cuda_ssd_cache.c`。

**实测（RTX 5060 Ti 16 GiB，93 GB 内存，Q2，41 槽，同一 prompt，`--temp 0 -n 128`）**

| 缓存预算 | 落盘读 | 解码 t/s |
|---|---|---|
| 关（`DS4_HOST_EXPERT_CACHE=off`） | 316 GiB | 3.86 / 3.84 |
| 4 GiB | 228 GiB | 4.19 |
| 8 GiB | 168 GiB | 5.23 |
| 16 GiB | 118 GiB | 6.23 |
| 32 GiB | 84 GiB | 6.30 |
| 48 GiB | 80 GiB | 6.38 |
| 78 GiB（可用内存全给） | 80 GiB | 6.17 |
| **默认（可用内存的 1/3 = 26.18 GiB）** | 89 GiB | **6.29 / 6.32** |

默认**不是**把可用内存都吃下：曲线在 16–32 GiB 就到平台，32 GiB 已 75.7% 命中，
78 GiB 只多 1.3 个百分点还慢一点（更多 pinned 内存挤压页缓存）。所以默认取
`usable / 3`（上限 32 GiB），显式 `NGB` 或 `DS4_HOST_EXPERT_CACHE=NGB` 时按用户给的值走。
默认档命中率 73.7%，227 GiB 从内存供给，81 GiB 仍走盘。

正确性：四次运行（开/关各两遍）的 128 token 正文 **md5 逐字一致**。

**测试**：`tests/test_cuda_ssd_cache.c` 新增用例——8 个专家的缓存装满后清空源文件
（`ftruncate`），同一批 routed MoE 必须复现参考输出（只能来自缓存）；
再把预算压到 2 个专家使每轮都强制淘汰，连续三轮结果仍逐字节相同。

**与计划文档的偏差**

- 填充开关按"请求专家数"判定，而不是按 decode/prefill 入口（原因见上）。
- 默认预算取可用内存的 1/3 而非全部（依据上面的预算扫描）。
- 顺序回退路径 `cuda_model_copy_to_device_streamed` 没有接缓存：它只在并行暂存起不来时才走，
  收益面很小，先不增加一条额外路径。

---

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
