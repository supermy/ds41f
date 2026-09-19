<p align="center">
  <img src="logo.svg" alt="DwarfStar logo" width="220">
</p>

**DwarfStar** 的目标是成为在消费级硬件（也就是普通人真正买得起的那类机器）上跑少数几个优秀大模型的**最佳方式**。
为了达成这个目标，我们写了一个体量小、纯原生的推理引擎，优先为以下模型优化：
**DeepSeek V4 Flash**（含实验性视觉模型）、**DeepSeek V4.1 Flash**（Metal，以及 CUDA 上的文本推理），
另外还支持 **GLM 5.2 与 5.3**、**GLM 5.3 Flash**、**DeepSeek V4 PRO** 以及 **Qwen3.8 Flash Next**（Metal）。
代码自包含、刻意做窄：它**不是**通用 GGUF 运行器，必须使用本项目自己产出的、属于本项目一部分的 GGUF 文件。

我们以集成方式进行测试：模型加载、prompt 渲染、工具调用、KV 状态、HTTP server 和编码 agent
是一起构建、一起测的。仓库里还包含了 GGUF、imatrix、质量评估与速度测试的工具和数据。

**English version**: [README.md](README.md)。

## 支持的硬件

- **Metal**（主目标）：96 GB 及以上内存的 Mac。更小的机器可以用 SSD 流式，
  128 GB 机器跑完整 GLM 5.x（非 Flash）也必须走 SSD 流式。
- **NVIDIA CUDA**：DGX Spark 是我们的主要目标。同时也支持其他后端不支持的多 GPU 配置，
  例如可以在 Ada Lovelace 显卡上跑 DeepSeek v4 Flash。
- **ROCm**：Strix Halo 类系统，例如 Framework Desktop。

如果没有 **llama.cpp 与 GGML**，这个项目不会存在，请务必阅读致谢部分——
非常感谢 Georgi Gerganov 以及所有贡献者。

**模型支持是刻意机会主义的**。项目跟随对个人机器尺寸最合适的开放权重，
尤其是 128 GB 笔记本与 256/512 GB 工作站。出现更好的替代品时，某个模型可能会被移除。

# 那这套软件能拿来做什么？

- 在消费级硬件（MacBook、DGX Spark、Strix Halo）上跑很强的模型。即使内存不够，
  用 SSD 流式也能得到可接受的速度。
- 把多张 CUDA 卡当多用户 LLM server 用。支持 Ada Lovelace（含 L40S）：
  一些较新的模型在这里能跑起来，而它们的其他推理实现要求更新的 GPU。
  我们的八卡 L40S Flash 配置在 16 个会话下达到了约 126 t/s 的聚合生成速度。
- 用 RDMA 连接两台 128 GB Mac，以张量并行跑 4-bit DeepSeek Flash 或 GLM 5.3 Flash。
  更大的 GLM 5.2 量化需要更大的机器，比如 Mac Studio。
- 也可以用流水线并行把多台系统拼起来，累加它们的内存以运行更大模型。

## 动机

- 强力的开放权重模型现在已经能放进高端个人机器。
- DeepSeek V4 Flash 与 PRO、GLM 5.2 都能容忍激进的路由专家量化。
- 压缩 KV cache 加上快速的本地 SSD，让长上下文变得实用。
- 想要一个只为少数几个模型特化的推理系统。

# AI 完全披露

- 本软件是在**AI 编码助手的强力参与**下开发的，由人类主导想法、测试与调试。
  我们公开说明这一点，是因为它决定了这个项目的构建方式。
  如果你不喜欢 AI 参与编写的代码，这套软件不适合你。下面的致谢同样重要：
  没有 `llama.cpp` 和 GGML（大部分是手写出来的），它就不会存在。

## 致谢 llama.cpp 与 GGML

`ds4.c` 并不链接 GGML，但它**之所以能存在，是因为 llama.cpp 项目开辟的道路**，
以及那里发展出来的 kernel、量化格式、GGUF 生态和来之不易的工程知识。
我们感谢并亏欠 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 及其贡献者。
在实现这条 DeepSeek V4 专用推理路径时，它们的实现、kernel、测试和设计选择都是必需的参考。
部分源码级片段依据 MIT 许可保留或改编在此：GGUF 量化布局与查表、CPU 量化和点积逻辑，以及部分 kernel。
出于这个原因，也因为我们由衷感激，`LICENSE` 文件里保留了 GGML 作者的版权声明。

## 状态

软件目前变化非常快，请当作 beta 质量。每次发布前都会执行一轮大 QA，
但仍然存在不稳定与回归的可能。

# 该怎么使用这个项目？

我（Salvatore）认为，AI 改变了项目发布与使用的方式，主要区别是：

1. 有了 AI，用户可以以很低的成本、很少的努力、甚至缺乏深度领域知识的方式，
   大幅修改软件。举例来说，一个 DwarfStar 用户拿着特定的硬件配置，
   可以让编码助手为这个硬件优化推理速度，要求在不影响正确性的前提下达到最快的
   prefill 与生成速度，并且再要求做一轮深度 QA。
2. 同理，因为"1"，软件的发布形态可以与以往不同。它更应该是一份覆盖最大用例的**可用模板**，
   而不是试图覆盖所有可能的配置。如果 DwarfStar 展示了几个好的张量并行实现，
   这些代码就会成为在别的条件、别的模型上实现同一特性的轨道。

所以，虽然本项目在已支持的模型和最常见硬件上力求可用，我还是建议：
如果你有编码助手，请把它们当作**探索这个项目、做修改、定制配置**的界面。
这样你很可能做到比我们发布的更多的事；有些没有文档、没有实现但你需要的东西，
实现起来可能非常容易。

## 从这里开始

```sh
git clone https://github.com/antirez/ds4.git
cd ds4
```

选择你的构建方式。各平台指南涵盖前置条件、内存估算和硬件特定配置：

| 平台指南 | 构建命令 |
| --- | --- |
| [Metal on Apple Silicon](docs/METAL.md) | `make` |
| [DGX Spark](docs/DGX_SPARK.md) | `make cuda-spark` |
| [Strix Halo / Framework Desktop](docs/STRIX_HALO.md) | `make strix-halo` |
| [一张或多张 CUDA 卡，含 Ada/L40S](docs/CUDA_MULTI_GPU.md) | `make cuda-generic` |

在一台 96 或 128 GB 的机器上首次运行，下载 DeepSeek V4 Flash Q2：

```sh
./download_model.sh ds4f-q2
```

下载到 `gguf/`。中断后重复该命令可续传。记得给上下文、运行时 buffer 留出内存。
参见 [其他模型](docs/MODELS.md)，或在更小的 Mac 上使用 [SSD streaming](docs/SSD_STREAMING.md)。

## 日常使用

构建好并下载模型之后：

```sh
./ds4
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent
./ds4-server --ctx 32768
```

默认模型是 `ds4flash.gguf`，这是一个由主模型下载脚本更新的软链接。
用 `-m FILE` 显式指定。命令通常在仓库根目录执行；
在其他目录启动时用 `--chdir /path/to/ds4`。

server 默认监听 `http://127.0.0.1:8000`；API 访问和多会话见 [serving](docs/SERVER.md)。

交互式 CLI 保持多轮对话。用 `/help`、`/read FILE`、`/ctx N`、`/quit`。
Ctrl+C 中断生成并回到提示符。每个二进制加 `--help` 可以看全部选项。

### 原生编码 agent

`ds4-agent` 直接跑推理，不需要单独的 HTTP server。它把 token 历史和实时模型状态放在一起，
显示 prefill 进度，并使用模型原生的工具格式。DeepSeek 和 GLM 各有自己的模板。

`/hints on` 可以偶尔给出关于工作背后编程选择的简短解释，`/hints off` 关闭。
改动在下一个对话边界生效，无需重建已缓存的上下文。新建和恢复的会话默认不开 hints。

会话存放在 `~/.ds4/kvcache`：

| 命令 | 作用 |
| --- | --- |
| `/save` | 保存当前会话 |
| `/list` | 列出已保存会话 |
| `/switch <sha>` | 恢复某个会话 |
| `/del <sha>` | 删除已保存会话 |
| `/strip <sha>` | 保留文本与标题，删掉大的 KV 负载 |

兼容的本地 KV 快照可以避免重建 prompt。被 strip 的会话和跨网络 TP 的恢复需要 prefill。
含图片的会话目前还不能保存。保存的对话与 trace 可能包含隐私信息。

如果你用 Pi、OpenCode、Codex CLI 或 Claude Code，请改用 `ds4-server`，
并按 [客户端配置指南](docs/CLIENTS.md) 接入。

### 模型、图像与投机解码

[模型与视觉](docs/MODELS.md) 列出了支持的下载与内存需求。
DeepSeek Vision Experimental 使用的是与 Flash 0731 不同的 checkpoint；
GLM 5.3 Flash 和 Qwen3.8 Flash Next 通过一个独立编码器，在同一个文本模型上增加视觉能力。

DeepSeek V4.1 Flash 的文本与视觉在 Metal 上运行；文本也能在 DGX Spark 上跑。
Q2 可以在一台 128 GB Mac 或 Spark 上以 SSD 流式运行，也可以用 RDMA 常驻在两台 Mac
或两台 Spark 上。Q4 需要 SSD 流式或 512 GB Mac。无论哪种模式，Engram 表都留在盘上，
所以要配合一块快的本地 SSD。下载与配置见 [模型指南](docs/MODELS.md#deepseek-v41-flash)。

把匹配的编码器用 `--vision FILE` 传入后，CLI 里用 `/read image.png`，
原生 agent 里用 `view_image`。

Qwen3.8 较小的 Q2 发布版主权重/MTP 共 **41.73 GiB**，
专家部分为 imatrix IQ2_XXS 的 gate/up 以及补齐过的 Q2_K down 投影，
它是 64 GB Mac 的起步选择。该 GGUF 还包含了 95.37 GiB 的原始 BF16 n-gram，
直接从盘上读取而不是载入内存，所以请放在高速 SSD 上。先用 8K 上下文起步：

```sh
./download_model.sh qwen38-q2
./ds4 --ctx 8192 --prefill-chunk 1024
```

这次下载拉取一个 137.10 GiB 的文件并更新 `ds4flash.gguf`。
加 `--mtp` 启用投机解码。也有更大的 `qwen38-q4k` 可选。
视觉编码器用 `./download_model.sh qwen38-vision` 下载，用 `--vision` 传入。
细节见 [Qwen 配置](docs/QWEN38_FLASH_NEXT.md)。

投机解码是可选开启的。GLM 与 Qwen 用 `--mtp`；V4 Flash DSpark 需要配套的辅助 GGUF。
它能提升生成速度，但并非所有负载都受益。见
[投机解码](docs/SPECULATIVE_DECODING.md) 了解配置方式，
以及默认的机会主义采样与 `--mtp-exact-sampling` 的区别。

### SSD 流式调优（CUDA）

SSD 流式把问题从"这个模型装得下吗"变成"**每个 token 必须穿过多少字节**"。
下面给出的是本项目在 CUDA 单卡上实测过的调优结论，
完整推导见 [docs/OPTIMIZATION_HISTORY.md](docs/OPTIMIZATION_HISTORY.md)，
按机器复现的步骤见仓库根目录的 `优化步骤教程.md`。

#### 1. 先构建，注意别跑旧二进制

```sh
make cuda                       # 没给 CUDA_ARCH 时用 nvidia-smi 自动探测
make tests/test_cuda_ssd_cache  # 关键：make cuda 不会重建测试
./tests/test_cuda_ssd_cache     # 约 5 秒；预取池出问题时它是"卡死"而不是报错
```

改过 `ds4_cuda.cu` 却忘了重建测试，就会拿着旧二进制的挂死现象去做错误归因。

#### 2. 最优命令与参数（每个模型实测最快的那一组）

```sh
# ── V4 Flash IQ2XXS：15.16 t/s
#    专家 72.56 GiB 装得下 → 启动时整包 pin。960 槽是这张卡的极限（1200 会分配失败）；
#    长 prompt 要退回 512 槽 + -c 4096。
./ds4 --cuda -m ds4flash-iq2xxs.gguf --ssd-streaming \
      --ssd-streaming-cache-experts 960 -c 2048 --prefill-chunk 512 \
      --nothink --temp 0 -n 128 -p "<prompt>"

# ── V4.1 Flash Q2：6.31 t/s
#    专家 142.38 GiB 装不下 → 靠读透式主机缓存。41 是槽位数不是字节预算，
#    写 NGB 会塌成 1 个槽。
./ds4 --cuda -m DeepSeek-V4.1-Flash-Q2.gguf --ssd-streaming \
      --ssd-streaming-cache-experts 41 -c 2048 --prefill-chunk 512 \
      --nothink --temp 0 -n 128 -p "<prompt>"

# ── GLM 5.3 Flash Q2：7.20 t/s
#    默认预留下差约 3 GiB，压到 6 GiB 就能把 81.63 GiB pin 下来。两个 env 都必须给：
#    不给守卫，16 GiB 卡上直接拒绝启动。GLM 不接受 --prefill-chunk，所以这里没有它。
DS4_GLM_MEMORY_GUARD=0 DS4_RAM_RESIDENT_RESERVE_MB=6144 ./ds4 --cuda \
      -m GLM-5.3-Flash-Q2.gguf --ssd-streaming -c 2048 \
      --nothink --temp 0 -n 128 -p "<prompt>"
```

量"这些优化值多少"时用基线对照——同一份二进制关掉三项新增即可，不用重新编译：

```sh
DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1 DS4_RAM_RESIDENT_EXPERTS=0 \
DS4_HOST_EXPERT_CACHE=off ./ds4 --cuda -m <模型> ...   # V4 3.71 · V4.1 1.78 · GLM 1.57
```

本节所有数字都来自同一套口径：固定一个 prompt（"Explain what mmap is, briefly."）、
`--nothink --temp 0 -n 128`，并且**把生成的正文做 md5 校验，各配置必须逐字一致**——
一个"更快"但改了输出的配置是 bug，不是收益。

#### 3. 测试环境（本文档所有数字的来源）

| | |
| --- | --- |
| GPU | NVIDIA RTX 5060 Ti 16 GiB（可用 15.48 GiB，sm_120） |
| 主机内存 | 96 GB —— `MemTotal` 97959540 kB ≈ 93.4 GiB，`MemAvailable` 约 90 GiB |
| 模型所在盘 | Fanxiang S910Pro 2TB NVMe，`/data` 为 ext4（`/dev/nvme1n1p6`） |
| 盘速 | 裸盘 O_DIRECT 顺序读 8.7 GB/s；重叠读约 13 GB/s |
| 系统 / 内核 | Ubuntu 24.04.2 LTS，Linux 6.8.0-139-generic |
| CUDA / 编译器 | CUDA 13.3，gcc 13.3.0，`make cuda` |
| checkpoint | V4 Flash IQ2XXS 80.76 GiB · V4.1 Flash Q2 340.60 GiB · GLM 5.3 Flash Q2 89.88 GiB |

这台机器有三个特性决定了后面所有数字：显存只有 16 GiB（专家槽稀缺）、
可用主机内存约 90 GiB（72 GiB 的专家装得下，142 GiB 的装不下）、
一块约 8.7 GB/s 的 NVMe。换机器时按这三条重新推算，别直接抄数值。

#### 4. 开关与环境变量

| 开关 / 环境变量 | 默认值 | 什么时候用 |
| --- | --- | --- |
| `--ssd-streaming-cache-experts N` | 自动 | 必须写**槽位数**，不要写 `NGB`：`NGB` 会被理解为"再预留两个完整 prefill 层"，通常会塌成 1 个槽并报 `CUDA SSD cache cannot stage ... with system headroom`。 |
| `--ram-resident-experts off\|auto\|NGB` | `auto` | `auto` 在 `MemAvailable − 预留` 装得下**全部**路由专家时，把它们在启动时 pin 进主机内存，此后 staged read 全部变成 Host→Device 拷贝。语义刻意是**要么全装、要么不装**；`NGB` 给池设上限，`off` 关闭（`DS4_RAM_RESIDENT_EXPERTS=0` 同效）。 |
| `--host-offload-token-embd` | 关 | 把 `token_embd` 放进 pinned 主机内存，腾出约 1.2 GiB 显存。只对**每 token 只读一行**的行 gather 张量有效；被 GEMV 整块读的权重不要用这种方式卸载。 |
| `DS4_CUDA_EXPERT_READ_DEPTH=N` | 8 | 专家 `pread` 的重叠队列深度。实测 8 是峰值，再深没有收益，只多占 pinned 内存。 |
| `DS4_HOST_EXPERT_CACHE=off\|NGB` | 可用内存的 1/3，上限 32 GiB | 路由专家**装不进内存**时生效：读透式主机缓存把读过的专家留在 pinned 内存，命中就走 H2D 而不是 `pread`。`off` 退回纯 SSD，`NGB` 手动定预算。 |
| `DS4_HOST_EXPERT_CACHE_STATS=1` | 未设 | 退出时打印命中率、多少字节来自内存、多少字节仍走盘。 |
| `DS4_HOST_EXPERT_CACHE_SLOTS=N` | 8 | 「允许填充」的判定阈值 = 单个 token 的路由专家数。模型每 token 路由更多专家时要调大，否则缓存永远不会被填充。 |
| `DS4_EXPERT_POOL_READERS=N` | 16 | 启动时一次性预载专家的线程数（按文件偏移排序、8 MiB 分块）。8/16/32 → 11.9 / 10.0 / 10.3 s，默认 16。 |
| `DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1` | 未设 | 串行基线。**任何 A/B 对比都要以它为基准**。 |

启动日志会告诉你是走了哪条路：看 `ds4: RAM-resident experts: ...`
（要么打出已 pin 的层数与 GiB，要么说明为什么跳过），
以及 `ds4: expert cache: N slots x X MiB = Y GiB VRAM`。

#### 5. 实测（同一 prompt，`--temp 0 -n 128`）

**基线**＝上游行为，用同一份二进制关掉三项新增来近似：
`DS4_CUDA_DISABLE_EXPERT_PARALLEL_READ=1 DS4_RAM_RESIDENT_EXPERTS=0 DS4_HOST_EXPERT_CACHE=off`。

| 模型 | 基线 | +并行预取 | +读透缓存 | +全常驻 | 总幅度 |
| --- | --- | --- | --- | --- | --- |
| V4 Flash IQ2XXS | 3.71 | 7.14 | — | **15.16** | **4.09×** |
| V4.1 Flash Q2 | 1.78 | 3.85 | **6.31** | — | **3.54×** |
| GLM 5.3 Flash Q2 | 1.57 | 3.82 | 6.07 | **7.20** | **4.59×** |

后两列**互斥**：checkpoint 要么装得进内存（整包 pin，读透缓存根本不会建），
要么装不进（那就只剩读透缓存）。所以空格不是漏测：

- **V4 Flash**（72.56 GiB 专家，装得下）：3.71 → 7.14（预取）→ 13.46（池，稳妥的 512 槽）
  → **15.16**（960 槽，这张卡的极限；1200 会分配失败）。落盘 132 → 79 GiB。
- **V4.1 Flash Q2**（142.38 GiB，装不下）：1.78 → 3.85 → **6.31**。
  读透缓存把落盘从 317 削到 89 GiB（命中率 73.7%），全部收益都来自这里。
- **GLM 5.3 Flash Q2**（默认预留下差 2.6 GiB）：拿到读透缓存 6.07；
  用 `DS4_RAM_RESIDENT_RESERVE_MB=6144` 把 81.63 GiB pin 下来 → **7.20**（prefill 7.51）。
  落盘 310 → 89 GiB。

prefill 同向变化：2.34 → 6.47、2.60 → 5.77、2.64 → 7.51 t/s。

两个值得从数字里读出来的结论：

- **并行预取搬的字节一个没少**（GLM 上 310 GiB，和基线完全相同），速度却 2.43×。
  盘本来就在供这些字节，丢掉的是等待——**墙是队列深度，不是带宽**。
- **全常驻不等于能碰到带宽天花板**：V4 Flash 到 PCIe 上限的 97%（15.16 / 15.6），
  GLM 只有 65%（7.20 / 11.0）——46 层里有 3 层不进池，且它走 full-attention 而非压缩 KV，
  剩下的瓶颈是 attention 而不是字节。

一次性预载 72.56 GiB 用 10.0 s（7.7 GB/s），已达本机裸盘 O_DIRECT 顺序读约 90%；
GLM 全常驻预载 81.63 GiB 用 11.8 s（7.5 GB/s）。

**读透缓存的预算扫描**（Q2，同一 prompt，`--temp 0 -n 128`）：

| 缓存预算 | 落盘读 | 解码 t/s |
|---|---|---|
| 关 | 316 GiB | 3.86 |
| 8 GiB | 168 GiB | 5.23 |
| 16 GiB | 118 GiB | 6.23 |
| 32 GiB | 84 GiB | 6.30 |
| 78 GiB（可用内存全给） | 80 GiB | 6.17 |
| **默认：可用内存的 1/3 = 26.18 GiB** | 89 GiB | **6.29 / 6.32** |

曲线 16–32 GiB 就到平台：32 GiB 已 75.7% 命中，78 GiB 只多 1.3 个百分点反而略慢
（更多 pinned 内存挤压页缓存）。所以默认**不**把可用内存吃满，
要更大预算就显式给 `DS4_HOST_EXPERT_CACHE=48`。默认档命中率 73.7%，
227 GiB 来自内存、81 GiB 仍走盘；开/关各跑两遍的正文 md5 逐字一致。

本机的总体判据：**每 token 字节 × 实测 t/s ≈ 供给带宽**。
逼近盘带宽时该做的是**削字节**（更小 checkpoint / 批处理摊薄 / 换更快的介质），
而不是继续加管道（更大的并发、更深的预取）。

#### 6. 坑

- `--temp 0` 是 A/B 对比的硬要求：默认采样不确定。
- 长 prompt 会为"每层 × 每个唯一路由专家"占槽：IQ2XXS 上 799 token 的 prompt
  最多只能用 512 槽，960 槽会以 `gpu layer 33 ffn batch encode failed` 失败；
  只有短 prompt 压测才能推到 960（约 9.2 t/s 上限）。
- `--host-offload-token-embd` 释放出来的 1.2 GiB **不会变成更多专家槽**：
  规划器按显存总量估算槽位，真正的收益是上下文/prefill/OOM 余量。
- **GLM 在小显存卡上必须加 `DS4_GLM_MEMORY_GUARD=0`**：它的内存守卫要预留 32 GiB，
  16 GiB 卡上预算直接算成 0，直接拒绝启动。
- **GLM 不接受 `--prefill-chunk`**（"GLM uses graph-selected prefill chunks"），
  且 `-c` 大小不影响解码速度（512 与 2048 实测相同）。
- **压预留换全常驻是有代价的**：pin 掉 GLM 的 81.63 GiB 后系统只剩约 9 GiB。
  `ds4` 把自己的 `oom_score_adj` 设成 1000，真 OOM 时先杀它，
  但内存另有用途的机器别这么干。
- `ds4-bench` 的预算算法和 `ds4` 不一样，同样的 `NGB` 会塌成 1 个槽，
  且要求 prompt 不少于 `--ctx-start`；做解码对比建议直接用 `ds4` 一次性生成。
- `ds4` 有单实例锁，残留进程会让后续运行直接报 "already running"：
  `pgrep -af "ds4 --cuda"` 先查一下。

### 输出与功耗

思考（thinking）默认开启。用 `--nothink` 或 `/nothink` 直接要答案，
`--think` 或 `/think` 重新开启。对 V4.1，`ds4` 和 `ds4-agent` 还支持
`--think-level 25` 或 `/think 25`：1 到 100 设置推理强度，0 关闭思考。
`--think` 选 75，`--think-max` 选 100。在对话中改变等级会重建已缓存的前缀。
默认采样参数是 temperature 1、top-p 1、min-p 0.05；`--temp 0` 是贪心输出。

对 DeepSeek V4，`--power N` 用吞吐换更低的持续 GPU 负载，默认 100。
V4.1 和 GLM 目前要求 `--power 100`。

DeepSeek V4 Flash 与 GLM 5.3 Flash 还支持方向性 steering。
用 `--dir-steering-file FILE` 载入向量；在 CLI 或 agent 会话里用 `/steer F`
调整它对后续 token 的强度，不需要重建已有的 KV cache。
见 [steering 文档](dir-steering/README.md)。

`--prefix-file FILE` 在正式对话之前预载完整的 `USER:` / `ASSISTANT:` 配对。
turn 标记必须位于行首，角色必须交替，最后一轮必须是 `ASSISTANT:`。

## 能力评估

`ds4-eval` 针对真实 GGUF 跑内嵌的能力回归测试。
这些是 DwarfStar 的集成检查，不是官方榜单分数。

```sh
./ds4-eval -m ds4flash.gguf --trace /tmp/ds4-eval.txt
./ds4-eval -m ds4flash.gguf --suite hard-smoke
./ds4-eval -m ds4flash.gguf --suite hard --retry-incomplete
```

默认 suite 是 `core`；`--suite all` 跑 core 和 hard 用例。
`--list-cases` 不加载模型直接列出用例。`--plain` 选择非交互输出，
`--regrade-trace FILE` 对已有 trace 打分而不再生成。
来源与许可见 [EVAL_DATA.md](EVAL_DATA.md)。
推理正确性与发布检查见 [testing](docs/TESTING.md)。

## 速度

这份 DeepSeek V4 Flash Q2 扫描记录使用一台 128 GB RAM 的 M5 Max，
2048 token 的续写 prefill 间隔，每条前沿 128 个贪心生成 token。
它是一条基线，不是每个 commit 的全新基准。

![M5 Max Flash Q2 throughput](speed-bench/m5_max_ts.svg)

完整数字、DGX Spark 结果、对比条件和基准命令见 [性能与基准](docs/PERFORMANCE.md)。

## 详细指南

- [模型与视觉](docs/MODELS.md)：Flash、PRO、GLM、Qwen 及配套编码器。
- [Qwen3.8 Flash Next](docs/QWEN38_FLASH_NEXT.md)：模型配置、MTP、视觉与校验。
- [SSD streaming](docs/SSD_STREAMING.md)：跑超过内存的模型，以及如何定缓存尺寸。
- [跨机器推理](docs/DISTRIBUTED.md)：双 Mac TP/RDMA 与层流水线。
- [投机解码](docs/SPECULATIVE_DECODING.md)：DSpark、GLM 与 Qwen MTP，以及采样。
- [Serving](docs/SERVER.md)：API、图像、批处理与磁盘 KV cache。
- [编码 agent 客户端](docs/CLIENTS.md)：Pi、OpenCode、Codex CLI、Claude Code。
- [性能](docs/PERFORMANCE.md)：可复现的测量与已记录基线。
- [测试与开发](docs/TESTING.md)：回归测试、调试与模型构建工具。
- [优化记录与目标](docs/OPTIMIZATION_HISTORY.md)：CUDA SSD 流式路径上每一次优化的
  目标、做法、实测结果与被否决的方案；其中读透式主机专家缓存的设计笔记保留在
  [PLAN-host-expert-cache.md](docs/PLAN-host-expert-cache.md)（已实现，作为实现记录）。
- [相对上游做了哪些优化](docs/OPTIMIZATIONS_VS_UPSTREAM.md)：本 fork 相对 antirez/ds4
  在 CUDA SSD 流式上的全部改动，以及逐模型的实测幅度
  （V4 Flash 4.09×、V4.1 Flash 3.54×、GLM 5.3 Flash 4.59×）。

提交 PR 之前请先读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## Logo

DwarfStar 的 logo 由 Salvatore Sanfilippo 手绘设计，用 AI 图形化，
再由 Ben Gnomino 手工重做，他的那点人情味让它真正立住了。
