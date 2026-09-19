# 开发环境教程：从零跑通这个项目

面向没接触过推理引擎的人。目标是让你**在这台机器上把代码编出来、跑起来**，
并且知道下一步该读什么。更细的机制、实测数据、优化过程都记在项目里，本文只负责把你送到门口，
细节一律指向对应文档。

## 0. 开工前先读这一段：预期管理

这是 **DeepSeek V4 / V4.1 Flash 与 GLM 5.3 Flash 的专用推理引擎**，不是通用 GGUF 运行器——
必须用项目自己提供的 GGUF 文件。

| 事实 | 含义 |
| --- | --- |
| 模型体积 45 GiB ~ 340 GiB | 磁盘要够大，下载要花时间 |
| 最小模型也要 45 GiB | **手机/平板（8–16 GB 内存）跑不动**，Termux 只能当开发环境 |
| 推理速度单位是 t/s | 每秒生成的 token 数；CPU 后端实测约 0.06 t/s，慢到没法聊天 |
| 真正好用的配置 | NVIDIA 显卡 + 96 GB 级内存 + 快 NVMe，或 96 GB+ 的 Mac（Metal） |

本 fork（`supermy/ds41f`）相对上游 [antirez/ds4](https://github.com/antirez/ds4) 只多做一件事：
把 **CUDA 单卡上的 SSD 流式推理**推到带宽上限。上游的主战场是 Metal（Mac）与 DGX Spark。
改动清单与实测幅度：[OPTIMIZATIONS_VS_UPSTREAM.md](OPTIMIZATIONS_VS_UPSTREAM.md)。

## 1. 先选一条路

| 你的设备 | 构建目标 | 能跑真实模型吗 |
| --- | --- | --- |
| Debian/Ubuntu + NVIDIA 显卡 | `make cuda` | ✅ 能，带本 fork 的全部优化 |
| Debian/Ubuntu 只有 CPU / 虚拟机 / CI | `make cpu` | ⚠️ 能跑但极慢（0.06 t/s），适合验证流程 |
| 手机 / 平板 Termux | `make cpu` | ❌ 内存不够，只能编译和跑测试 |
| Mac（Apple Silicon，96 GB+） | `make` | ✅ 上游主战场（本 fork 没改 Metal） |

## 2. 通用第一步：拿代码

```sh
git clone https://github.com/supermy/ds41f.git    # 本 fork
cd ds41f
```

想跟上游就 `git clone https://github.com/antirez/ds4.git`。
想给上游提改动前先读 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 3. Debian / Ubuntu 服务器

### 3.1 装基础工具

```sh
sudo apt update
sudo apt install -y build-essential git curl
```

### 3.2 有 NVIDIA 显卡

```sh
nvidia-smi                      # 先确认驱动在
sudo apt install -y nvidia-cuda-toolkit
nvcc --version
make cuda                       # 不给 CUDA_ARCH 时用 nvidia-smi 自动探测架构
```

显存小于 96 GB 的卡（比如 16 GiB）**必须走 SSD 流式**：

```sh
./ds4 --cuda -m <模型.gguf> --ssd-streaming \
      --ssd-streaming-cache-experts 41 -c 2048 --prefill-chunk 512 --nothink -n 128 -p "你好"
```

各模型的最优参数与实测速度见 [README_CN.md 的调优章节](../README_CN.md) 与
[OPTIMIZATIONS_VS_UPSTREAM.md](OPTIMIZATIONS_VS_UPSTREAM.md)。

> apt 的 toolkit 版本可能偏老。RTX 50 系（sm_120）需要 CUDA 12.8+，
> 那就从 NVIDIA 官网装对应版本，再 `make cuda CUDA_ARCH=sm_120a`。

### 3.3 只有 CPU

```sh
make cpu
./ds4 --help
```

`make cpu` 编出的是 `DS4_NO_GPU` 版本：不依赖任何 GPU 运行时，能编过、能启动、
能跑不依赖 GPU 的测试。真跑模型会非常慢（CPU 后端定位是参考/调试代码，见 [AGENT.md](../AGENT.md)）。

### 3.4 下载模型

```sh
./download_model.sh ds4f-q2     # DeepSeek V4 Flash Q2
./download_model.sh glm53-q2    # GLM 5.3 Flash Q2
ls -la gguf/
```

下载到 `gguf/`，中断后**重复执行同一条命令就能续传**。可选模型见 [docs/MODELS.md](MODELS.md)。

### 3.5 跑起来

```sh
./ds4                                        # 交互式（默认 ds4flash.gguf）
./ds4 -p "Explain Redis streams in one paragraph."
./ds4-agent                                  # 终端里的编码 agent
./ds4-server --ctx 32768                     # HTTP 服务，默认 127.0.0.1:8000
```

想接 Pi / OpenCode / Codex CLI / Claude Code 看 [docs/CLIENTS.md](CLIENTS.md)。

## 4. Termux（手机 / 平板）

> **这一段我无法验证** —— 手上没有 Android 设备。它建立在"Android 是 Linux 内核、
> 有 `mmap` 和 pthread"这一事实上，步骤是通用的。遇到报错请把错误信息发出来。

```sh
pkg update
pkg install -y git clang make
git clone https://github.com/supermy/ds41f.git
cd ds41f

# ARM 上 gcc/clang 经常认不出 -march=native，置空最稳
make cpu NATIVE_CPU_FLAG=
./ds4 --help
```

**能做的**：

- 编译验证——改了代码能编过，这在手机上就够了
- 跑不依赖模型的测试：`make ds4_test && ./ds4_test --server`
- 读代码、学结构（[AGENT.md](../AGENT.md) 有模块说明）

**做不到的**：跑真实模型（内存不够）、GPU 加速（Termux 上没有 CUDA / Metal）。
想在手机上真跑起来，请换 96 GB 内存的机器或用服务器。

## 5. 用 CodeBuddy 开发这个项目

CodeBuddy 是 AI 编码工具（就是现在在跟你说话的这个）。本项目本身就是 AI 重度参与开发的，
上游 README 里写明了这一点。用法上的几条经验：

1. **先把规范喂给它**：让它读 [AGENT.md](../AGENT.md)。里面写了这个项目的硬规矩——
   不引入 C++、正确性优先于速度、注释解释"为什么"、不留死代码。不先读这个，
   它很容易写出风格不符的补丁。
2. **再让它读历史**：[docs/OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md)
   记录了做过什么、**否决过什么**。不读就会重复试已经证明无效的方向
   （跨层预取、MoE 放 CPU、权重压缩都试过，都否决了）。
3. **提需求要带上下文**：机器配置（显卡/内存/盘）、模型、复现命令、实测数字。
   只说"让它更快"，它只能猜。
4. **先要方案再要代码**："先给我方案和判据，别改代码"能省掉大量返工。
5. **改完必须重建测试**：`make cuda` **不会**重建测试二进制，改了 `ds4_cuda.cu` 后
   漏掉这一步会拿着旧二进制的现象做错误归因。

   ```sh
   make cuda && make tests/test_cuda_ssd_cache && ./tests/test_cuda_ssd_cache
   ```

   这个测试约 5 秒，52 项；预取池出问题时它表现为**卡死**而不是报错，所以比基准更早发现问题。

## 6. 怎么确认自己成功了

- [ ] `make cpu` 或 `make cuda` 没有 `error`
- [ ] `./ds4 --help` 打印出用法
- [ ] 有 GPU：`./tests/test_cuda_ssd_cache` 全 PASS（约 5 秒）
- [ ] 有 GPU + 模型：能生成一段文本，并且 `--temp 0` 两次运行输出一致
- [ ] 改过 `ds4_cuda.cu` 的话：测试是**重新编过**的，不是旧的

## 7. 排错表

| 现象 | 原因 / 处理 |
| --- | --- |
| `nvcc: not found` | 没装 CUDA toolkit，或 PATH 没包含它 |
| `undefined reference to ds4_gpu_...`（`make cpu`） | CPU-only 构建没有 GPU 符号；本项目约定是**调用点自己包 `#ifndef DS4_NO_GPU`**，见 `ds4.c:387` 的注释 |
| `-march=native` 相关报错（ARM/Termux） | `make cpu NATIVE_CPU_FLAG=` |
| `ds4: another ds4 process is already running` | 单实例锁（`/tmp/ds4.lock`），先 `pgrep -af ds4` 看残留进程 |
| 下载中断 | 重复执行 `./download_model.sh <target>`，可续传 |
| 显存不够 | 调小 `--ssd-streaming-cache-experts`（槽位数）或 `-c` |
| 长 prompt 报 `ffn batch encode failed` | 槽位不够长 prompt 的"每层唯一专家"占用，退回 512 槽 |
| 本机脚本里 `env VAR=x cmd` 静默失效 | 这台机器 PATH 里的 `env` 不转发参数，用 shell 前缀赋值 |

## 8. 下一步读什么

| 想了解 | 读 |
| --- | --- |
| 项目全貌 / 怎么用 | [README.md](../README.md)、[README_CN.md](../README_CN.md)（中文） |
| 相对上游做了什么优化、值多少 | [docs/OPTIMIZATIONS_VS_UPSTREAM.md](OPTIMIZATIONS_VS_UPSTREAM.md) |
| 每一步优化的推导与否决方案 | [docs/OPTIMIZATION_HISTORY.md](OPTIMIZATION_HISTORY.md) |
| 本机一步步调优的操作教程 | 仓库根目录 `优化步骤教程.md` |
| 测试与调试 | [docs/TESTING.md](TESTING.md)、[CONTRIBUTING.md](../CONTRIBUTING.md) |
| 代码规范（给 AI / 给人） | [AGENT.md](../AGENT.md) |
| 模型下载与内存需求 | [docs/MODELS.md](MODELS.md) |
| SSD 流式机制 | [docs/SSD_STREAMING.md](SSD_STREAMING.md) |
| 每次改动的原始记录 | `changelog.md` |
