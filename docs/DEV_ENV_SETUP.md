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
| 最小模型也要 45 GiB | **手机/平板（8–16 GB 内存）跑不动**，别指望在平板上推理 |
| 推理速度单位是 t/s | 每秒生成的 token 数；CPU 后端实测约 0.06 t/s，慢到没法聊天 |
| 典型分工 | **Debian 台式机**负责构建和运行；**手机/平板的 Termux 负责 SSH 回来干活** |
| 谁写代码 | **CodeBuddy**（终端里的 AI 编码工具）实际动手改代码；你下需求、批方案、验收结果 |

于是整个流程是这样的：**Termux 登录 → 台式机上跑 CodeBuddy → 它改代码、跑构建和测试 →
你看结果验收**。你人在哪里不重要，因为活是在台式机上干的。

本文按这个分工写：台式机是主力，Termux 是登录端，CodeBuddy 是实际干活的那个。

本 fork（`supermy/ds41f`）相对上游 [antirez/ds4](https://github.com/antirez/ds4) 只多做一件事：
把 **CUDA 单卡上的 SSD 流式推理**推到带宽上限。上游的主战场是 Metal（Mac）与 DGX Spark。
改动清单与实测幅度：[OPTIMIZATIONS_VS_UPSTREAM.md](OPTIMIZATIONS_VS_UPSTREAM.md)。

## 1. 先选一条路

| 你的设备 | 角色 | 构建目标 | 能跑真实模型吗 |
| --- | --- | --- | --- |
| **Debian 台式机 + NVIDIA 显卡** | 开发 + 运行主力 | `make cuda` | ✅ 能，带本 fork 的全部优化 |
| Debian 台式机只有 CPU | 开发 + 流程验证 | `make cpu` | ⚠️ 能跑但极慢（0.06 t/s） |
| **手机 / 平板 Termux** | **移动办公的登录端**（SSH 回台式机） | 不在这上面构建 | ❌ 内存不够 |
| Mac（Apple Silicon，96 GB+） | 上游主战场 | `make` | ✅（本 fork 没改 Metal） |

Termux 那段说的是"怎么在手机上连回台式机干活"，不是"怎么在手机上编译"。

## 2. 通用第一步：拿代码

在**台式机**上执行（Termux 不参与这一步，它是拿来登录的）：

```sh
git clone https://github.com/supermy/ds41f.git    # 本 fork
cd ds41f
```

想跟上游就 `git clone https://github.com/antirez/ds4.git`。
想给上游提改动前先读 [CONTRIBUTING.md](../CONTRIBUTING.md)。

## 3. Debian 台式机（开发 + 运行主力）

这一段在你那台装 Debian/Ubuntu 的台式机上做。有 NVIDIA 显卡就走 3.2，没有就走 3.3；
两边的第 3.4~3.5 是通用的。想在外面用手机 SSH 回来干活的，顺手把第 4 节的 4.2
（台式机开 SSH）也做一下。

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

## 4. Termux：移动办公的登录端

手机/平板的 Termux 在这里**不负责构建**——45 GiB 起的模型和它 8–16 GB 的内存凑不到一起。
它的活是：你人在外面时，SSH 回台式机，接着干活。编译、下载、跑模型都在台式机上。

### 4.1 手机上装什么

```sh
pkg update
pkg install -y openssh mosh git
```

* `openssh`：`ssh` / `ssh-keygen` / `ssh-copy-id`
* `mosh`：比 ssh 更扛移动网络——切 WiFi/4G、IP 变了也不掉线
* `git`：想直接在手机上改点小东西时用（可选）

`tmux` 要装在**台式机**上，不是手机上（会话保持跑在远端，见 4.4）。

### 4.2 台式机上开好门（只做一次）

```sh
sudo apt install -y openssh-server mosh tmux
sudo systemctl enable --now ssh          # 开机自启
ip -4 addr show scope global             # 记下局域网 IP
systemctl is-active ssh                  # 确认是 active
```

> 我这台机器上是 `192.168.0.168`（无线网卡，DHCP 分的）。**建议去路由器把 MAC 和 IP 绑死**，
> 否则重启后 IP 变了你就连不上了。想用名字而不是 IP，可以装 avahi 用 `主机名.local`。

### 4.3 用密钥登录，别用密码

在 Termux 上：

```sh
ssh-keygen -t ed25519                    # 一路回车
ssh-copy-id my@192.168.0.168             # 输最后一次密码
ssh my@192.168.0.168                     # 之后就不用密码了
```

嫌 IP 难记，在 Termux 的 `~/.ssh/config` 里写一段：

```
Host home
    HostName 192.168.0.168
    User my
```

于是 `ssh home` 就够了。

### 4.4 断线不丢工作：tmux

手机网络说断就断，直接敲命令断一次就白干。长任务一律放进 tmux 会话：

```sh
ssh home
tmux new -s build        # 开一个叫 build 的会话
make cuda                # 在会话里跑
# —— 断线了 —— 重连后：
tmux attach -t build     # 回去了，make 还在跑
```

常用的就三个：`tmux ls` 列会话、`Ctrl-b d` 脱离（断开但保留）、`tmux attach -t 名字` 回来。

### 4.5 移动网络用 mosh 更稳

```sh
mosh home
```

mosh 走 UDP（默认 60000–61000），台式机防火墙要放行这段端口。
它的好处是切网络、休眠唤醒后不用重连，敲字还有本地回显，延迟高时手感好很多。

### 4.6 不在同一个局域网时

* **同一个 WiFi**：直接用上面的局域网 IP。
* **在外面**：需要内网穿透或 VPN。图省事可以用 Tailscale（两台机器装好、登录同一账号，
  各自拿到一个固定内网 IP）——具体安装方式看它官网，别照抄来路不明的脚本。
  自建就是 WireGuard 或 frp。**把 SSH 直接暴露到公网不建议**。
* **台式机睡着了**：BIOS 里开 Wake-on-LAN，手机发个 magic packet 把它叫醒。

### 4.7 Termux 自身的小坑

* 熄屏后进程被杀：系统设置里把 Termux 加进电池优化白名单；跑长任务时先 `termux-wake-lock`
* 要访问手机上的文件：`termux-setup-storage`
* 输入体验：外接蓝牙键盘，或装带 Ctrl/方向键的输入法（如 Hacker's Keyboard）

> **这一段我无法验证**——手上没有 Android 设备。SSH/mosh/tmux 的用法是通用的，
> 但 Termux 自身的坑（4.7）来自它的常见行为。遇到具体报错把错误信息发出来。

## 5. 实际干活：用 CodeBuddy

**代码是 CodeBuddy 写的。** 它是跑在终端里的 AI 编码工具（就是现在在跟你说话的这个），
本项目本身就是 AI 重度参与开发的——上游 README 的 "AI full disclosure" 一节写明了这一点。
你的角色是**下需求、批方案、验收结果**，而不是自己一行行敲。

### 5.1 在移动办公场景里怎么跑它

CodeBuddy 是终端程序，所以正好放进前面说的 tmux 会话：

```sh
ssh home
tmux new -s work      # 开会话
codebuddy             # 在会话里启动，把需求交给它
# —— 手机锁屏、网络切换、SSH 断了 ——
tmux attach -t work   # 回来，它还在原处等你
```

好处是它改代码、跑 `make`、跑测试这一整套可能要好几分钟到几十分钟，
你不需要保持连接——挂回会话就能接着看。断点之后可以从 `/help` 看有哪些命令，
提交相关的是 `/commit`。

### 5.2 用法上的几条经验

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
6. **验收看证据，不看它的自述**：它说"提升了"不算数。要看——
   `git diff` 到底改了什么、测试过没过、实测 t/s 和落盘字节是多少、
   以及 `--temp 0` 下生成的正文**md5 是否与改动前逐字一致**。
   任何"更快但输出变了"的结果都是 bug，不是收益。

## 6. 怎么确认自己成功了

- [ ] `make cpu` 或 `make cuda` 没有 `error`
- [ ] `./ds4 --help` 打印出用法
- [ ] 有 GPU：`./tests/test_cuda_ssd_cache` 全 PASS（约 5 秒）
- [ ] 移动办公：从 Termux `ssh home` 能登上台式机，且不用输密码
- [ ] 移动办公：`tmux new -s x` → 断网重连 → `tmux attach -t x` 还在
- [ ] 交给 CodeBuddy 的活：验收时看过 `git diff`、测试、实测数字，而不只是它说"完成了"
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
| `ssh: connect to host ... port 22: Connection refused` | 台式机 sshd 没跑：`sudo systemctl enable --now ssh` |
| `Permission denied (publickey)` | 公钥没拷上：在 Termux 跑 `ssh-copy-id 用户@IP`；台式机 `~/.ssh/authorized_keys` 权限应为 600 |
| 昨天能连今天连不上 | DHCP 换 IP 了，去路由器把 MAC 和 IP 绑死 |
| `mosh` 连不上 | 台式机防火墙要放行 UDP 60000–61000 |
| Termux 熄屏就断 | 系统电池优化白名单 + `termux-wake-lock` |

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
