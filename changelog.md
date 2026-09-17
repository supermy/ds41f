# Changelog

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
