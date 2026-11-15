# 上游检查记录

检查日期：2026-11-15，Asia/Shanghai。工作区最初没有相关仓库、检查点或并行系统实现，现以固定提交 Git submodule 复用。

| 项目 | 版本 |
|---|---|
| Swift | `622d125e0af04a45622312361a91ee79849d47c6` |
| AdapTok | `a72076cf6474f930a181aa78971de70d65289b49` |
| 官方检查点 | `yeahhhh326/AdapTok`, `c514b22615a140c06f4c9792e93cbef9e5c79f32` |

来源：[Swift](https://github.com/SINRG-Lab/swift)、[AdapTok](https://github.com/VisionXLab/AdapTok)、[官方权重](https://huggingface.co/yeahhhh326/AdapTok)。直接阅读本地固定版本源代码和实际下载的 config / safetensors，未依赖路径猜测。

- `models/adaptok.py` 的 `encode(x, with_latent_mask=False)` 返回 `bottleneck_rep` 离散索引及完整 `encoded` Embedding。Encoder 不接收调度预算，缓存后从同一份索引取任意范围。
- `models/bottleneck.py` 的 `regularizer.decode()` 以 `embedding(indices)` 读取 Codebook；检查点启用 L2 归一化，随后 `bottleneck.project_out()` 映射至 Decoder 维数。恢复接口必须调用这一原生路径，不能把裸 Codebook 向量当作 Decoder 输入。
- `decode_with_mask(z, latent_mask)` 接收 `[B,N,D]` 和 `[B,N]` 布尔 Mask，输出 `[B,3,16,128,128]`。它仍计算完整窗口，计时包含真实 CUDA 执行。
- `decode_with_drop()` 删除 Mask 掉的 Token 并重新构造注意力；`decode_from_bottleneck()` 也默认走此路径。本项目没有启用它，没有声称与 Mask 路径数值一致。
- `models/mask_generator.py` 的分组前缀为 `arange(total_toks) < num_latents` 后按组展平。Decoder 使用 `full_causal_type2`，按 `latent_first` 重排原生注意力位置。未来 Token 扰动实验已经通过，未修改上游注意力或模型参数。
- YAML 的 Tokenizer 为 1024 IDs、每组 256；Scorer YAML 步长为 32，默认训练标签是经数据集均值/标准差标准化后的 MSE。PSNR/SSIM 标签路径还会变号，不能将输出直接等同任意质量指标。
- **实际公开 Tokenizer 检查点为 2048 IDs、每组 512；公开 Scorer 检查点也为 2048，步长为 64。** 查询当前唯一分支及之前的已发布上传提交，均未找到原生 1024 检查点。因此实际运行显式选择 native width 512，仅开放每组前 256 个位置。原生架构、权重张量和参数全保持不变。模型加载严格检查此选择，不静默适配。
- `eval/eval_adaptok.py` 调用批量评估、可选 Scorer 和整段 FVD；本系统只复用原生编解码，不引入训练、ILP、FVD 或 TensorFlow 依赖到调度器。
- Swift `streamer/common.py::load_trace()` 按两列读取秒/Mbps。本系统直接加载该函数；文件选择通过临时目录中的原始文件软链接实现，原样本与时间轴不被改写。
- Swift `streamer/simulator.py` 使用区间末端带宽、跨样本积算下载、播放缓冲和缓冲内质量升级。本项目保留这些网络驱动思想，改成 4 帧组事件模型。
- Swift `config.py` / `grad.ini` 使用 2 秒视频段、30 秒缓冲；`net.py`、`train_grad.py`、`test_grad.py` 是 TensorFlow Actor-Critic 路径，原模拟器包含 SVC 和固定有效负载比例假设。本项目不导入这些固定开销或策略。

未改动上游已跟踪源代码，不裁剪检查点，不重新训练。精确 1024 权重规格仍是未满足项，不能以本次 2048 原生模型的受限码流代替验收。
