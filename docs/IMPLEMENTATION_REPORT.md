# AdaptiveTokStream 实现与验证报告

日期：2026-11-15（Asia/Shanghai）。项目位置：`/home/whisperliang/AdaptiveTokStream`。

## 验收结论

已实现可执行的预编码 VOD 渐进 Token 编解码、二进制协议、Swift Trace 事件模拟器、三种调度器、独立进程 Sender/Receiver，以及真实 A/B/C/D 实验。**严格的原生1024-Token模型规格尚未完全验收**：官方公开权重实际为2048 Token，每组512；本次保留模型架构与全部权重，仅在原生组内使用前256个位置，最多传输1024个ID。不能将这些结果描述为原生1024检查点的结果。

### 已实现且通过运行验证

- 原生128×128、16帧 AdapTok 推理，冻结权重；每窗口只编码一次，缓存四个组各256个可传输ID。
- 任意合法组内增量、嵌套前缀、13-bit无损打包、稳定大端包头、CRC32、持久TCP长度帧。
- 去重、乱序、缺口前缀、超时增强丢弃、播放冻结、窗口退休，以及历史Mask变化后的解码失效处理。
- 真实Codebook Embedding恢复对照、原生直接解码与增量/一次接收一致；扰动未来Token不影响过去输出。
- Swift原始Trace加载；按4帧PTS驱动的网络、单Decoder和播放事件；到包与解码完成严格分离。
- FixedToken、GreedyToken、DeadlineAwareMPC，共享实际序列化尺寸、已完成传输的EWMA、Deadline、设备实测解码成本和接收状态。
- 独立Socket进程真实收发与实际接收端视频；96帧、128×128、全部原PTS和双向应用字节均核对一致。
- 101 passed in 34.90s；无失败、无跳过。核心源码合并测试/实验执行覆盖率 95.39%。
- 六种真实RD预算、24组B策略/网络组合、3组C消融，以及D真实Socket传输。

### 已实现但验证范围有限

- MPC在该96帧真实片段、一个原始Swift Trace、一个固定带宽和一个受控突降/恢复Trace上完成验证；尚不足以声称对多类长视频和大规模Trace有统计普遍性。
- RD预测固定历史预算为256，是VOD近似；实际质量已经用最终显示历史重新评价，未验证所有6^4种历史组合的预测精度。
- TCP仅验证同机独立进程，使用共同monotonic clock；没有跨主机时钟同步或真实WAN整形实验。
- 模拟器计量视频方向的TokenPacket及TCP外帧；真实Socket另外计量控制元数据与反向反馈，没有验证双向共用一个瓶颈时的QoE。
- Byte counter为应用层真实序列化字节，不包括TCP/IP、重传段或以太网头。
- 变异检查保留 39 个存活项，未伪称100%变异分数；审计见下文。

### 由于模型权重限制未完成

- 原生4×256=1024布局的官方预训练权重未找到；两次公开上传及当前分支均是4×512。加载器拒绝把它静默当作1024模型。需要真实兼容检查点才能完成严格规格的最后验证，不能靠裁剪参数或随机权重补齐。
- YAML Scorer步长32；实际公开Scorer步长64，输出为标准化标签预测，未做精度校准，未用于调度。任务指定的VOD实测RD路径已经完成。

## 复用与主要修改

上游以固定Git submodule保留，未修改已跟踪模型/训练源代码：

- AdapTok `a72076cf6474f930a181aa78971de70d65289b49`：`models/adaptok.py` 的encode/decode_with_mask、`models/bottleneck.py` 的decode、原生分组注意力/Mask、Transformer及Embed模块。
- Swift `622d125e0af04a45622312361a91ee79849d47c6`：直接调用 `streamer/common.py::load_trace()`，参考 `simulator.py` 的跨Trace样本积算、缓冲和质量升级。未移植TensorFlow策略、2秒分段或SVC开销常量。
- 官方权重 `yeahhhh326/AdapTok` revision `c514b22615a140c06f4c9792e93cbef9e5c79f32`，SHA256 `be5432949a87dee253c86f7ccb87bb5370b435ed180f8cda546b09d6a63c8cdd`。

| 文件 | 职责 |
|---|---|
| `src/adaptivetokstream/codec.py` | AdapTokBackend、ProgressiveVideoEncoder/Decoder、严格输入与原生槽位映射 |
| `protocol.py` / `buffer.py` | TokenPacker、TokenPacket、TokenBuffer、计数、冻结和清理 |
| `media.py` / `asset.py` | 原FPS/PTS、128×128准备、末窗补齐、单次编码资产 |
| `metrics.py` / `profiling.py` / `evaluation.py` | 实测PSNR/LPIPS/DISTS、RD/计时、最终历史重放及实际接收文件评估 |
| `network.py` / `simulator.py` | Swift读取、真实包大小的网络积分和一致事件时钟 |
| `scheduler.py` | 固定、收益/字节Greedy、有限Beam MPC和解码成本反馈 |
| `playback.py` | 双端与模拟器复用的接收、解码完成、Deadline和播放状态机 |
| `transport.py` / `socket_runtime.py` | 单TCP帧协议、状态增量反馈、流水发送、排空握手、独立双端 |
| `config.py` / `cli.py` / `experiments.py` | 唯一配置、统一CLI、B/C/D可复现实验 |
| `tests/` | 协议、Buffer、网络、调度、播放、真实模型、独立Socket、完整19帧CLI流程 |
| `scripts/fetch_weights.py` / `quality_audit.py` | 固定权重下载、复杂度/覆盖率CRAP审计 |
| `configs/default.yaml` / `requirements.lock` | 运行参数和本次实际依赖版本 |

新增接口：`encode_window(frames, window_id=None)`、`get_prefix(window_id,block_id,start,end)`、`append_tokens(packet,now_us)`、`decode_block(window_id,block_id)`、`finalize_block(window_id,block_id)`；组内预算保持 `[8,16,32,64,128,256]`。协议详情见 `DOMAIN.md`，原始API和检查点差异见 `UPSTREAM_AUDIT.md`。

## 实际测试与质量检查

```bash
ADAPTOK_DEVICE=cuda:1 .venv/bin/python -m pytest -q --cov=adaptivetokstream --cov-report=json:artifacts/coverage.json --cov-report=xml:artifacts/coverage.xml --cov-report=term-missing
.venv/bin/python -m pytest tests/test_network.py -q
.venv/bin/ruff check src tests scripts
.venv/bin/radon cc src -j
.venv/bin/mutmut run --max-children 4 'adaptivetokstream.protocol.*'
npx --yes jscpd@4.0.5 src/adaptivetokstream --min-tokens 80 --reporters json --output artifacts/duplication
.venv/bin/python scripts/quality_audit.py
```

最终主套件（重新收集覆盖率，没有追加旧版本数据）：**101 passed in 34.90s**。网络导入清理后另外7项通过；最后的等价布尔表达式整理后，播放/网络/模拟器14项回归通过。早期TDD缺失模块失败及修复日志保留，最终没有失败测试。硬件为RTX A6000 48GB，Python3.10，PyTorch2.5.1+cu121；完整版本见 `requirements.lock`。

`crapper/dryer/mutator`名称工具不可用，实际使用radon+coverage、jscpd和mutmut作为对应检查。172个核心函数的最大圈复杂度9，最大CRAP 9.728，无CRAP≥10项。jscpd在80-token阈值发现0处重复块。协议变异共239项，杀死200项；其余36项仅改变ValueError错误文字，3项被后续约束等价拒绝。完整存活diff与人工分类在 `artifacts/mutation_audit.json`；没有通过降低协议约束掩盖功能问题。

最后一次覆盖率合并限定在系统源码，排除PyTorch动态生成且已删除的临时模块。测试覆盖率与实验执行覆盖率在记录中分别可追溯，不把未执行路径说成已验证。

## 实验A：真实渐进率失真

输入为本机已有真实 `RECAP/video_data/road.mp4` 的前96帧，30FPS，共3.2秒；缩放/裁剪后所有模型输入输出均为128×128。没有修改源文件，没有合成质量。PSNR按4帧块的dB均值报告，LPIPS为AlexNet，DISTS为官方VGG16版本；参考/输出均以RGB [0,1]评价。

| 每组预算 | 96帧实际视频包字节 | 平均 PSNR (dB) | LPIPS | DISTS | Decoder P50/P95 (ms) |
|---:|---:|---:|---:|---:|---:|
| 8 | 1872 | 13.714 | 0.6967 | 0.4609 | 88.67/88.80 |
| 16 | 2184 | 14.874 | 0.5359 | 0.3659 | 88.85/89.07 |
| 32 | 2808 | 15.697 | 0.3575 | 0.2862 | 89.31/89.45 |
| 64 | 4056 | 19.194 | 0.2181 | 0.2006 | 89.40/89.58 |
| 128 | 6552 | 20.799 | 0.1549 | 0.1546 | 89.48/89.61 |
| 256 | 11544 | 22.555 | 0.1000 | 0.1334 | 89.51/89.67 |


字节为每块一次性发送所选预算的真实TokenPacket+长度/类型帧，B中的多次增量则按各实际包分别计数。表中均为完整原生16帧窗口Decoder耗时，CUDA同步、3次warmup，每预算30个计时样本；低Token预算几乎不减少计算。

预编码6个窗口共 0.664 秒（Encoder调用计时，模型加载另计）；RD/Profiling预计算 32.962 秒，JSON元数据 66541 字节；缓存Token数组文件12416字节。预编码、RD和质量模型初始化不计入VOD在线延迟。Profiling进程峰值为 1024508416 字节，包含当时驻留的质量网络；接收端峰值另见D。

## 实验B：网络自适应

共24组，使用同一源视频、真实包格式、同一Decoder设备Profile。原始Swift Trace未改写；另测0.02Mbps固定链路和明确标记为受控条件的突降/恢复Trace。调度器只接收已完成传输反馈，不接触未来Trace。

| 策略（突降/恢复） | 实际视频包字节 | PSNR (dB) | LPIPS | DISTS | Stall (s) |
|---|---:|---:|---:|---:|---:|
| fixed_8 | 1872 | 13.714 | 0.6967 | 0.4609 | 0.238 |
| fixed_16 | 2184 | 14.874 | 0.5359 | 0.3659 | 0.294 |
| fixed_32 | 2808 | 15.697 | 0.3575 | 0.2862 | 0.368 |
| fixed_64 | 4056 | 19.194 | 0.2181 | 0.2006 | 0.373 |
| fixed_128 | 6552 | 20.799 | 0.1549 | 0.1546 | 0.383 |
| fixed_256 | 11544 | 22.555 | 0.1000 | 0.1334 | 0.404 |
| greedy | 13845 | 21.056 | 0.1897 | 0.1806 | 0.238 |
| deadline_mpc | 4576 | 20.629 | 0.2112 | 0.1977 | 0.238 |


Greedy在本短片的部分条件下花费更多包头开销且未获得固定256的质量；MPC不保证支配全部基线。完整PSNR/LPIPS/DISTS、码率、卡顿次数/时长、画质波动、Deadline Miss、播放延迟和调度耗时均在每组evaluation.json及汇总CSV。

## 实验C：相同实际字节预算消融

| 消融（共同上限4000字节） | 实际消耗字节 | PSNR (dB) | LPIPS | DISTS | Stall (s) |
|---|---:|---:|---:|---:|---:|
| bandwidth_only | 3978 | 15.365 | 0.5749 | 0.3913 | 0.238 |
| bandwidth_content | 3978 | 19.603 | 0.2425 | 0.2170 | 0.238 |
| bandwidth_content_decode | 3978 | 19.603 | 0.2425 | 0.2170 | 0.238 |


三组共同约束4000实际序列化字节，且恰好均消耗3978字节，未事后缩放吞吐或质量。内容质量预测提高了该片段的重建质量；本片段中加入Decoder Time没有额外改善，不能据此声称普遍收益。实际Decoder耗时在全部消融运行中始终计入模拟时钟，只对调度决策开关做消融。

## 实验D：实际Socket与输出文件

- 视频包发送/接收：**4576/4576字节**，33包，重复有效负载0；逐组预算见 `receiver.json::playback`，没有重复下载整个前缀。
- 含元数据/结束控制的前向应用字节：**5527/5527**；反向状态反馈/控制：**43580/43580**。双向均严格一致，反馈开销没有隐藏在视频字节内。
- 实际接收端文件96帧、128×128，全部96个原PTS一致，使用FFV1/NUT无损输出。
- PSNR **20.628dB**，LPIPS **0.2112**，DISTS **0.1977**，来自实际接收文件，包含最终像素量化。
- 视频方向会话平均吞吐 **8.485kbps**；每个视频包从请求至Socket写入平均 **47.16ms**，包含应用Trace节奏控制，不冒充纯网络单向时延。
- Decoder平均/P50/P95：**91.08/91.19/91.57ms**，实际执行27次完整窗口解码；此为覆盖率插桩运行，不等同未插桩极限性能。
- 两次Stall，总 **0.241s**；Deadline Miss Rate **8.33%**；平均按PTS播放延迟 **1.199s**。
- 包括初始1秒缓冲和卡顿后的媒体呈现区间 **4.441s**，有效 **21.62FPS**。最后一块输出时在线循环已完成，媒体呈现区间另外包含该块持续时间。
- 接收进程GPU峰值 **959303168字节**（0.893GiB）。MPC平均/最大决策时间 **2.11/14.03ms**。

此演示是实际TCP Socket加应用层Trace节奏控制，没有声称操作系统链路已被整形。持久连接下数据流水发送，反馈异步更新；TCP可靠传输仍有队头阻塞，已写入的数据不能撤回，过期/已播放增强只在接收端丢弃并计量。

## 数据位置和复现

- `artifacts/road/prepared/`：真实128×128参考帧、FPS及原PTS。
- `artifacts/road/encoded/`：单次编码Token、模型SHA256和编码时间。
- `artifacts/road/profile/`：RD曲线、六个预算视频/PNG、Experiment A CSV和计时样本。
- `artifacts/road/experiments/results.csv`：27组B/C汇总；子目录各有事件、包、播放、真实质量和重建文件。
- `artifacts/road/socket_D/experiment_d.json`：D汇总；`receiver/reconstruction.nut`为实际接收视频，双端日志和完整包/播放记录均保留。
- `artifacts/final_tests.txt`、`coverage_report.txt`、`quality_audit.json`、`mutation_audit.json`、`duplication/jscpd-report.json`：实际检查证据。

主要复现命令见README的7个必需CLI入口，以及 `experiments`、`socket-demo`。配置以每个输出目录的 `effective_config.json`、实验记录中的config或D的 `run_config.yaml` 为准。

## 当前瓶颈与边界

1. 完整窗口Mask Decoder约89–91ms，即使只显示4帧也需要计算16帧；重复增强及历史失效可能增加Decoder任务。本次没有启用未验证的with_drop优化。
2. 状态增量反馈已替代周期全量反馈，但本短片仍有43580字节反向控制，相对低Token视频开销明显；部署前应进一步评估控制开销和真实双向网络条件。
3. 65KB级逐块RD元数据与约33秒离线预计算只适用于本次预编码VOD，不是可在线使用的实时Scorer。
4. 缺少原生1024检查点是严格验收的主要外部限制；本系统未重新训练、裁剪权重、伪造质量或利用未来网络信息。
5. 尚无大样本内容/Trace统计、真实丢包、跨主机同步和WAN数据；本报告只主张已执行的结果。

## ub-mode执行记录与交付状态

按“先跑通→校正根因→遵循实践”推进；先失败测试再实现协议/Buffer、Codec、媒体、网络、调度、事件和TCP，并以真实模型/视频验证。每项动作及理由见 `WORK_LOG.md`。保留有用实验、模型和源数据，仅清理本次生成的变异缓存及无效临时缓存；没有并行维护旧实现或兼容层。

代码已在main分阶段提交，并按用户后续要求推送至私有仓库 [whisperLiang/AdaptiveTokStream](https://github.com/whisperLiang/AdaptiveTokStream)，本地main跟踪origin/main。严格1024模型项继续标记为未完成，其余已实现能力与有限验证范围分别如上。
