# AdaptiveTokStream

预编码视频点播系统：真实 AdapTok 离散 Token → 13-bit 码流 → 持久 TCP → 接收端原生 Decoder → 128×128 视频。包含 Swift Trace 驱动的事件模拟器、FixedToken / GreedyToken / DeadlineAwareMPC、真实 RD Profiling 和实验 CLI。

**模型规格存在已核实的上游差异。** 仓库 YAML 是 1024 Token，但官方公开权重实际是 2048 Token（4×512）。本项目不裁剪权重、不改 Transformer：每组仅传输原生位置中的前 256 个 ID，接收端放回原来的 512 槽位，剩余槽位 Mask 为 False。传输上限仍为 4×256=1024。此运行结果不能冒充“原生 1024-Token 权重”的验证；有对应权重时，把 `native_tokens_per_block` 改为 256，接口会严格核对布局。详见 [上游检查](docs/UPSTREAM_AUDIT.md)。

## 安装

```bash
cd /home/whisperliang/AdaptiveTokStream
# 克隆项目后先获取固定版本上游；当前工作区已经准备完成。
git submodule update --init --recursive
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -e '.[codec,dev]'
.venv/bin/python scripts/fetch_weights.py
```

官方权重需同时具有 `config.json` 和 `model.safetensors`。缺失文件、布局不符或校验失败会明确报错。不会使用随机模型替代。LPIPS 的 AlexNet、DISTS 的 VGG16 和 DISTS 权重由相应官方库加载；首次运行可能下载其预训练参数。本次环境已具备这些权重。

唯一配置为 [configs/default.yaml](configs/default.yaml)；YAML 中路径相对于 YAML 文件，CLI 中路径相对于当前目录。`--weights`、`--device`、`--seed` 可统一覆盖。`requirements.lock` 记录本次实际安装版本。

## 运行

以下输入是本机已有真实视频，只读使用。也可以替换为自己的视频文件。

```bash
.venv/bin/python -m adaptivetokstream prepare-video --input /home/whisperliang/RECAP/video_data/road.mp4 --frames 96 --output artifacts/road/prepared
.venv/bin/python -m adaptivetokstream encode-video --prepared artifacts/road/prepared --output artifacts/road/encoded
.venv/bin/python -m adaptivetokstream profile-codec --asset artifacts/road/encoded --output artifacts/road/profile
.venv/bin/python -m adaptivetokstream simulate --asset artifacts/road/encoded --rd artifacts/road/profile/rd.json --trace third_party/swift/streamer/datasets/test/oboe_trace_2 --policy deadline_mpc --output artifacts/road/simulation
.venv/bin/python -m adaptivetokstream evaluate --asset artifacts/road/encoded --log artifacts/road/simulation/simulation.json --output artifacts/road/evaluation
```

`prepare-video` 默认读取源 FPS 和逐帧 PTS，按上游评估方式缩放短边、中心裁剪至 128×128；`--fps 30` 会显式覆盖时间轴，同时保留原 PTS 元数据。末尾不足 16 帧仅在编码时重复末帧，输出严格截回原帧数。不同预算共用磁盘中同一次编码生成的 Token。

`simulate` 支持 `--policy fixed --token-budget 64`、`--policy greedy`、`--policy deadline_mpc`、`--bandwidth-mbps 0.02`、`--trace <原始Trace文件>`、`--byte-cap 4000` 和 `--decode-scale 2`。抖动、RTT、初始带宽估计、缓冲上限、MPC horizon / beam 和消融开关在统一 YAML 中配置。模拟器直接调用 Swift 的 `common.load_trace()`，保留原始秒/Mbps 样本与区间末端带宽约定，不继承 2 秒分段、SVC 开销或 TensorFlow 策略。

真实双端演示在两个终端分别执行：

```bash
.venv/bin/python -m adaptivetokstream run-receiver --host 127.0.0.1 --port 8765 --output artifacts/manual_receiver
.venv/bin/python -m adaptivetokstream run-sender --host 127.0.0.1 --port 8765 --asset artifacts/road/encoded --rd artifacts/road/profile/rd.json --policy deadline_mpc --output artifacts/manual_sender
```

接收端仅加载自己的模型和网络收到的元数据/Token，不读取发送端的 Token 数组。解码线程只接收不可变任务快照；Socket、TokenBuffer、播放状态均由接收端主循环单一写入。重建文件使用 NUT 容器中的无损 FFV1，保留微秒 PTS，可用 FFmpeg / ffplay 读取。

```bash
ffplay artifacts/manual_receiver/reconstruction.nut
```

自动创建两个独立进程并核对字节与视频的 Experiment D：

```bash
.venv/bin/python -m adaptivetokstream socket-demo --asset artifacts/road/encoded --rd artifacts/road/profile/rd.json --trace configs/traces/drop_recovery.txt --device cuda:1 --output artifacts/road/socket_D
```

`--trace` 在真实 Socket 上执行应用层发送节奏控制，并非内核网络整形。数据可以在前一包传播期间继续排队传输，不为每个增量增加连接握手。双端演示使用同一机器的 monotonic clock；尚未验证跨主机时钟同步。

## 实验与结果

- A：`profile-codec` 生成 `experiment_a.csv`、`rd.json`、`profile_summary.json` 和六种预算的真实重建视频/代表帧。
- B/C：下面命令运行 3 种网络条件 × 8 种策略，以及同一实际序列化字节上限下的 3 组消融。每组质量均重新以最终播放历史 Mask 解码评估。
- D：`socket-demo` 的质量来自实际接收端文件，核对全部源 PTS、尺寸、帧数和双向应用字节。

```bash
.venv/bin/python -m adaptivetokstream experiments --asset artifacts/road/encoded --rd artifacts/road/profile/rd.json --output artifacts/road/experiments
```

已运行结果在 `artifacts/road/`，汇总见 [实现与验证报告](docs/IMPLEMENTATION_REPORT.md)。该目录包含模型、真实重建和完整日志，保留于磁盘并由 Git 忽略；没有生成模拟质量或随机模型实验。

## 协议与时间语义

- 13-bit ID 按最高位优先连接；最后一个字节的低位补零，非零填充被拒绝。
- TokenPacket 包头固定 60 字节，所有多字节整数使用大端；CRC32 覆盖 CRC 置零后的完整包头及负载。字段表见 [术语与协议](docs/DOMAIN.md)。
- TCP 外帧为 4 字节长度 + 1 字节消息类型。所有决策通过实际 `serialize()` 得到包大小，不用 Token 数乘一个估计开销。
- 基础包的协议过期时间为 `NO_EXPIRY`，允许在卡顿恢复后使用；块的播放截止时间独立记录，错过时必须产生 Stall。增强包在请求时的播放截止时间过期后丢弃。
- 播放开始即冻结实际显示的前缀；已收到但未完成解码的增强不会计入显示质量。历史预算变化时，旧的未来块重建失效，必须重新解码。
- 模拟器按同一时间轴排队传输、接收、单 Decoder 计算与播放。解码成本来自设备 Profiling；质量仅由后续真实解码评估生成。
- 有效视频码率按实际有效负载字节计算；传输码率包含视频包头及外帧。真实 Socket 另外记录元数据和反向反馈字节，两端总数必须一致。计数为应用层字节，不声称包含 TCP/IP 或网卡帧头。

## 测试与质量检查

```bash
ADAPTOK_DEVICE=cuda:1 .venv/bin/python -m pytest -q --cov=adaptivetokstream --cov-report=json:artifacts/coverage.json
.venv/bin/ruff check src tests scripts
.venv/bin/radon cc src -s
.venv/bin/mutmut run --max-children 4 'adaptivetokstream.protocol.*'
npx --yes jscpd@4.0.5 src/adaptivetokstream --min-tokens 80 --reporters json --output artifacts/duplication
.venv/bin/python scripts/quality_audit.py
```

真实模型与端到端测试在没有所需权重、CUDA 或参考视频时明确跳过并说明原因；普通协议、Buffer、网络和调度测试不依赖 CUDA。测试中的合成 Token / RD 表只用来验证协议和时序行为，不参与任何实验质量结果。

目前仅实现预编码 VOD、完整窗口 Mask Decoder 和单 TCP 连接。未实现 4 帧增量 Encoder、KV Cache、Tokenizer 训练、熵编码、FEC 或真实丢包恢复。TCP 可靠传输会保留队头阻塞，已写入 Socket 的增强不能撤回。静态 RD 曲线固定过去组预算为 256，是内容感知近似，不是已验证的在线 Scorer。
