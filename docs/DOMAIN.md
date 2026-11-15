# 术语和协议

| 术语 | 定义 |
|---|---|
| Window | 原视频连续 16 帧，一次原生 Encoder 调用；窗口之间无隐藏状态 |
| Block / Group | 一个窗口中的 4 帧时间组，0..3；名义时长为 4/FPS |
| Native group width | 检查点实际布局，公开权重为 512，不能把它误当成 256 |
| Stream group width | 可发送的原生组内前 256 个离散 ID；4 组共最多 1024 个 |
| Prefix | 连续有效 ID 数；尾部已经到达但中间缺失时不会扩大有效前缀 |
| Enhancement | 仅发送 `[current_prefix,target_budget)`，不重发完整前缀 |
| Mask | `[4,256]` 分组布尔前缀，在推理边界放回原生宽度；未来组隐藏 |
| Frozen history | 实际显示时确定的最终 Mask，后续时间组必须使用它 |
| Display ready | 已完成解码，且解码所用历史前缀仍与最终历史一致 |
| Nominal deadline | 初始缓冲时间 + 原始 PTS 偏移；永久保留在日志中 |
| Scheduled deadline | 在已记录 Stall 之后的播放时间；不会抹除此前 Deadline Miss |
| Packet deadline | 本包接收端接受期限；基础恢复包可设置 NO_EXPIRY |
| RD metadata | 真实原生解码与原视频比较得到的固定历史预算曲线，仅供预编码 VOD 预测 |
| Decode time | CUDA 同步、完整原生 16 帧窗口解码的真实耗时；不声称只计算 4 帧 |

二进制 TokenPacket 的唯一结构为 `>4sBI16sIBQHHQIHI`：

| 字段 | 字节 | 含义 |
|---|---:|---|
| magic | 4 | ASCII ATKS |
| version | 1 | 1 |
| stream_id | 4 | 当前流标识 |
| model_id | 16 | 检查点 SHA256 前 16 字节 |
| window_id | 4 | 窗口序号 |
| block_id | 1 | 窗口内时间组 0..3 |
| frame_pts | 8 | 该时间组第一帧原始微秒 PTS |
| token_start | 2 | 原生组内增量起点 |
| token_count | 2 | 本包增量 ID 数 |
| deadline | 8 | 会话 monotonic epoch 后的微秒接收期限 |
| payload_length | 4 | 实际 13-bit 负载字节数 |
| flags | 2 | 0 或基础包标记 1 |
| checksum | 4 | CRC32，包头该字段置零后与负载共同计算 |
| payload | 可变 | 最高位优先的 13-bit IDs，末字节低位补零 |

TCP 消息统一使用 `uint32 大端长度 + 1 字节类型 + 内容`，长度包含类型、不包含长度字段本身。H 为元数据，P 为 TokenPacket，F 为接收/播放/解码增量反馈，C/E/D 为结束与排空握手。控制消息使用严格 JSON，视频数据只通过 P 的离散二进制负载传输，不发送浮点 Tensor 或 pickle。

基础恢复包不会修改块的 Deadline Miss 历史；迟到增强丢弃、已播放块冻结、窗口顺序退休。已播放窗口的退休水位阻止迟到包重新创建状态。
