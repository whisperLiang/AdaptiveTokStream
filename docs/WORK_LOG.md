# Execution log

2026-11-15 (Asia/Shanghai)

- Read the complete task and ub-mode skill before implementation.
- Searched the workspace and /home, /srv, /tmp: no Swift/AdapTok/AdaptiveTokStream checkout or checkpoint found. Existing unrelated projects and data left intact.
- Checked hardware: two RTX A6000 48 GB, CUDA driver 580.126.09; system Python has no PyTorch. Created a project-local Python 3.10 virtual environment.
- Created an independent repository; cloned unmodified upstreams to third_party. Read actual codec, masks, bottleneck, configs, evaluation and Swift trace, simulator, config, training/testing modules.
- Fixed upstream revisions: AdapTok a72076cf6474f930a181aa78971de70d65289b49; Swift 622d125e0af04a45622312361a91ee79849d47c6. Official checkpoint revision c514b22615a140c06f4c9792e93cbef9e5c79f32.
- Confirmed encode(with_latent_mask=False) exposes bottleneck_rep; bottleneck.decode restores normalized codebook vectors then projects to decoder dimension; decode_with_mask takes [B,1024,D] and [B,1024] mask. decode_from_bottleneck uses with_drop, so it is excluded from the production path.
- Scorer selects positions 31::32, trained against normalized scores (default MSE), and has public weights. It is not a validated online quality estimator. Use measured VOD metadata only.
- Swift common.load_trace is dependency-free and will be imported directly. Preserve timestamp/Mbps columns and interval-end convention. Exclude its 2-second chunks, TensorFlow policy, synthetic sizes and SVC/payload overhead assumptions.
- Searched available tools and PATH for crapper/dryer/mutator: unavailable. Will use available complexity/coverage, duplicate-code and mutation tools, reporting actual checks and limitations.
- Begin protocol/buffer with failing behavioral tests (ub-mode).
- Protocol/buffer tests first failed on missing modules (artifacts/tdd_protocol_red.txt), then all 32 passed; added strict MSB-first 13-bit coding, CRC32 headers and receiver ownership, overlap checks, reordering, freeze/expiry and retirement.
- Important upstream mismatch: both public checkpoint revisions have 2048 IDs (512/group); public scorer step is 64. No public exact-1024 checkpoint found. Preserved weights/architecture, explicitly configured native width 512 and restricted transport to first 256/group in original native slots. Exact-1024 validation remains unfulfilled without compatible weights.
- Codec tests first failed on missing codec module; actual frozen official model passed 4 CUDA tests, including independent future-group perturbation and progressive/one-shot equality. Full suite at this stage: 36 passed, 5.18s.
- Cleared corrupt bytecode generated only in this task's newly created virtual environment after a NumPy upgrade; did not alter existing environments or user caches.
- Added PTS-preserving NUT/FFV1 output, source-FPS preparation, last-window padding and asset persistence; media roundtrip test passed for 19 frames with nonzero PTS origin.
- Began real RD profiling on 96 frames from the existing user video RECAP/video_data/road.mp4, resized/cropped to 128x128. Original file is read-only and retained.
- Real 96-frame profiling completed in 32.96 seconds (quality-model initialization included). Six fixed-budget videos and frames saved. CUDA-synchronized full-window P50 is about 89 ms across all budgets; low token count does not reduce mask-decoder compute. RD curves explicitly fix historical groups at 256; final playback is re-evaluated from displayed masks.
- Added direct Swift common.load_trace reuse via a temporary symlink to original samples, exact piecewise network integration, persistent pipelining and feedback-only EWMA. Seven network tests passed.
- Added FixedToken, GreedyToken and bounded-beam DeadlineAwareMPC with serialized sizes, deadline/decode constraints and explicit base recovery. Seven scheduler tests passed.
- Added shared playback state machine and event simulator: receipt, decode completion and display are separate; stale historical decodes are invalidated, displayed prefixes frozen, missed deadlines cause explicit stalls. Synthetic unit fixtures are used only for state/ordering tests, never as experimental quality. Six playback/simulator tests passed.
