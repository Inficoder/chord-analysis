# 和弦分析工具 — 设计文档

## 概述

面向音乐人/编曲者的 Web 扒谱工具。上传音频文件，自动分析调性、和弦、歌词、和声功能，以带时间轴的和弦谱形式展示。

---

## 架构

```
Browser (React + TS)
  │ HTTP REST (polling)
  ▼
FastAPI Backend
  ├── POST /api/upload       → analysis_id
  ├── GET  /api/status/:id    → 进度
  ├── GET  /api/result/:id    → 202 (processing) | 200 (完整 JSON)
  └── GET  /api/export/:id    → PDF/TXT 导出

Analysis Pipeline (异步):
  Audio ──→ Demucs Vocal Sep ──→ Vocals → WhisperX → Lyrics (逐句时间戳)
      │                                            (无人声时返回空数组)
      │
      ├──→ HPSS / harmonic component (可选，Phase 2)
      │
      └──→ full mix (原始音频) ──┐
                                 │
  ┌──────────────────────────────┘
  │  MERT-330M / MusicFM Backbone (共享，一次前向传播)
  │
  ├── Beat/Downbeat Head (高时间分辨率特征) → 节拍/强拍位置
  │       │
  │       └── beat map (每个 beat 的时间、小节、拍内位置)
  │
  ├── Chord Head:
  │     beat-synchronous feature pooling (按 beat 聚合 frame features)
  │     → root prediction (12-class)
  │     → quality prediction (22-class, 含 N)
  │     → bass prediction (12-class, Phase 1.5)
  │     → Learned Viterbi / semi-Markov decoding (数据驱动转移先验)
  │
  └── Key Head (长时间 pooling + attention):
        → global key (24-class)
        → local key segments (分段调性)
        → multi-source fusion:
            SSL posterior + chroma K-S + bass profile +
            chord progression evidence + cadence evidence

  Rule Engine → chord seq + local key → 罗马数字/功能标记 (I, IV, V7, bVII, V/V...)
```

**关键设计：**

- 三任务共享 backbone，但使用不同时间尺度的特征：Beat 用高分辨率帧级，Chord 用 beat-synchronous pooling，Key 用全局/段级 pooling
- Chord Head 分层预测：先 root，再 quality，再 bass；输出 top-k alternatives
- Viterbi 转移矩阵从标注数据中学习，不硬编码古典和声先验；支持 semi-Markov duration model
- 调性检测输出全局 key + 局部 key segments；融合 SSL/chroma/bass/cadence 多源信息
- Phase 1: 独立模块；Phase 2: 联合微调 + HPSS 多流输入

---

## 数据模型

### 和弦词汇表

和弦状态 = root (12 pitch class) × quality + optional bass (12 pitch class)

**Quality 词汇表 (22 类):**

```
N         无和弦/静音/仅鼓/无明确和声
maj, min, dim, aug,
sus2, sus4,
6, min6,
7, maj7, min7, m7b5, dim7, minMaj7,
add9, 9, maj9, min9,
11, 13,
alt
```

对外显示时结合调性做 enharmonic spelling（如 C# major 调中显示 C#，Db major 调中显示 Db）。

Bass note 在 Phase 1 中作为结构化字段预留，Phase 1.5 实现独立的 Bass Head（12-class 分类）。
Phase 1 实际输出中 bass 字段为 null。slash chord 显示（如 `C:maj/E`）留待 Phase 1.5。

### 核心类型

```typescript
interface AnalysisStatus {
  id: string;
  filename: string;
  status: "uploading" | "queued" | "processing" | "done" | "error";
  progress: number;  // 0-100
  stages: {
    vocal_sep: boolean;
    beat_tracking: boolean;
    chord_detection: boolean;
    key_detection: boolean;
    lyrics: boolean;
    harmony_analysis: boolean;
  };
  error?: string;
}

interface BeatPoint {
  time: number;           // 秒
  beat_index: number;     // 全局节拍序号
  bar_index: number;      // 小节序号
  beat_in_bar: number;    // 小节内拍号 (1-based)
  is_downbeat: boolean;
  confidence: number;
}

interface KeyResult {
  key: string;             // "C major" | "D minor"
  confidence: number;      // 0-1
  method: "fused" | "ssl" | "ks" | "cadence";
  alternatives: { key: string; confidence: number }[];
}

interface KeySegment {
  start: number;           // 秒
  end: number;
  key: string;
  confidence: number;
}

interface ChordAlternative {
  label: string;           // "C:maj" "Am:min" "N"
  confidence: number;
}

interface ChordSegment {
  index: number;
  start: number;           // 秒
  end: number;

  // Structured chord fields
  label: string;           // "C:maj" "G:7" "N" — display label
  root: string;            // "C" "G" — pitch class, or "" for N
  quality: string;         // "maj" "7" — quality, or "N"
  bass: string | null;     // bass note pitch class, null if no inversion (Phase 1)
                           // Phase 1.5+ populated by Bass Head

  // Beat alignment
  beat_start: number;      // beat index
  beat_end: number;
  bar: number;             // bar index

  // Harmonic analysis
  roman: string;           // "I" "V7" "V/V" "bVII"
  local_key: string;       // key context used for this chord's roman analysis
  function: string;        // "tonic" | "predominant" | "dominant" | "borrowed" | ...

  confidence: number;
  alternatives: ChordAlternative[];  // top-3 alternative labels
}

interface LyricLine {
  start: number;
  end: number;
  text: string;
}

interface TimeSignature {
  value: string;           // "4/4" "3/4" "6/8"
  confidence: number;
}

interface AnalysisResult {
  id: string;
  global_key: KeyResult;
  key_segments: KeySegment[];
  chords: ChordSegment[];
  lyrics: LyricLine[];
  beats: BeatPoint[];
  duration: number;
  tempo: { bpm: number; confidence: number };
  time_signature: TimeSignature;
}
```

---

## API 设计

```
POST /api/upload
  multipart/form-data:
    file:          .mp3, .wav, .m4a, .flac, .ogg
    output_lang:   "zh" | "en" | "auto"
  → 201 { analysis_id, status_url }

GET /api/status/:id
  → { status, progress, stages }

GET /api/result/:id
  分析完成 → 200 AnalysisResult
  处理中   → 202 { status: "processing", progress: N }
  不存在   → 404 { error: "not found", code: "NOT_FOUND" }

GET /api/export/:id?format=pdf|txt
  完成 → 200 文件下载
  未完成 → 409 { error: "not ready", code: "NOT_FOUND" }
```

文件限制：

- 最大 50MB
- 最大 10 分钟时长
- 单任务超时 5 分钟
- 结果 TTL：1 小时

### 错误响应

```json
{ "error": "描述信息", "code": "INVALID_FORMAT" | "FILE_TOO_LARGE" | "NOT_FOUND" | "ANALYSIS_FAILED" }
```

---

## 前端设计

### 技术栈
- React 18 + TypeScript
- Vite (构建)
- React Router v6 (路由)
- Tailwind CSS (样式)
- fetch + 轮询 (无额外状态库)

### 页面路由
```
/              → HomePage      (上传)
/analyze/:id   → AnalyzePage   (进度 + 结果)
```

### 组件树
```
App
├── HomePage
│   └── UploadZone           # 拖拽上传 + 语言选择
│
└── AnalyzePage
    ├── ProgressPanel        # 流水线阶段进度
    ├── ErrorPanel           # 分析失败时显示错误信息
    │
    └── (结果展示区)
        ├── ResultToolbar    # 调性/BPM/拍号 + 和弦/功能切换 + 导出
        ├── TimelineSidebar  # 左侧时间轴，小节/拍号网格
        └── ChordChart       # 主和弦谱视图
            └── ChordLine[]  # 每行: 和弦标签 → 歌词 → 罗马数字
```

### 和弦谱视图 (方案 C)
- 左侧时间轴：小节网格 + 拍号标注
- 和弦标签：根据时间戳对齐到对应歌词位置
- 罗马数字标注：每行和弦下方显示功能标记，toolbar toggle 开关
- 低置信度和弦：虚线/浅色显示，悬浮显示 alternatives
- 段落标记：Verse/Chorus 标签
- 导出：PDF / TXT
- 手动编辑：点击和弦可修改 label，拖动边界可调整位置（Phase 1.5）
- 和弦与歌词的对齐精度：按句对齐（line-level），非逐字精确对齐

---

## 后端流水线

### 阶段依赖

```
Vocal Sep ──→ Lyrics (依赖人声轨)

Beat Track ──→ Chord Detect ──→ Key Detect ──→ Harmony
(依赖原始音频)  (依赖 beat map)   (依赖和弦序列 + 段级特征)
```

所有 SSL 任务（Beat/Chord/Key）均使用原始完整音频，共享 backbone 但使用不同时间尺度的特征。
Vocal Sep 和 Beat Track 可并行启动。

### 流水线模块

| 模块 | 输入 | 输出 | 方法 |
|------|------|------|------|
| vocal_sep | 音频文件 | 人声轨 + 伴奏轨 | Demucs htdemucs |
| beat_track | 原始音频 | beat map (BeatPoint[]) | SSL backbone 浅层特征 + BeatHead (TCN) |
| chord_detect | 原始音频 + beat map | 和弦序列 (含 alternatives, bass=null) | SSL backbone 中层特征 + beat-sync pooling + ChordHead + learned Viterbi |
| key_detect | backbone 特征 + beat map + 和弦序列 | global key + key_segments | KeyHead (24-class) + multi-source fusion |
| lyrics | 人声轨 | 逐句歌词+时间戳 | WhisperX (large-v3, 支持中文) |
| harmony | 和弦序列 + key_segments | 罗马数字标注 | 规则引擎 (local-key-aware) |

time_signature 由强拍间隔统计得出。对 4/4、3/4、6/8 做区分，输出置信度。
beat map 输出每个 beat 的详细信息（时间、小节、拍内位置、是否强拍）。

无人生音频：WhisperX 输出大概率无有效文本 → lyrics 返回空数组 `[]`，和弦谱视图仅显示和弦行。

### Chord Head 设计

```
frame-level backbone features (T_frame, D)
  ↓
beat-synchronous pooling (按 beat 区间取 mean/max)
  ↓
per-beat feature vectors (N_beats, D)
  ↓
├── root classifier (Linear, 12-class + no-root ≡ 13)
├── quality classifier (Linear, 22-class)
└── bass classifier (Linear, 12-class, Phase 1.5)
  ↓
per-beat chord posteriors
  ↓
learned Viterbi / semi-Markov decoding
  (转移矩阵从标注数据中学习；duration model)
  ↓
smoothed chord sequence + top-k alternatives + N detection
```

### Viterbi 解码（改进版）

- 转移矩阵从标注语料中学习（log bigram counts + smoothing），不硬编码古典和声先验
- semi-Markov duration model：对每种和弦学习最小/典型持续时长分布
- downbeat-aware boundary prior：强拍位置和弦变化概率更高
- 输出 top-3 alternatives 而非仅最优路径
- N（no-chord）作为合法状态参与解码，避免静音/鼓段被强制赋予和弦

### 调性检测多源融合

1. SSL KeyHead：backbone 全局/段级 pooling → 24 类分类
2. Chroma K-S：全曲/段落 chroma distribution × 24 调性模板 → 相关系数
3. Chord profile：和弦根音 duration-weighted distribution × 模板
4. Bass profile：低音 pitch class distribution × 模板
5. Cadence evidence：终止式检测（V-I, V-i, IV-I, bVII-I 等）
6. 融合：加权多数投票；加权不一致时降低置信度，输出 alternatives

输出包含 key_segments：对全曲做滑动窗口检测音级分布突变，在突变点切分，每段独立调性。

### 和声功能规则引擎（改进版）

使用 local_key（非全局 key）作为基准：

1. 基础映射：local_key + root + quality → 音级
2. 借用和弦：大小调双向借用（bIII, iv, bVI, bVII, V, IV, #vii°）
3. 副属和弦：检测属七 + 解决方向（需看后续 1-2 个和弦是否解决到目标）
4. 转调/离调：由 key_segments 驱动，非规则推断
5. Slash chord 功能标注（Phase 1.5）：根音功能 + 低音线标注
6. Fallback：chromatic roman numeral（如 bII, #iv°）
7. 长期方向：规则引擎 → seq2seq 数据驱动模型

---

## 模型加载策略

流水线中按需加载/卸载，一次最多驻留 2 个大模型：

1. 启动时：仅加载轻量依赖
2. 流水线执行时：
   - Demucs 加载 → 分离完成 → 卸载
   - MERT backbone + 3 head 加载 → Beat/Chord/Key 推理完成 → 卸载
   - WhisperX 加载 → 转写完成 → 卸载
3. Demucs + MERT 不可同时驻留（6GB VRAM 下），或可同时（12GB+ VRAM）

- Backbone (MERT-330M / MusicFM)：按需加载，推理后卸载
- 各 Head：轻量（Linear / TCN），随 backbone 一起加载/卸载
- ONNX/TensorRT 导出用于生产推理加速
- 微调：全模型或 LoRA，在多个标注数据集上训练

---

## 评测指标

必须建立评测集和指标闭环，否则无法判断改进是否提高准确率。

| 指标 | 说明 |
|------|------|
| Beat F-measure | 节拍检测精度（±70ms tolerance） |
| Downbeat F-measure | 强拍检测精度 |
| Key accuracy | 全局调性正确率 |
| Local key accuracy | 分段调性正确率 |
| Chord root accuracy | 根音正确率 |
| Chord quality accuracy | 和弦性质正确率 |
| Weighted Chord Symbol Recall (WCSR) | 加权和弦符号召回率 |
| Bass note accuracy | 低音正确率（Phase 1.5+） |
| Roman numeral accuracy | 功能标注正确率 |
| Segmentation F-measure | 和弦边界精度 |
| N detection recall | 无和弦段检测召回率 |

训练/评测数据集：Isophonics, McGill Billboard, RWC Popular, Robbie Williams, 自建中文流行标注集。

---

## 硬件要求

GPU 必需，CPU 推理无法满足 5 分钟超时限制。

| 项目 | 最低 | 推荐 |
|------|------|------|
| GPU VRAM | 8 GB | 12 GB+ |
| RAM | 16 GB | 32 GB |
| CPU | 4 核 | 8 核+ |
| 磁盘 | 30 GB SSD | 100 GB SSD |

### 单模型显存占用 (fp16)

| 模型 | 峰值显存 |
|------|----------|
| MERT-330M | ~660 MB |
| Demucs htdemucs | ~1.5-2 GB |
| WhisperX large-v3 + alignment | ~3-4 GB |
| 3× Head + beat-sync pooling | ~100 MB |

6GB VRAM 场景：禁止 Demucs + MERT 同时驻留；WhisperX 可用 medium 模型；提供 CPU fallback 选项。

### 推理耗时估算（12GB VRAM / 3 分钟歌曲）

| 阶段 | 耗时 | 并行 |
|------|------|------|
| Vocal Sep (Demucs) | 30-60s | 与 MERT 并行（12GB+） |
| Beat + Chord + Key (MERT + head) | 20-40s | |
| Lyrics (WhisperX) | 15-30s | |
| Harmony (规则引擎) | <1s | |
| **总计** | **~1-3 分钟** | |

---

## 非功能需求

- 支持的音频格式：MP3, WAV, M4A, FLAC, OGG
- 最大文件大小：50MB；最大时长：10 分钟
- 分析超时：单文件 ≤ 5 分钟
- 并发：单 worker 串行处理（MVP 阶段）
- 结果 TTL：1 小时后自动清理
- 任务可取消（长音频或异常大文件场景）
- 无需用户系统/登录/持久化存储

---

## Phase 规划

| Phase | 内容 |
|-------|------|
| Phase 1 (MVP) | 完整流水线，bass=null，全局 key，N 支持，alternatives，beat map |
| Phase 1.5 | Bass Head 实现，slash chord 输出，手动编辑，低音线显示 |
| Phase 2 | 多流输入 (full mix + accompaniment + bass-enhanced)，联合微调，key_segments 精度提升 |
| Phase 3 | 风格感知模型，seq2seq 功能和声，word-level 歌词对齐，音频播放同步 |
