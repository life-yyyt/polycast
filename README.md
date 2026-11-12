# polycast

多智能体预测系统：输出校准过的概率，并留下完整审计轨迹。

一组彼此独立的分析师检索证据、分别给出对某个可判定问题的概率判断，再经过一轮红队质疑，最后聚合成一个带区间的概率、分歧度指标，以及一份能回溯的完整决策记录。

默认使用离线确定性桩模型，不依赖任何 API key，克隆下来即可运行。

## 为什么这样设计

大多数「AI 预测」demo 止步于生成一个数字。而数字恰恰是最不重要的部分：真正决定它有没有用的是概率是否校准、事后能不能审计、以及模型自相矛盾时管线会不会优雅降级。这三件事才是本项目的主要交付物。

- **分析师互相独立。** 面板成员看不到彼此的答案。带共享上下文的串行辩论会让分歧迅速消失，产出的是看起来自信、实则过度自信的预测。
- **红队在聚合之前运行。** 质疑者面对的是具体的概率和面板分歧，而不是一个空泛的计划；它的调整量被限制在 ±0.3，单轮质疑无法左右结论。
- **校准是被测量出来的，不是被声称的。** Brier 分数、对数损失、可靠性分箱、以及相对基准率的 skill 值；Platt 重校准在独立划分上拟合。一个排序很好但严重过度自信的预测器，会表现为可靠性曲线明显偏离对角线。
- **全程留痕。** 每次运行写一份只追加的 JSONL 轨迹：检索到的文档及其得分、每个分析师的推理、红队的异议、以及聚合的算式。
- **区间反映分歧。** 区间宽度由面板分歧度和红队调整量决定，而不是一个固定常数。面板因为共同原因达成一致时，收窄区间的是红队的调整。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

polycast forecast \
  --text "Will the central bank cut the policy rate at the next scheduled meeting?" \
  --criteria "Resolved yes if the published decision confirms a reduction." \
  --resolve-by 2026-12-18

polycast serve --port 8000
polycast eval --dataset evals/questions.json
polycast simulate --agents 40 --truth 0.7
```

## 流水线

```
问题
  -> 检索（TF-IDF，按时间新鲜度加权）
  -> [base_rate | bull | bear]      相互独立，并发执行
  -> red_team                        针对面板分歧提出质疑
  -> 聚合                            置信度加权、去极值、应用调整
  -> 概率 + 区间 + 分歧度
  -> sqlite（预测记录）+ jsonl（审计轨迹）
```

## 校准

`polycast eval` 用已判定的问题给管线打分，在训练划分上拟合 Platt 缩放，并分别报告留出集在校准前后的表现。内置数据集是 2024–2025 年 22 个真实、公开可判定的央行决议事件。

```json
{
  "scaler": { "a": 2.4013, "b": 1.0007 },
  "test_raw":          { "n": 7, "brier": 0.2555, "log_loss": 0.7043, "base_rate_brier": 0.2041 },
  "test_recalibrated": { "n": 7, "brier": 0.2235, "log_loss": 0.6528, "base_rate_brier": 0.2041 }
}
```

这组数字值得仔细读，而不是扫一眼。离线桩模型是一个没有世界知识的词法模型，它对货币政策问题的原始预测不含任何信息，因此 Brier 分数落在基准率附近。这组数字真正说明的是重校准这条路径是通的：把概率向基准率收缩，使留出集的对数损失从 0.704 降到 0.653，Brier 从 0.256 降到 0.224。这是评测工具本身在正常工作。把 `POLYCAST_MODEL` 指向一个真实模型，同样的命令就能测量那个模型。

工具本身有两条刻意的设计，并且都有测试覆盖：

- 训练/测试划分使用固定随机种子打散。顺序切分在数据集恰好按结果排序时会产出病态的训练集，而在上面拟合出的重校准会以一种极其自信的姿态外推到测试集。
- 当训练划分只包含单一结果类别时，Platt 缩放拒绝拟合——在那种情况下它学到的截距是无界的。

## HTTP API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/forecast` | 执行一次预测 |
| `GET` | `/forecast/{id}` | 取回已存储的预测 |
| `GET` | `/forecasts` | 列出最近的预测 |
| `GET` | `/metrics` | 计数、平均分歧度、当前使用的 provider |
| `POST` | `/simulate` | 基于智能体的信念演化模拟 |

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Will unemployment claims decline for a fifth consecutive week?",
    "resolution_criteria": "Resolved yes if the published count is below the prior week.",
    "resolve_by": "2026-12-31T00:00:00Z",
    "reference_class": "weekly continuing claims series"
  }'
```

## 配置

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `POLYCAST_MODEL` | 未设置 | 模型 id；不设置则使用离线桩模型 |
| `POLYCAST_API_KEY` | 未设置 | 托管 provider 的 bearer token |
| `POLYCAST_BASE_URL` | `https://api.openai.com/v1` | 任意 OpenAI 兼容端点 |
| `POLYCAST_CORPUS` | `data/corpus.json` | 证据语料 |
| `POLYCAST_DB` | `polycast.db` | 预测记录的 sqlite 路径 |
| `POLYCAST_AUDIT_DIR` | `runs` | JSONL 审计轨迹目录 |

`POLYCAST_BASE_URL` 也可以指向 Ollama、vLLM 等本地运行时，整条管线可以完全跑在本机。

## 模拟

`polycast simulate` 运行一个基于智能体的信念形成模型：每个智能体持有带噪声的私有信号，向邻居均值更新，并从随机先验出发。它回答单次运行无法回答的问题——一个群体需要多少个独立分析者才会收敛，以及社会影响会把群体推向哪里。结果在相同种子下可复现。

## 项目结构

```
src/polycast/
  schemas.py      管线与 API 之间共享的类型契约
  providers.py    provider 协议、离线确定性桩模型、OpenAI 兼容客户端
  retrieval.py    带时间衰减的 TF-IDF 索引
  agents.py       分析师角色、提示词构造、聚合规则
  pipeline.py     编排：检索 -> 并行分发 -> 红队 -> 聚合 -> 落库
  calibration.py  Brier、对数损失、可靠性分箱、Platt 缩放
  evaluation.py   带训练/测试隔离的已判定问题评测
  simulation.py   基于智能体的信念模拟
  audit.py        JSONL 审计轨迹与 sqlite 存储
  server.py       FastAPI 应用
  cli.py          命令行入口
data/corpus.json  示例证据语料
evals/            已判定问题集
tests/            单元、管线、模拟与 API 测试
```

## 测试

```bash
pytest
```

测试覆盖：评分函数与手算结果对照、聚合规则（去极值、红队调整、区间边界）、管线的端到端执行（含审计轨迹）、API 契约、模拟的可复现性，以及重校准可能静默失效的两种方式。

## 开发

| 命令 | 作用 |
| --- | --- |
| `make install` | 以可编辑模式安装，含开发依赖 |
| `make test` | 运行 pytest 测试套件 |
| `make lint` | 对 `src` 和 `tests` 执行 ruff 检查 |
| `make eval` | 用已判定问题给管线打分 |
| `make serve` | 启动 API（带热重载） |
| `make clean` | 清理运行产物与缓存 |

## 局限

既然这个项目声称的比它实际做到的要多，那就明确写出来：

- 内置语料是十二篇合成文档。在真实语料上的检索质量在这里没有被验证。
- 聚合权重是手工设定的。正确的做法是在已判定的历史数据上学习每个角色的权重。
- 离线桩模型是没有世界知识的词法模型。它检验的是机制，不是预测能力，对内置的货币政策问题也没有任何预测力。
- 评测集是 22 个真实决策。它足以检验评测工具，但不足以测量一个预测器：这个样本量下 Brier 分数的标准误远大于被比较的差异。
- API 是单进程、在 SQLite 上同步执行的。对以读为主的内部使用足够；在扩容之前需要队列和真正的数据库。
- API 没有认证和限流。

## 许可

MIT
