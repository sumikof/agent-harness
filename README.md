# agent-harness

数時間〜数日規模の長時間コーディングタスクを安定して継続実行するためのエージェントハーネス。

単一のLLMセッションを長時間維持するのではなく、**役割を持つ複数のエージェントをFresh Sessionとして起動し、Git・SQLite・構造化Artifactを介して状態を引き継ぐ**。長時間実行は「長いSession」ではなく「短いSessionの連続」として実現する。

v2からは **DGX Spark 1台 + Qwen3.6-27B-FP8 (vLLM)** を主要ターゲットとし、複数Taskの**並列実行**によりマシン全体のaggregate throughputを最大化する。単一Agentの最速化ではなく、「単位時間あたりに正しく完了できるCoding Task数」が最適化対象である。

## アーキテクチャ

```text
                         DGX Spark
                            │
                            ▼
             ┌──────────────────────────┐
             │ Qwen3.6-27B-FP8 / vLLM   │  FP8 / Speculative Decoding (MTP)
             │ OpenAI-compatible API    │  Prefix Caching / Continuous Batching
             │ max ~16 active requests  │  Chunked Prefill / text-only serving
             └────────────┬─────────────┘
                          │
              LLM Request Gate (16 slots,
              prefix-affinity + starvation防止)
                          │
              ┌───────────┴───────────┐
              │  Parallel Scheduler   │ ← 決定論的・dependency-aware(LLMは使わない)
              │  Task DAG (Planner産) │
              └───────────┬───────────┘
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
      Task A            Task B            Task C     … 最大 max_parallel_tasks
    Worktree A        Worktree B        Worktree C   (Git worktree隔離)
      Analyst           Analyst           Analyst
        ↓                 ↓                 ↓        各RoleはFresh Session、
     Developer         Developer         Developer   同一Task Attempt内では
        ↓                 ↓                 ↓        同じWorktreeを共有
      Tester            Tester            Tester
        ↓                 ↓                 ↓
   Verification      Verification      Verification  (heavy_test pool)
        ↓                 ↓                 ↓
     Reviewer          Reviewer          Reviewer    (独立・read-only)
        │                 │                 │
        └────────┬────────┴────────┬────────┘
                 ▼                 ▼
           Task Branch Commit(task_commit; Harnessのみがcommit)
                 │
                 ▼
         Serialized Integrator(必ず1件ずつmerge --no-ff)
                 │
                 ▼
        Integration Branch(integration_commit = 正式Checkpoint)

Persistent State: SQLite(single writer) + Append-only Event Ledger
                + Git + Task Worktrees + Structured Artifacts
```

### 設計原則

1. **Orchestratorは決定論的State Machine** — 次に何を実行するかをLLMに判断させない。Task dependencyはHarnessが管理する
2. **Sessionを長期記憶として使わない** — 状態の正本は SQLite + Git + Artifacts
3. **役割ごとにFresh Context、Agent間は構造化Artifactで接続** — Conversation・hidden reasoningを引き継がない
4. **並列化はWorktree隔離の上でのみ** — Taskごとに独立branch+worktree。同一worktreeへのmutating Agentは常に1
5. **IntegrationはSerialized** — Integration branchへのmergeは必ず1件ずつ。ConflictはFresh Repair Attempt(新しいbaseで再実装)へ流れる正常系
6. **Deterministic Verification + 独立Reviewer** — Agentの自己申告を完了根拠にしない
7. **Git lifecycleはHarness専有** — Agentのgit commit/merge/push/reset/worktreeはHookで拒否
8. **SQLiteはSingle Writer** — 全writeはprocess-wideのwriter直列化を通る(WAL / busy_timeout / stream単位のcontiguous event seq)
9. **Retryは有限・3層** — Provider Retry(transient・attempt消費なし。429/overloadはbackpressure)/ Episode Recovery / Reasoning Retry
10. **SecurityはPromptに依存しない** — Tool Permission / コマンド検査 / 書込パス検査をHarness側で強制
11. **Prefix CacheはFresh Sessionと両立** — Session再利用ではなく「Prompt prefixの再利用」。Context はstable→volatileの順に構成し、PrefixGroupKeyでdispatchをグルーピングする

### 推論基盤(標準Provider)

標準Providerは **ローカルvLLM (OpenAI互換API)** 上の `Qwen/Qwen3.6-27B-FP8`。
デプロイ・チューニング・検証は [`deploy/inference/`](deploy/inference/README.md) を参照。

- FP8 + MTP speculative decoding + prefix caching + chunked prefill を標準有効化(4bitを標準にしない)
- Process共有のHTTP client + LLM Request Gate(16並列、prefix-affinity dispatch)
- Role別に固定順のTool Schema(prefix cacheを壊さない)、Role別Sampling Profile
- `agent-harness health` / 起動時検証: tool calling・structured output・prefix cache を**実測**で確認してからTaskを開始する
- Provider抽象は維持: `provider.type: claude` や role別override で他エンジンへ切替可能

## セットアップ

```bash
# 1. 推論基盤 (DGX Spark)
cd deploy/inference
cp .env.example .env        # VLLM_IMAGE を検証済みtagに固定(latest禁止)
./setup_dgx_spark.sh        # ARM64 / GPU / Docker / vLLM>=0.19 検証
./start.sh && ./healthcheck.sh
./benchmark.sh all          # 実測 → config/inference-tuning.json を固定

# 2. Harness
pip install -e ".[dev]"     # ローカルProviderは追加SDK不要 (httpx同梱)
cp config.example.yaml config.yaml
agent-harness init
# workspace/repository/ に対象リポジトリを配置
agent-harness health        # 推論エンドポイント検証
agent-harness run           # 実行(中断後に再実行すれば自動Recovery)
agent-harness status
agent-harness events --limit 100
```

```text
workspace/
├── harness.db          # SQLite(single writer / WAL)
├── repository/         # Integration branch checkout(正式Checkpoint)
├── worktrees/          # Task並列実行用worktree(T001-A1/ など、Harnessが管理)
├── artifacts/          # Agent間Blackboard + ContextManifest + diagnostics
└── logs/
```

## 並列実行モデル

- `parallelism.max_parallel_tasks`(default 16)個までのREADY Taskを同時実行。READY判定はTask DAG(依存が全てCOMPLETED/SKIPPED)による
- Resource Pool分離: `llm: 16` / `heavy_build: 2` / `heavy_test: 2` / `git_integration: 1`。Agentのビルド・テスト実行中もGPU推論slotは他Agentが使う
- LLM Gateのdispatch順: ①dependency correctness ②starvation防止(aging) ③prefix affinity ④FIFO
- vLLM過負荷(429等)はProvider内retry(同一session・履歴保持・attempt消費なし)で吸収するbackpressure
- Integration Conflictの正常系フロー: `Task branch PASS → merge失敗 → abort → INTEGRATION_CONFLICT → Fresh Repair Attempt(現Integration HEADをbase、元diffをArtifactで提示) → 再実装`

## 耐障害性レイヤー

- **Event Ledger** — append-only。`(stream_type, stream_id, seq)`単位でgap/duplicateなし。並列Task下でもsingle writerがserializeする
- **Operation Intent/Result** — Agent dispatch / Verification / Git commit / **Git integration** を「Intent永続化→副作用→Result永続化」でjournal。Commit/Mergeには `Harness-Task` / `Harness-Operation-Id` trailerが付き、crash後はtrailerでreconcile(二重merge・二重commitなし)
- **ContextManifest / ResolvedAgentRunSpec / AgentCapabilities** — dispatch前に入力・実行内容を確定・永続化。Capability不足のProviderは実行前に拒否
- **Invariant Checker** — `RUNNING AgentRun ≤ max_parallel_agent_runs`、`同一Attemptのmutating Agent ≤ 1`(DB unique indexでも強制)、worktree不変条件(存在・非共有・期待branch・base commit解決可能)等を起動時検証
- **Large Output Spill / Repeat Action Guard / Budget 3階層** — 従来どおり(ローカルProviderにも適用)

## 中断・復旧(並列対応)

起動時Recoveryは **RUNNING中の全Task Attemptを列挙し、worktree単位で** 復旧する:

1. SQLite integrity → Event Ledger seq検証(破損は修復せずBLOCKED)
2. GIT_INTEGRATION intentのreconcile — trailerでmerge済みを検出しDBを追いつかせる/未完了merge (`MERGE_HEAD`) はabort。二重mergeしない
3. GIT_COMMIT intentのreconcile — task branch上のcommitを検出(`git log --all`)。task_commit記録+Attempt PASSED(完了はintegration経由のみ)
4. 各RUNNING Attemptのworktree: dirty diffを全量アーカイブ → worktree+branch削除 → Attempt INTERRUPTED → Task READYへ。**他TaskのWorktreeには触れない**。orphan worktreeも同様に退避・削除
5. Main checkout(integration branch)のdirty treeは従来のevidence-hash規則で処理。説明不能なdirtyは起動拒否

## Benchmark

`scripts/benchmark_inference.py`(または `deploy/inference/benchmark.sh`)で実測する。

- Primary score: **concurrency 16でのaggregate useful output tokens/sec**(単一stream速度やAcceptance Rate単独では判断しない)
- Scenario: concurrency 1/4/8/16(+24/32)、shared-prefix 20K/50K、MTP OFF/1/2/3/4、max-num-seqs 8/16/24/32、max-num-batched-tokens 16K/32K/64K、gpu-memory-utilization 0.75〜0.90、soak(8〜16並列連続負荷)
- 結果は `config/inference-tuning.json` にknown-good profileとして固定(起動時auto-tune禁止)。`--label baseline`(speculative OFF / cache OFF / concurrency 1)をreference保存し、悪化時は警告

## 開発

```bash
python -m pytest -q     # 188 tests。LLM不要。FakeRunnerで並列オーケストレーション全体を検証
```

主要モジュール:

- `harness/orchestrator/` — Project/Taskループ、**ParallelTaskScheduler**、**ResourcePools/PrefixAffinityGate**、State Machine、Recovery、Invariants、Budget
- `harness/agents/` — Provider抽象、**openai_compat**(ローカルvLLM)、**local_tools**(固定順Tool Schema+実行)、**health**(endpoint検証)
- `harness/context/` — Context Builder(stable→volatile順・canonical JSON・token budget)、**prefix**(PrefixGroupKey)
- `harness/database/` — SQLite(single-writer)リポジトリ層
- `harness/git/` — Git操作、Checkpoint、**worktree**、**integration**(serialized merge)
- `harness/verification/` / `harness/security/` / `harness/prompts/` — 従来どおり
- `deploy/inference/` — DGX Spark vLLMスタック、`scripts/benchmark_inference.py` — 実測ハーネス

詳細な導入・チューニング手順と最終構成レポート: [`docs/dgx-spark-deployment.md`](docs/dgx-spark-deployment.md)
