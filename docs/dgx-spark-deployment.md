# DGX Spark + Qwen3.6-27B-FP8 高並列構成 — 最終レポート

本書は改修後の最終構成を、実機投入時に確定・記録すべき項目とあわせてまとめる。
ベンチマーク数値はハードウェア実測でのみ確定する — 本リポジトリの
`deploy/inference/benchmark.sh` を DGX Spark 上で実行し、本書の
「Benchmark記録欄」と `config/inference-tuning.json` を実測値で埋めてから
production 投入すること。**測定していない数値をこの文書に書いてはならない。**

---

## 1. Environment(実機で記録する)

| 項目 | 確認コマンド | 記録欄 |
|---|---|---|
| DGX OS | `cat /etc/os-release` | _実機で記入_ |
| Architecture | `uname -m`(aarch64 であること) | _実機で記入_ |
| Driver | `nvidia-smi` | _実機で記入_ |
| CUDA | `nvcc --version` | _実機で記入_ |
| Docker | `docker --version` / `docker info` | _実機で記入_ |
| Container GPU access | `docker run --rm --gpus all ubuntu nvidia-smi` | _実機で記入_ |
| Container image | `.env` の `VLLM_IMAGE`(検証済みtag固定・latest禁止) | _実機で記入_ |
| vLLM version | `docker run --rm --entrypoint vllm $VLLM_IMAGE --version`(>= 0.19) | _実機で記入_ |
| Model revision | `.env` の `MODEL_REVISION`(固定推奨) | _実機で記入_ |

上記は `deploy/inference/setup_dgx_spark.sh` が一括検証する。ARM64ホストに
x86イメージ/wheelを導入しないことも同スクリプトが確認する。

## 2. Final vLLM command

`deploy/inference/start.sh` が `.env` から生成する(生成物は
`deploy/inference/.generated/launch.sh` に残る)。baseline構成での完全なコマンド:

```bash
vllm serve Qwen/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b-fp8 \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching --enable-chunked-prefill \
  --max-model-len 65536 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-dtype auto \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

これは **KNOWN-GOOD BASELINE** であり最終値ではない。最終値はSweep実測
(§7)で決定し `config/inference-tuning.json` に固定する。使用するvLLM
versionでオプション名が異なる場合は `.env` の `EXTRA_ARGS` で調整する。
hybrid-model向け prefix-match-unit 相当のオプションがサポートされる場合は
default / 32 / 64 / 128 をbenchmark対象に加える(unsupported optionを無理に
使わない)。

ホスト公開は compose 側で `127.0.0.1:8000` にバインドされる(harnessは
`http://127.0.0.1:8000/v1`、model alias `qwen3.6-27b-fp8` を使用)。

## 3. Final Harness concurrency

`config.yaml`(スキーマは `config.example.yaml`):

```yaml
parallelism:
  max_parallel_tasks: 16          # 同時実行Task数(DAGのREADYから充填)
  resource_pools:
    llm: 16                        # in-flight LLM request上限 = max-num-seqs
    heavy_build: 2
    heavy_test: 2
    git_integration: 1             # 常に1(コードでも強制)
inference:
  concurrency:
    max_requests: 16               # process-wide LLM Gate
```

- LLM slotとビルド/テストslotは独立(Agentがツール実行中でも他AgentがGPUを使う)
- `RUNNING AgentRun ≤ max_parallel_agent_runs(=max_parallel_tasks)` はDB検証+invariant
- `同一Task Attemptのmutating Agent ≤ 1` はSQLite partial unique indexで強制

## 4. Git strategy

| 項目 | 構成 |
|---|---|
| Task worktree | `workspace/worktrees/<KEY>-A<cycle>`、Task Attemptサイクルごとに独立 |
| Task branch | `harness/task/<KEY>/attempt-<cycle>`(現Integration HEADから分岐) |
| Task commit | Reviewer PASS後にHarnessがtask branch上でcommit(`Harness-Task` / `Harness-Operation-Id` trailer付) |
| Integration | IntegrationManagerが `merge --no-ff` を**必ず1件ずつ**実行。`integration_commit` が正式Checkpoint(SQLiteに `task_commit` と分離保存) |
| Conflict recovery | merge失敗 → `merge --abort` → Task=INTEGRATION_CONFLICT → 元diffをArtifact化しworktree破棄 → **現Integration HEADをbaseにFresh Repair Attempt**(`max_integration_repairs` 回まで、超過でBLOCKED) |
| Crash recovery | GIT_INTEGRATION intentをtrailerでreconcile(二重merge禁止)、RUNNING Attemptはworktree単位でarchive→削除→READY復帰。他Taskのworktreeに触れない |

AgentはすべてのRoleで `git commit/merge/rebase/push/reset/worktree` を
Hook/コマンド検査で拒否される(ローカルProviderのtool loopでも同一ポリシー)。

## 5. SQLite strategy

- **PostgreSQLへは移行しない。** SQLite + WAL + `foreign_keys=ON` + `busy_timeout`(configurable, default 5000ms)
- **Single Writer**: 全statement/transactionはprocess-wideのwriter lockで直列化
  (`harness/database/connection.py`)。並列Agent coroutine・worker thread
  (verification等)からのwriteが競合しない
- **Transaction boundary**: 「状態UPDATE + ledger event INSERT」は常に1
  transaction。dispatch前のRUNNINGチェック+run row作成は `BEGIN IMMEDIATE`
- **Event sequence**: `(stream_type, stream_id, seq)` per-streamでcontiguous。
  8 thread × 8 stream並列emitのテストでgap/duplicateゼロを検証済み

## 6. Prefix Cache

**Prompt構造**(stable → volatile; `harness/context/builder.py`):

```
(system prompt: 安定したglobal指示 + Role profile)
1. harness_rules       デプロイ全体で不変
2. project_context     プロジェクト単位で不変(worktreeパス等を含まない)
--------------------- ここまでが共有prefix ---------------------
3. task_context        Task固有
4. attempt_context     Attempt固有(失敗・feedback・diff)
5. extra_context       呼び出し固有
6. volatile_metadata   working dir / attempt_no / branch(必ず末尾)
7. assignment          Role固定の最終指示
```

- timestamp / UUID / run id / attempt id / diff / worktreeパスをstable prefixに
  置かない(テストで回帰防止)
- Prompt内JSONはcanonical serialization(sorted keys・固定separator)
- **PrefixGroupKey** = SHA256(model + agent_profile_hash + project_context_hash
  + tool_schema_hash + common_prompt_hash)。AgentRun行に記録され、LLM Gateが
  dispatch順の第3基準(①dependency ②starvation防止 ③affinity ④FIFO)に使う
- Role別Tool Schemaは固定順・固定serialization(`local_tools.py`)
- **検証**: `agent-harness health` / `deploy/inference/healthcheck.sh` が同一
  long-prefixリクエストを2回送り、vLLM `/metrics` の prefix_cache 系カウンタ
  (名称はversion差異があるためsubstringで発見)が前進することを確認する。
  optionが付いているだけではPASSにしない
- Fresh Sessionは維持する — 高速化はSession再利用ではなくPrompt prefix再利用で行う

Context長は最大値でなくthroughput最適値: production profileは
`max_model_len=65536` / input budget 48〜52K tokens(既定50K、超過分は
dynamic sectionのみ切詰め+Spill/Artifact参照)。long=131072 / maximum=262144
は必要Taskのみ。

## 7. Benchmark(実機で実行して記録する)

実行手順:

```bash
cd deploy/inference
# reference baseline(speculative OFF / prefix cache OFF / concurrency 1)
#   .env: SPECULATIVE_CONFIG= 空 + EXTRA_ARGS=--no-enable-prefix-caching 相当に編集
./start.sh && ./benchmark.sh sweep --label baseline
# baseline構成に戻して本測定
./start.sh && ./benchmark.sh all --label mtp2
# MTPスイープ(server再起動が必要): num_speculative_tokens を 1/3/4 に変え
./benchmark.sh sweep --label mtp1   # 等
# max-num-seqs 8/16/24/32、max-num-batched-tokens 16384/32768/65536、
# gpu-memory-utilization 0.75/0.80/0.85/0.90 も同様にlabel付きで実測
./benchmark.sh soak --minutes 60    # 長時間soak(OOM/leak/hang/エラー率)
```

**Primary score = concurrency 16 の aggregate useful output tokens/sec。**
Secondary: TTFT p95 / per-agent tok/s / error rate / OOM有無 / prefix cache hit /
verification throughput。Acceptance Rateだけでspeculative設定を決めない —
Acceptanceが下がってもaggregateが上がるならそちらを採用する。

記録欄(`config/benchmark-results.json` から転記):

| Scenario | aggregate tok/s | per-req tok/s | TTFT p50 | TTFT p95 | latency p50/p95 | prefix cache | speculative |
|---|---|---|---|---|---|---|---|
| concurrency 1 | _実測_ | _実測_ | _実測_ | _実測_ | _実測_ | _実測_ | _実測_ |
| concurrency 4 | _実測_ | | | | | | |
| concurrency 8 | _実測_ | | | | | | |
| concurrency 16 | _実測_ | | | | | | |
| (24 / 32 saturation) | _実測_ | | | | | | |

| Speculative | aggregate tok/s @16 | single tok/s | accepted/forward | TTFT p95 | 備考 |
|---|---|---|---|---|---|
| OFF | _実測_ | | — | | |
| MTP=1 | _実測_ | | | | |
| MTP=2 (baseline) | _実測_ | | | | |
| MTP=3 | _実測_ | | | | |
| MTP=4 | _実測_ | | | | |

Sweep確定後、`scripts/benchmark_inference.py` が `config/inference-tuning.json`
にknown-good profileを書き出す。production起動はこのprofileを固定使用し、
**起動時auto-tuneは行わない**。optimized構成がbaselineより悪化した場合は
benchmarkが警告する(絶対tok/sはCIのPASS条件にしない)。

Async schedulingは、使用vLLM versionがspeculative decodingとの併用を正式
サポートする場合のみ追加benchmarkする。優先順位: Speculative Decoding >
Prefix Caching > Continuous Batching > Chunked Prefill > その他。

非Streaming(`inference.streaming: false`)がharness既定。`--no-stream` での
benchmark比較でaggregateが改善することを確認済みの場合のみstreamingへ戻す
理由はない(内部AgentはToken逐次表示不要)。

## 8. DSpark 拡張点

`.env` の `SPECULATIVE_CONFIG` を差し替えるだけでmethod/draft model/段数を
変更できる(harness側コード変更不要):

```
現在:  {"method":"qwen3_next_mtp","num_speculative_tokens":2}
将来:  {"method":"dspark","model":"/models/qwen3.6-27b-dspark","num_speculative_tokens":7}
```

**Qwen3.6-27Bとの互換が確認されたDSpark speculatorのみ使用可**。現時点の
production baselineはMTP。

## 9. Final recommendation(構成根拠)

採用構成(実測で上書きされるまでのbaseline):
`FP8 / MTP(k=2) / prefix caching / chunked prefill / max_model_len 65536 /
max_num_seqs 16 / max_num_batched_tokens 32768 / gpu_mem 0.80 / KV auto /
text-only serving / 非streaming / harness並列 16 task・16 LLM slot・build 2・
test 2・integration 1`

この構成がDGX Spark上の長時間・高並列Coding Agent workloadに適する理由:

1. **本workloadはlatencyではなくthroughputが価値** — Harnessは人間非対面の
   多Task並列であり、単一応答の速さよりマシン全体の正答Task完了数/時が
   成果に直結する。continuous batching + 16並列がGPUを飽和させる
2. **FP8はBlackwellのネイティブ精度・速度スイートスポット** — 4bitは
   品質リスク(コード生成の正確性低下→Verification/Reviewer差戻し増→
   実効throughput低下)に対して利得が不明なため標準にしない
3. **Agent promptは構造的にprefix共有率が高い** — 同一Role・同一Projectの
   session間で数千〜数万tokenの先頭が一致するようContextを設計したため、
   prefix cachingがprefill費用を大幅に削る。TTFT短縮はspeculative比で
   副作用ゼロ
4. **MTPはbatch照会あたりの実効token数を引き上げる** — decode-bound な
   長い生成(コード)で有効。段数はaggregate実測で決める(高concurrency では
   検証コストがbatch容量を食うため小さめのkが勝ちやすい — だからk=2が
   baseline、絶対値ではない)
5. **正しさと復旧性は並列化と直交に担保** — worktree隔離+serialized
   integration+single-writer SQLite+per-stream event seq+intent/result
   journalにより、「速くなったが壊れる」経路(repo破損・変更競合・二重merge・
   復旧不能・Reviewer汚染)を構造的に排除している。最終評価は
   `aggregate throughput × task success rate ÷ (recovery+conflict+retry overhead)`
   で行い、単一指標に最適化しない
