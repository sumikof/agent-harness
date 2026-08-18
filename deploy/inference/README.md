# DGX Spark 推論スタック (vLLM + Qwen3.6-27B-FP8)

Agent Harness の標準推論基盤。DGX Spark 1台の **aggregate throughput 最大化**
(単一Agentの最速化ではない)を目的に、FP8 + Speculative Decoding (MTP) +
Prefix Caching + Continuous Batching + Chunked Prefill を有効化した
OpenAI互換エンドポイントを提供する。

## ファイル

| ファイル | 役割 |
|---|---|
| `.env.example` | 全serving設定(コピーして `.env` に) |
| `compose.yaml` | Docker Compose定義(GPU / HFキャッシュ永続化 / healthcheck) |
| `setup_dgx_spark.sh` | 環境検証: `uname -m` / `nvidia-smi` / `nvcc` / `docker` / GPUアクセス / container内 `vllm --version` |
| `start.sh` | tag固定検証 → vLLM>=0.19検証 → `vllm serve` コマンド生成 → 起動 |
| `stop.sh` | 停止(モデルキャッシュはホストに残る) |
| `healthcheck.sh` | tool calling / reasoning parser / prefix cache / speculative decode の**実測**検証 |
| `benchmark.sh` | `scripts/benchmark_inference.py` のラッパ |

## セットアップ手順

```bash
cd deploy/inference
cp .env.example .env          # VLLM_IMAGE を実際に検証したtagへ固定する
./setup_dgx_spark.sh          # 環境検証(ARM64 / GPU / Docker / vLLM version)
./start.sh                    # 起動(初回はmodel downloadで時間がかかる)
./healthcheck.sh              # 機能検証(flagではなくmetrics/応答で確認)
./benchmark.sh all            # 実測 → config/inference-tuning.json を生成
```

## 設計上の固定事項

- **`latest` タグ禁止** — `start.sh` が拒否する。起動確認済みのtagを `.env` に固定する。
- **vLLM >= 0.19** — container内 `vllm --version` を起動前に検証。満たさない場合は
  Qwen3.6を正式サポートするより新しいNGC containerを使う。
- **FP8が標準** — 4bit量子化を標準にしない。
- **Text-only serving** (`--language-model-only`) — Vision encoderのメモリを
  KV cache / batchingへ回す。multimodalは別profile。
- **Prefix Caching必須** (`--enable-prefix-caching` 明示) — 有効かどうかは
  healthcheck / benchmark が `/metrics` の prefix_cache 系カウンタで実測検証する。
- **Speculative Decoding** — 現時点のproduction baselineはQwen3.6自身のMTP
  (`{"method":"qwen3_next_mtp","num_speculative_tokens":2}`)。
  `num_speculative_tokens` は絶対値ではなく、OFF/1/2/3/4のbenchmark対象。
  評価はAcceptance Rateではなく **aggregate output tokens/sec** を主指標とする。
- **DSpark拡張点** — `.env` の `SPECULATIVE_CONFIG` を差し替えるだけでDSpark
  speculatorへ移行できる。ただし **Qwen3.6-27Bとの互換性が確認されたmodelのみ**。
- **`gpu-memory-utilization` baseline 0.80** — DGX SparkはHarness / Git /
  Compiler / Tests / SQLite / OSと同居する。0.95へ盲目的に上げない。
  0.75 / 0.80 / 0.85 / 0.90 を負荷試験して長時間安定する最大値を採用。
- **KV cache dtype = auto** — Weight FP8とKV cache量子化を混同しない。
  fp8 KVは16並列+必要contextがメモリに収まらない場合のみ別profileでbenchmark。
- **Context profile** — production は `MAX_MODEL_LEN=65536`(performance)。
  long=131072 / maximum=262144 は必要なタスクのみ。
- **モデルキャッシュ永続化** — `~/.cache/huggingface` をmount。container再作成で
  30GB再downloadが発生しない。

## チューニングの確定

benchmark結果から `config/inference-tuning.json` を生成し、**known-good profile
として固定**する。production起動時に毎回auto-tuneしない。サーバ再起動が要る
sweep(MTP段数 / max-num-seqs / max-num-batched-tokens / gpu-memory-utilization)は
`.env` を書き換えて `./start.sh` → `./benchmark.sh sweep --label <point>` を繰り返す。

Async schedulingは、使用するvLLM versionでspeculative decodingとの併用が正式
サポートされる場合のみbenchmark対象に加える。優先順位は
Speculative Decoding > Prefix Caching > Continuous Batching > Chunked Prefill >
その他scheduler最適化であり、Speculative Decodingを無効化してまでasync
schedulingを優先しない。
