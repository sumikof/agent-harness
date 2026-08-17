# agent-harness

数時間〜数日規模の長時間コーディングタスクを安定して継続実行するためのエージェントハーネス。

単一のLLMセッションを長時間維持するのではなく、**役割を持つ複数のエージェントを逐次的に起動し、Git・SQLite・構造化Artifactを介して状態を引き継ぐ**。長時間実行は「長いSession」ではなく「短いSessionの連続」として実現する。

## アーキテクチャ

```text
                 Persistent State
              SQLite + Git + Artifacts
                       │
                       ▼
                ┌─────────────┐
                │ Orchestrator│   ← 決定論的 State Machine(LLMは使わない)
                └──────┬──────┘
                       ▼
                    Planner
                       ▼
                   Task Queue
                       ▼
        ┌──────── Task Loop ────────┐
        │  Analyst                  │
        │     ↓                     │
        │  Developer ◀──────────┐   │
        │     ↓                 │   │
        │  Test Engineer        │   │
        │     ↓                 │   │
        │  Deterministic Verify │   │
        │     ↓                 │   │
        │  Reviewer             │   │
        │   │    │              │   │
        │  PASS REPAIR ─────────┘   │
        │   ↓                       │
        │  Harness Commit           │
        │   ↓                       │
        │  Next Task                │
        │                           │
        │  繰り返し失敗 → Diagnostician │
        │   → Retry / Split / Replan│
        └───────────────────────────┘
```

### 設計原則

1. **Sessionを長期記憶として使わない** — 状態の正本は SQLite + Git + Artifacts
2. **役割ごとにFresh Context** — 思考の偏りを次の役割へ引き継がない
3. **Agent間は構造化Artifactで接続** — Conversationを直接継承しない
4. **Orchestratorは決定論的** — 次に誰を実行するかはPythonのState Machineが決める
5. **Agent自身に完了判定を任せない** — Harnessが exit code を確認する Deterministic Verification + 独立Reviewer
6. **Git CommitはHarnessのみが実施** — Task単位の正式Checkpoint
7. **Retry回数制限** — 一定回数失敗で Diagnostician → RETRY / SPLIT / REPLAN / BLOCKED
8. **逐次実行** — 並列化よりも安定性・再現性・障害解析容易性を優先
9. **Tool Permission / Hookで役割を強制** — Promptだけに依存しない

### 6つのエージェントロール

| ロール | 責務 | 書き込み権限 |
|---|---|---|
| Project Planner | 要求分析・Task分割・依存関係整理 | なし(Read Only) |
| Task Analyst | 1 Taskの詳細調査 → Task Brief生成 | なし |
| Developer | Task Briefに基づく実装 | ソースコード全体 |
| Test Engineer | テスト観点の独立確認・テスト追加 | テストファイルのみ |
| Reviewer | 独立レビュー → PASS / REPAIR / REPLAN | なし |
| Diagnostician | 繰り返し失敗の原因分析 → RETRY / SPLIT / REPLAN / BLOCKED | なし |

全ロールで `git commit/push/merge/rebase/reset`、sudo、credential操作等は PreToolUse Hook が拒否する(`harness/security/`)。加えてDeveloper以外のロールは書き込み系シェルコマンド(リダイレクト、`sed -i`、`git apply`、`python -c` 等)も拒否される。

なお、このシェルコマンド検査は defense-in-depth であり完全なサンドボックスではない(テストを実行できるシェルは最終的に任意コードを実行できる)。読み取り専用ロールの完全な強制が必要な場合は、コンテナ等のOSレベル隔離の中でHarnessを実行すること。

## セットアップ

```bash
pip install -e ".[claude,dev]"     # claude = Claude Agent SDK アダプタ
cp config.example.yaml config.yaml
# config.yaml を編集(project.goal, verification.language 等)

agent-harness init                  # workspace/ と harness.db を作成
# workspace/repository/ に対象リポジトリを clone または配置

agent-harness run                   # 実行(中断後に再実行すれば自動Recovery)
agent-harness status                # Project / Task の状態表示
agent-harness events --limit 100   # イベントログ表示
```

`workspace/` はディレクトリ単位で移動・バックアップ可能:

```text
workspace/
├── harness.db          # 状態管理(WAL / foreign_keys / busy_timeout)
├── repository/         # 対象リポジトリ(Gitが正式Checkpoint)
├── artifacts/          # Agent間のBlackboard
│   ├── project-plan.json
│   ├── tasks/T001/{task-brief,implementation,test-result,review,verification-result}.json
│   └── diagnostics/
└── logs/
```

## 耐障害性レイヤー

状態の正本(SQLite + Git + Artifacts)の上に、durable executionのための層を持つ:

- **Event Ledger** — `events` はappend-onlyのstream台帳。`(stream_type, stream_id, seq)` で連続番号が保証され、State変更(`UPDATE tasks` 等)と対応するEventのINSERTは同一トランザクションでatomicに行われる。
- **Operation Intent/Result** — 外部副作用(Agent dispatch / Verification / Git commit)は「Intentを永続化 → 副作用実行 → Resultを永続化」の順で `operations` にjournalされる。HarnessのGit Commitには `Harness-Task` / `Harness-Operation-Id` trailerが付く。
- **ContextManifest** — Agent起動前に、そのAgentへ渡す全Contextセクションをartifactとして固定・ハッシュ化する。Manifestの保存に失敗したAgentは起動しない。
- **ResolvedAgentRunSpec** — provider / model / profile hash / allowed tools / timeout 等、実際に実行される内容をdispatch前に確定しdurableに保存する。
- **AgentCapabilities / AgentProfile** — Providerの能力差を明示し、必要Capabilityを満たさないProviderは実行前に設定エラー(BLOCKED/FAILED)として拒否する。各AgentRunはprofile hash/versionを記録する。
- **Retry 3層** — Provider Retry(transientのみ・有限・指数backoff・attempt消費なし)/ Episode Recovery(中断復旧、原則Fresh Session)/ Reasoning Retry(Verification FAIL・REPAIR、常に新AgentRun + attempt増加)。Permanent failure(認証・設定エラー)はretryしない。
- **Invariant Checker** — 起動時に「COMPLETEDにはVerification PASS / Reviewer PASS / 解決可能なCommitがある」「RUNNING AgentRunは最大1件かつManifest/Specを持つ」「stream seqにgapがない」等を検証し、違反は黙って修復せずBLOCKEDにする。
- **Large Output Spill** — 巨大なdiff・ログはartifactへ全量保存し、Agent Contextにはhead/tail preview + locator(path / sha256 / bytes)のみ渡す。
- **Repeat Action Guard** — 同一Tool Callの連続繰り返しを検出し、警告→拒否+LOOP_DETECTEDをOrchestratorへ報告する(pre-tool hookを持つProviderのみ)。LOOP検出されたRunは成功出力があってもFAILED扱いとなり、Provider retryではなくreasoning retry(Fresh Attempt)経路へ送られる。

## 中断・復旧

Harnessプロセスは途中終了を前提とする。起動時は次の順で照合する:

1. SQLite integrity check → Event Ledgerのseq連続性検証(破損は修復せずBLOCKED)
2. 未完了Operationの検出。GIT_COMMIT Intentは `Harness-Operation-Id` trailerでGit実状態とreconcileする — Commit済みならDB側を追いつかせ(二重Commitしない)、未実行ならFAILEDとして通常のretryへ
3. RUNNINGのままのAttemptがあれば: Dirty Diffを `artifacts/diagnostics/` へ保存 → `git reset --hard HEAD` → AttemptをINTERRUPTED記録 → TaskをREADYへ戻しFresh Sessionで再実行
4. RUNNINGのままのAgentRunをINTERRUPTEDとして閉じ、Invariant検証後にProject Loopを再開

RUNNINGのAttemptで説明できないdirty treeはユーザの作業とみなし、破壊せず起動を拒否する。

**Recoveryのreset原則**: recoveryが `reset --hard` してよいのは、「そのdiffの正確なsha256をdurableに記録済みの状態」だけである。副作用系Operation(dispatch / verification / commit)はintent作成時にbase diff hashを、検証はコマンド完了毎に更新hashを、recovery自身はreset直前にsettlement hashを記録し、現在のdiffがいずれかと一致する場合のみresetする。

既知の受容制限: 「副作用がtreeを変更した直後〜そのhash永続化前」のcrash windowは、記録が副作用の後にしか行えない以上、原理的に閉じられない。このwindowに落ちた場合(かつRUNNING Attemptが無い場合)の帰結はデータ破壊ではなく**fail-safeな起動拒否**であり、operatorがtreeを確認・清掃してから再実行する。

## Budget管理

Project / Task / Agent Run の3階層でUSD予算を管理し、加えて
`max_attempts` / `max_turns` / `max_agent_runs_per_task` / `max_execution_seconds`
を制限する。超過時はTaskをBLOCKED、ProjectをPAUSEDへ遷移させる(クラッシュしない)。
PAUSEDのProjectは予算を引き上げて `run` し直せば継続する。

## Provider抽象化

Claude Agent SDKへの依存は `harness/agents/base.py` の `ClaudeAgentRunner` に閉じている。
`AgentRunner` Protocolは `capabilities()` / `resolve(request) -> ResolvedAgentRunSpec` /
`run(spec)` の3段階で、実行内容の確定と実行が分離されている。これを実装すれば
他のエンジン(Codex、ローカルLLM等)へ差し替え可能で、`config.yaml` の
`provider.roles` によりロールごとに異なるProvider/Modelを選択できる。
Provider固有機能(hook / telemetry / resume等)はCapabilityで宣言し、
Coreの正当性条件には含めない。

## 開発

```bash
python -m pytest -q     # LLM不要。FakeRunnerでオーケストレーション全体を検証
```

主要モジュール:

- `harness/orchestrator/` — Project/Taskループ、State Machine、Recovery、Budget
- `harness/agents/` — Provider抽象化とロール定義
- `harness/context/` — Project / Task / Attempt の3階層Context Builder
- `harness/database/` — SQLiteリポジトリ層(projects / tasks / task_attempts / agent_runs / evaluations / events / operations)
- `harness/verification/` — Deterministic Verification(java / python / node プリセット)
- `harness/git/` — Git操作とCheckpoint
- `harness/security/` — 禁止コマンド・Tool Permission・PreToolUse Hook
- `harness/prompts/` — 各ロールのシステムプロンプト(config.yamlと同じディレクトリに `prompts/` を置くと上書き可能)
