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

全ロールで `git commit/push/merge/rebase/reset`、sudo、credential操作等は PreToolUse Hook が拒否する(`harness/security/`)。

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

## 中断・復旧

Harnessプロセスは途中終了を前提とする。起動時にSQLiteとGitの状態を照合し、
RUNNINGのままのAttemptを検出した場合:

1. Dirty Diff を `artifacts/diagnostics/` へ保存
2. `git reset --hard HEAD`(最後の正常Commitへ)
3. Attempt を INTERRUPTED として記録
4. Task を READY に戻し、Fresh Session で再実行

## Budget管理

Project / Task / Agent Run の3階層でUSD予算を管理し、加えて
`max_attempts` / `max_turns` / `max_agent_runs_per_task` / `max_execution_seconds`
を制限する。超過時はTaskをBLOCKED、ProjectをPAUSEDへ遷移させる(クラッシュしない)。
PAUSEDのProjectは予算を引き上げて `run` し直せば継続する。

## Provider抽象化

Claude Agent SDKへの依存は `harness/agents/base.py` の `ClaudeAgentRunner` に閉じている。
`AgentRunner` Protocolを実装すれば他のエンジン(Codex、ローカルLLM等)へ差し替え可能で、
`config.yaml` の `provider.roles` によりロールごとに異なるProvider/Modelを選択できる。

## 開発

```bash
python -m pytest -q     # LLM不要。FakeRunnerでオーケストレーション全体を検証
```

主要モジュール:

- `harness/orchestrator/` — Project/Taskループ、State Machine、Recovery、Budget
- `harness/agents/` — Provider抽象化とロール定義
- `harness/context/` — Project / Task / Attempt の3階層Context Builder
- `harness/database/` — SQLiteリポジトリ層(projects / tasks / task_attempts / agent_runs / evaluations / events)
- `harness/verification/` — Deterministic Verification(java / python / node プリセット)
- `harness/git/` — Git操作とCheckpoint
- `harness/security/` — 禁止コマンド・Tool Permission・PreToolUse Hook
- `harness/prompts/` — 各ロールのシステムプロンプト(config.yamlと同じディレクトリに `prompts/` を置くと上書き可能)
