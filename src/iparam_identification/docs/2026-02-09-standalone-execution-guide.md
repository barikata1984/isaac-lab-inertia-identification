# ROS 2非依存での軌道最適化と可視化ガイド

**日付**: 2026-02-09
**目的**: ROS 2環境を使わずに、スタンドアロンで慣性パラメータ推定用の軌道最適化と可視化を実行する方法

---

## 背景

このプロジェクトは元々ROS 2ワークスペースとして設計されていたが、Isaac Sim環境でROS 2を使わずにスタンドアロンで実行したいという要件が発生した。本ドキュメントでは、その実装方法と遭遇した問題の解決策を記録する。

## 環境

- **コンテナ**: Isaac Sim 5.1.0 + Isaac Lab
- **Python**: Isaac Sim内蔵Python 3.11
- **エイリアス**: `isaac-python` (Isaac SimのPython実行)、`isaac-pip` (パッケージインストール)
- **ワークスペース**: `/workspace` (ROS 2 colcon構造だが、colcon buildは使用しない)

---

## 問題1: モジュールが見つからないエラー

### エラー内容

```bash
$ isaac-python src/iparam_identification/scripts/run_optimization.py
ModuleNotFoundError: No module named 'kinematics'
```

### 原因

ROS 2ワークスペースのパッケージ（`kinematics`, `trajectories`, `iparam_identification`）が、Pythonのインポートパスに含まれていない。通常はROS 2の`source install/setup.bash`でパスが設定されるが、今回はROS 2を使わない。

### 解決策: 開発モードでパッケージをインストール

各パッケージを`pip install -e`（editable mode）でインストールすることで、ROS 2なしでインポート可能にする。

```bash
# 依存順にインストール
isaac-pip install -e /workspace/src/kinematics
isaac-pip install -e /workspace/src/trajectories
isaac-pip install -e /workspace/src/iparam_identification
```

**注意事項**:
- `trajectories`パッケージは`docstring-parser`のバージョン要求が若干異なるが、動作には影響しない
- 開発モードなので、ソースコードの変更は即座に反映される

---

## 軌道最適化の実行

### 基本コマンド

```bash
isaac-python src/iparam_identification/scripts/run_optimization.py
```

### ボックス制約付き最適化（推奨）

タスク空間でのボックス制約を設定して最適化:

```bash
isaac-python src/iparam_identification/scripts/run_optimization.py \
  --harmonics 5 \
  --duration 10.0 \
  --restarts 20 \
  --max-iter 300 \
  --box-lower -0.3 -0.3 -0.2 \
  --box-upper 0.3 0.3 0.4 \
  --output src/iparam_identification/data/optimized_trajectory_box.json
```

### クイックテスト（開発用）

時間を短縮したテスト実行:

```bash
isaac-python src/iparam_identification/scripts/run_optimization.py \
  --harmonics 3 \
  --restarts 5 \
  --max-iter 100 \
  --box-lower -0.3 -0.3 -0.2 \
  --box-upper 0.3 0.3 0.4
```

### 主要パラメータ

| パラメータ | 説明 | デフォルト | 推奨値（本格） | 推奨値（テスト） |
|-----------|------|-----------|--------------|----------------|
| `--harmonics` | フーリエ高調波の数 | 5 | 5 | 3 |
| `--duration` | 軌道の長さ（秒） | 10.0 | 10.0 | 10.0 |
| `--base-freq` | 基本周波数（Hz） | 0.1 | 0.1 | 0.1 |
| `--restarts` | モンテカルロリスタート回数 | 20 | 20 | 5 |
| `--max-iter` | リスタートごとの最大反復数 | 200 | 300 | 100 |
| `--box-lower X Y Z` | ボックス制約下限[m] | なし | -0.3 -0.3 -0.2 | 同左 |
| `--box-upper X Y Z` | ボックス制約上限[m] | なし | 0.3 0.3 0.4 | 同左 |
| `--output` | 出力ファイルパス | data/optimized_trajectory.json | 任意 | 任意 |

### 実行結果の例

**クイックテスト** (3高調波、5リスタート、100反復):
- 実行時間: 約4分
- 条件数: 2.24
- 評価回数: 19,153回
- 全制約充足: ✅

**本格最適化** (5高調波、20リスタート、300反復):
- 実行時間: 約74分
- 条件数: 1.87-2.00（より良い）
- 評価回数: 360,000回以上
- 全制約充足: ✅

---

## 問題2: 軌道の可視化

### エラー内容

```bash
$ isaac-python src/iparam_identification/scripts/visualize_trajectory.py
ModuleNotFoundError: No module named 'ur'
```

### 原因

`visualize_trajectory.py`は`ur.spawning`というROS 2パッケージに依存しており、これがインストールされていない。

### 解決策: run_isaac_identification.pyを使用

`run_isaac_identification.py`はIsaac Labを使用しており、`ur.spawning`に依存していない。このスクリプトで軌道の再生と慣性パラメータ推定の両方が可能。

```bash
# GUIモードで軌道を再生（デフォルト）
isaac-python src/iparam_identification/scripts/run_isaac_identification.py

# 特定の軌道ファイルを指定
isaac-python src/iparam_identification/scripts/run_isaac_identification.py \
  --trajectory src/iparam_identification/data/optimized_trajectory_box.json

# ヘッドレスモード（GUI不要、高速実行）
isaac-python src/iparam_identification/scripts/run_isaac_identification.py --headless
```

### 実行内容

`run_isaac_identification.py`は以下を実行:
1. Isaac SimでUR5eロボットとアルミニウムペイロードをスポーン
2. 指定された軌道を実行（デフォルトは`data/optimized_trajectory.json`）
3. 力/トルクと関節データを収集
4. 3つの手法で慣性パラメータを推定:
   - OLS (Ordinary Least Squares)
   - TLS (Total Least Squares)
   - RTLS (Recursive Total Least Squares)
5. 推定結果と真値を比較表示

---

## 最適化済み軌道ファイルの比較

現在のワークスペースには2つの最適化済み軌道ファイルが存在:

### optimized_trajectory.json (最新)
- **作成日時**: 2026-02-09 05:00
- **条件数**: 2.24
- **高調波数**: 3
- **最適化パラメータ**: 5リスタート、100反復
- **ボックス制約**: ✅ あり
- **用途**: クイックテスト

### optimized_trajectory_box.json (高品質)
- **作成日時**: 2026-02-09 03:55
- **条件数**: 2.00（より良い）
- **高調波数**: 5
- **最適化パラメータ**: より多くのリスタート/反復
- **ボックス制約**: ✅ あり
- **用途**: 本格的な実験・評価

### 推奨

**`optimized_trajectory_box.json`の使用を推奨**

理由:
- 条件数が低い（2.00 vs 2.24）→ レグレッサ行列の数値的安定性が高い
- 5高調波使用 → より豊富な励起、より多くの姿勢をカバー
- より多くのリスタート/反復 → より良い局所最適解を見つけている可能性が高い

使用方法:
```bash
isaac-python src/iparam_identification/scripts/run_isaac_identification.py \
  --trajectory src/iparam_identification/data/optimized_trajectory_box.json
```

---

## 条件数の解釈

条件数（κ）は、レグレッサ行列の数値的安定性を示す指標:

- **κ < 10**: 非常に良好（数値的に安定）
- **κ = 10-100**: 良好
- **κ = 100-1000**: 注意が必要
- **κ > 1000**: 不安定（推定精度が低下する可能性）

本実装の結果:
- **最適化前**: κ ≈ 4.57（ランダム係数）
- **最適化後**: κ ≈ 1.87-2.24（約60-90%改善）

参考文献（Kubus et al. 2008）では、κ = 2-3程度が典型的な最適化結果とされており、本実装の結果は妥当。

---

## トラブルシューティング

### パッケージのインポートエラー

```bash
ModuleNotFoundError: No module named 'XXX'
```

→ パッケージを開発モードで再インストール:
```bash
isaac-pip install -e /workspace/src/XXX
```

### 軌道ファイルが見つからない

```bash
FileNotFoundError: [Errno 2] No such file or directory: 'data/optimized_trajectory.json'
```

→ `--trajectory`オプションで絶対パスまたは正しい相対パスを指定:
```bash
isaac-python src/iparam_identification/scripts/run_isaac_identification.py \
  --trajectory src/iparam_identification/data/optimized_trajectory_box.json
```

### 最適化が遅い

→ パラメータを減らしてクイックテスト:
```bash
isaac-python src/iparam_identification/scripts/run_optimization.py \
  --harmonics 3 --restarts 3 --max-iter 50
```

---

## ワークフロー推奨

### 1. 初回セットアップ

```bash
# パッケージをインストール
isaac-pip install -e /workspace/src/kinematics
isaac-pip install -e /workspace/src/trajectories
isaac-pip install -e /workspace/src/iparam_identification
```

### 2. 軌道の最適化

```bash
# クイックテストで動作確認
isaac-python src/iparam_identification/scripts/run_optimization.py \
  --harmonics 3 --restarts 5 --max-iter 100 \
  --box-lower -0.3 -0.3 -0.2 --box-upper 0.3 0.3 0.4

# 本格的な最適化（時間があるとき）
isaac-python src/iparam_identification/scripts/run_optimization.py \
  --harmonics 5 --restarts 20 --max-iter 300 \
  --box-lower -0.3 -0.3 -0.2 --box-upper 0.3 0.3 0.4 \
  --output src/iparam_identification/data/optimized_trajectory_best.json
```

### 3. 軌道の再生と推定

```bash
# GUIで軌道を確認しながら推定
isaac-python src/iparam_identification/scripts/run_isaac_identification.py \
  --trajectory src/iparam_identification/data/optimized_trajectory_box.json

# ヘッドレスで高速実行
isaac-python src/iparam_identification/scripts/run_isaac_identification.py \
  --trajectory src/iparam_identification/data/optimized_trajectory_box.json \
  --headless
```

### 4. データの保存（オプション）

```bash
# 推定データをNPZ形式で保存
isaac-python src/iparam_identification/scripts/run_isaac_identification.py \
  --trajectory src/iparam_identification/data/optimized_trajectory_box.json \
  --save-data results/experiment_2026-02-09.npz
```

---

## まとめ

### 実装した解決策

1. **パッケージ依存の解決**: ROS 2の`colcon build`を使わず、`pip install -e`で開発モードインストール
2. **軌道最適化**: ボックス制約付きで条件数を約60-90%改善（κ ≈ 4.57 → 1.87-2.24）
3. **軌道可視化**: `ur.spawning`依存を回避し、`run_isaac_identification.py`で代替

### 利点

- ROS 2環境のセットアップが不要
- Isaac Sim環境で直接実行可能
- ソースコードの変更が即座に反映（開発モード）
- GUIモードとヘッドレスモードの両方をサポート

### 制限事項

- `visualize_trajectory.py`は現状では使用不可（`ur.spawning`依存）
  - 将来的にIsaac Lab方式に書き換える必要あり
- パッケージのインストールはコンテナの再起動後も必要
  - Dockerイメージに焼き込むか、起動スクリプトで自動化することを推奨

### 今後の改善案

1. **visualize_trajectory.pyの修正**: Isaac Lab方式でUR5eをスポーンするように書き換え
2. **自動インストールスクリプト**: コンテナ起動時に自動的にパッケージをインストール
3. **ドキュメントの充実**: 各パラメータの影響を詳細に記述
4. **ベンチマーク**: 異なるパラメータでの最適化結果を体系的に比較

---

## 参考文献

- Kubus et al. (2008): "On-line estimation of inertial parameters using a recursive total least-squares approach"
- [implementation_log.md](implementation_log.md): Phase 3の詳細な実装記録
- [excitation_trajectory_optimization_plan.md](excitation_trajectory_optimization_plan.md): 最適化アルゴリズムの詳細

---

**最終更新**: 2026-02-09
**著者**: Claude Code (Anthropic)
**レビュー**: 要
