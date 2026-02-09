# プロジェクト構成ガイド

Isaac Sim 上の UR5e ロボットを用いた慣性パラメータ同定システム。
ペイロードを把持した状態で励起軌道を実行し、力覚・関節状態データから
質量・重心・慣性テンソルの 10 パラメータを Recursive Total Least-Squares (RTLS) で推定する。

**参考文献:** Kubus, D., Kroger, T., & Wahl, F. M. (2008).
*On-line estimation of inertial parameters using a recursive total least-squares approach.*
IEEE/RSJ International Conference on Intelligent Robots and Systems.

---

## プロジェクト全体構成

```
iparam-identification/
│
├── pyproject.toml                 # PEP 621 プロジェクト定義
├── README.md                      # コンテナ環境のセットアップ手順
├── setup.sh                       # 初回ホストセットアップ（.env 生成, Docker ビルド）
├── .gitignore
│
├── docker/                        # コンテナ環境
│   ├── Dockerfile                 # Isaac Sim 5.1.0 + Isaac Lab ベース
│   ├── docker-compose.yml         # オーケストレーション（GPU, X11, マウント設定）
│   ├── entrypoint.sh              # コンテナ起動時のユーザー環境セットアップ
│   └── requirements.txt           # Isaac Sim Python 向け追加パッケージ
│
├── src/                           # ライブラリパッケージ（pip install -e . でインストール）
│   ├── kinematics/                # Pinocchio ベース運動学
│   ├── trajectories/              # 軌道生成（Fourier, スプライン, 窓関数）
│   ├── sensor/                    # 力覚センサインタフェース・データバッファ
│   ├── models/                    # ロボット定数・ペイロードモデル
│   ├── collision_check/           # FK ベース自己干渉チェック
│   ├── iparam_identification/     # 推定アルゴリズム・励起軌道最適化・ユーティリティ
│   └── isaac_utils/               # Isaac Sim 共通ユーティリティ
│
├── scripts/                       # 実行スクリプト
│   ├── run_optimization.py        # 励起軌道最適化（Isaac Sim 不要）
│   ├── spawn_ur5e.py              # UR5e スポーン確認（Isaac Sim GUI）
│   └── run_isaac_identification.py # 慣性パラメータ同定（Isaac Sim）
│
├── tests/                         # pytest テストスイート
│   ├── conftest.py
│   ├── test_kinematics.py
│   ├── test_estimation.py
│   ├── test_sensor.py
│   └── test_trajectory_optimization.py
│
├── data/                          # 最適化結果（スクリプトが生成）
│   ├── optimized_trajectory.json
│   └── optimized_trajectory_box.json
│
└── docs/                          # 設計文書
```

---

## src/ — ライブラリパッケージ

`pyproject.toml` の `[tool.setuptools.packages.find] where = ["src"]` により、
`/isaac-sim/python.sh -m pip install -e /workspace` で src/ 以下の全パッケージがインポート可能になる。

### kinematics/

Pinocchio (pin) を用いた UR ロボットの運動学計算。

- `kinematics.py` — `PinocchioKinematics` クラス
  - `for_ur5e()` — ROS 2 ur_description からモデル生成
  - `from_urdf_path(path)` — URDF ファイルから直接生成（ROS 2 不要）
  - `forward_kinematics(q)` — tool0 の位置・姿勢
  - `compute_regressor(q, dq, ddq, gravity)` — (6, 10) レグレッサ行列

出力規約: `[linear, angular]` (v, omega)。

### trajectories/

軌道の生成と評価。`@dataclass(kw_only=True)` の Config から軌道インスタンスを生成する設計。

| モジュール | クラス | 用途 |
|-----------|--------|------|
| `base_trajectory.py` | `BaseTrajectory`, `BaseTrajectoryConfig` | 全軌道の基底 |
| `fourier.py` | `FourierTrajectory` | Fourier 級数軌道 |
| `spline.py` | `SplineTrajectory` | 7次スプライン補間 |
| `window.py` | `WindowTrajectory` | 多項式窓関数（q(0)=q(T)=q0 を保証） |
| `windowed_fourier.py` | `WindowedFourierTrajectory` | 窓関数 × Fourier（励起軌道最適化で使用） |
| `windowed_fourier_spline.py` | `WindowedFourierSplineTrajectory` | スプライン + Fourier 混合 |
| `generator_cli.py` | — | `tyro.cli()` ベースの CLI（`trajectory-generator` コマンド） |

### sensor/

シミュレーション環境でのセンサデータ取得と蓄積。

| モジュール | 主要クラス | 説明 |
|-----------|-----------|------|
| `data_types.py` | `SensorData`, `WrenchStamped`, `EstimationResult` | データ型定義 |
| `data_buffer.py` | `DataBuffer` | 時系列データの蓄積。`get_stacked_data(kin)` でレグレッサ行列 A と観測ベクトル y を一括生成 |
| `contact_sensor.py` | `SimulatedForceSensor` | FK + レグレッサによる力覚シミュレーション（ガウスノイズ付加） |

### models/

ロボット固有の定数とペイロードモデル。Isaac Sim 非依存。

```
models/
├── robots/ur/ur5e.py    # URDF_PATH, Q0_IDENTIFICATION, JOINT_NAMES
└── payloads/cuboid.py   # CuboidPayload dataclass
```

`models.robots.ur.ur5e`:
- `URDF_PATH` — Isaac Sim コンテナ内の UR5e URDF 絶対パス
- `Q0_IDENTIFICATION` — 同定用初期姿勢 [π/2, -π/2, π/2, -π/2, -π/2, π/2] rad
- `JOINT_NAMES` — 6関節名のリスト

`models.payloads.cuboid`:
- `CuboidPayload` — 直方体ペイロード。寸法と密度から `mass`, `inertia_tensor`, `inertia_at_tool0`（平行軸定理）, `phi_true`（10次元慣性パラメータベクトル）を自動算出

### collision_check/

FK ベースの自己干渉チェッカー。Isaac Sim 非依存。

| モジュール | クラス | 用途 |
|-----------|--------|------|
| `collision_checker.py` | `CollisionChecker`, `CollisionConfig` | 球体近似。高速。励起軌道最適化の制約関数で使用 |
| `capsule_checker.py` | `CapsuleCollisionChecker` | カプセル近似。高精度。Isaac Sim 上でのランタイム安全チェックで使用 |

### iparam_identification/

慣性パラメータ同定の中核。3つのサブパッケージで構成。

#### estimation/ — 推定アルゴリズム

| モジュール | クラス | 手法 |
|-----------|--------|------|
| `batch_ls.py` | `BatchLeastSquares` | OLS, 重み付き LS, IRLS |
| `batch_tls.py` | `BatchTotalLeastSquares` | TLS, 一般化 TLS, 正則化 TLS |
| `rls.py` | `RecursiveLeastSquares`, `RecursiveInstrumentalVariable` | RLS（忘却係数付き） |
| `rtls.py` | `RecursiveTotalLeastSquares`, `WindowedRTLS`, `AdaptiveRTLS` | RTLS（メイン手法） |
| `svd_update.py` | `SVDState` | インクリメンタル SVD 更新 |
| `base_estimator.py` | `BatchEstimatorBase`, `RecursiveEstimatorBase` | 基底クラス, `compute_condition_number()` |

#### excitation_trajectory/ — 励起軌道最適化

- `excitation_optimizer.py` — `ExcitationOptimizer`: SLSQP + Monte Carlo 多点開始によるレグレッサ条件数最小化。`OptimizerConfig` で全パラメータを設定
- `constraints.py` — 関節位置/速度/加速度限界、ワークスペース変位制約、衝突回避制約を SciPy constraints として構築

#### utils/ — ユーティリティ

- `trajectory_io.py` — `load_optimized_trajectory()`: 最適化結果 JSON の読み込み。`generate_fallback_trajectory()`: JSON がないときの代替軌道生成
- `reporting.py` — `print_results()`: OLS/TLS/RTLS 推定結果の比較テーブル出力

### isaac_utils/

Isaac Sim / Isaac Lab 依存のユーティリティ。AppLauncher 起動後にインポートする（`preload_libassimp` のみ例外）。

- `bootstrap.py`
  - `preload_libassimp()` — cmeel 版 libassimp を先読みし、hpp-fcl とのシンボル衝突を回避。**AppLauncher 前**に呼ぶ
  - `setup_quit_handler()` — GUI ウィンドウ閉じイベントの購読。`os._exit(0)` で強制終了（`sim.step()` がポーズ中ブロックするため）。**AppLauncher 後**に呼ぶ
- `scene.py`
  - `make_ur5e_cfg(q0, enable_self_collisions)` — UR5e の `ArticulationCfg` を生成
  - `design_scene(ur5e_cfg)` — Ground + DomeLight + UR5e のシーン構築
  - `create_and_attach_payload(payload)` — USD でペイロード直方体を作成し、tool0 に FixedJoint で接続

---

## scripts/ — 実行スクリプト

### run_optimization.py

励起軌道のオフライン最適化。**Isaac Sim 不要**。

WindowedFourierTrajectory の Fourier 係数を SLSQP で最適化し、
レグレッサ行列の条件数を最小化する。関節限界・ワークスペース・衝突回避の制約付き。
結果は `data/optimized_trajectory.json` に保存される。

```bash
# デフォルト設定で実行
/isaac-sim/python.sh scripts/run_optimization.py

# パラメータ指定
/isaac-sim/python.sh scripts/run_optimization.py \
    --harmonics 5 --duration 10.0 --base-freq 0.1 \
    --restarts 20 --max-iter 200 --seed 42

# ボックス制約付き
/isaac-sim/python.sh scripts/run_optimization.py \
    --box-lower -0.3 -0.3 -0.2 --box-upper 0.3 0.3 0.4

# 短時間テスト
/isaac-sim/python.sh scripts/run_optimization.py \
    --harmonics 2 --restarts 2 --max-iter 10
```

### spawn_ur5e.py

UR5e を Isaac Sim GUI にスポーンし、PD 制御でデフォルト姿勢を保持する。
シーン構築と物理シミュレーションの動作確認用。**Isaac Sim 必要**。

```bash
/isaac-sim/python.sh scripts/spawn_ur5e.py
```

### run_isaac_identification.py

慣性パラメータ同定のメインスクリプト。**Isaac Sim 必要**。

1. UR5e + CuboidPayload のシーンを構築
2. 最適化済み軌道（または代替軌道）に沿って PD 制御で駆動
3. 各タイムステップで力覚センサデータを収集（ノイズ付き）
4. RTLS でオンライン推定、最後に OLS / TLS でバッチ推定
5. 真値との比較テーブルを出力

```bash
# GUI モード
/isaac-sim/python.sh scripts/run_isaac_identification.py

# ヘッドレスモード
/isaac-sim/python.sh scripts/run_isaac_identification.py --headless

# ノイズ設定・軌道・結果保存
/isaac-sim/python.sh scripts/run_isaac_identification.py \
    --headless \
    --noise-force 1.0 --noise-torque 0.1 \
    --noise-encoder-pos 1e-4 --noise-encoder-vel 1e-3 \
    --trajectory data/optimized_trajectory.json \
    --save-data results.npz
```

---

## docker/ — コンテナ環境

NVIDIA Isaac Sim 5.0.0 ベースの開発コンテナ。GPU パススルー + X11 転送により GUI 表示に対応。

- `Dockerfile` — Isaac Sim イメージ上に Isaac Lab + 開発ツールをインストール
- `docker-compose.yml` — GPU ランタイム, X11 ソケット, ワークスペースマウントの設定
- `entrypoint.sh` — コンテナ内ユーザー環境の初期化
- `requirements.txt` — Isaac Sim Python 向けの追加パッケージ（debugpy, ruff, pytest 等）

```bash
# 初回セットアップ（ホスト側）
./setup.sh

# コンテナ起動
docker compose -f docker/docker-compose.yml up -d

# コンテナに入る
docker compose -f docker/docker-compose.yml exec isaac-lab-inertia-identification zsh
```

---

## tests/ — テストスイート

```bash
# 全テスト
/isaac-sim/python.sh -m pytest tests/ -v

# 個別テスト
/isaac-sim/python.sh -m pytest tests/test_trajectory_optimization.py -x
```

| ファイル | 対象 |
|---------|------|
| `conftest.py` | 共通 fixture: 関節状態, 重力ベクトル（`scipy.constants.g`）, `PinocchioKinematics` |
| `test_kinematics.py` | FK 計算、レグレッサ行列の形状・対称性 |
| `test_estimation.py` | OLS / TLS / RLS / RTLS の推定精度、ノイズ耐性 |
| `test_sensor.py` | SimulatedForceSensor の出力検証、ノイズ特性 |
| `test_trajectory_optimization.py` | 軌道生成、条件数目的関数、制約充足、ExcitationOptimizer 統合テスト |

---

## 典型的なワークフロー

```
1. 軌道最適化（オフライン）
   $ /isaac-sim/python.sh scripts/run_optimization.py
   → data/optimized_trajectory.json

2. 動作確認（オプション）
   $ /isaac-sim/python.sh scripts/spawn_ur5e.py
   → GUI でロボットが正しくスポーンされるか確認

3. 慣性パラメータ同定
   $ /isaac-sim/python.sh scripts/run_isaac_identification.py --headless
   → 軌道実行 → データ収集 → OLS/TLS/RTLS 推定 → 結果比較表
```

---

## pyproject.toml 概要

| 項目 | 値 |
|------|-----|
| パッケージ名 | `iparam-identification` |
| バージョン | 0.3.0 |
| Python | >= 3.10 |
| ライセンス | MIT |
| ビルド | setuptools (src-layout) |
| 依存 | numpy, scipy, matplotlib, pin (Pinocchio), tyro |
| 開発依存 | pytest, pytest-xdist, ruff, mypy |
| CLI コマンド | `trajectory-generator` → `trajectories.generator_cli:entry_point` |
| テスト | pytest (`tests/`), マーカー: `slow`, `isaac` |
| リント | ruff (E, W, F, I, UP, B), isort 設定あり |
