# Isaac Sim 慣性パラメータ推定 統合スクリプト 実装計画

## Context

UR5e ロボットが把持した物体の慣性パラメータ（質量、重心、慣性テンソル）を Isaac Sim 上で推定するパイプラインを構築する。推定アルゴリズム（RTLS等）、軌道生成、センサモジュールは実装済み。不足しているのは「Isaac Lab API でこれらを統合する実行スクリプト」と「ROS2 に依存しない kinematics ローディング」。

## 変更対象ファイル

### 1. `src/kinematics/src/kinematics.py` — URDF 直接読み込みの追加

`PinocchioKinematics` に `from_urdf_path()` クラスメソッドを追加。ROS2（ament_index_python, xacro）に依存せず、URDF ファイルパスから直接 Pinocchio モデルを構築する。

```python
@classmethod
def from_urdf_path(cls, urdf_path: str, tool0_frame_name: str = "tool0") -> "PinocchioKinematics":
```

- Isaac Sim 同梱 URDF: `/isaac-sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf`
- URDF に `tool0` フレームが存在することを確認済み（`flange-tool0` fixed joint）
- 既存の `for_ur5e()` / `load_ur_model()` は変更しない（後方互換）

### 2. `src/iparam_identification/scripts/run_isaac_identification.py` — 新規作成（メイン）

Isaac Lab API ベースの統合スクリプト。`scripts/spawn_ur5e.py` のパターンを踏襲。

**構成:**

```
[A] AppLauncher ブートストラップ + CLI 引数
[B] GUI クローズハンドラ（carb event dispatcher + os._exit(0)）
[C] CuboidPayload（run_identification_test.py から流用）
[D] UR5E_CFG（spawn_ur5e.py から流用）
[E] design_scene() — ロボット + 地面 + ライト
[F] create_and_attach_payload() — ペイロード生成 & tool0 への固定結合
[G] load_trajectory() — optimized_trajectory.json or フォールバック生成
[H] run_identification() — メインループ（下記参照）
[I] print_results() — 結果比較テーブル
[J] main()
```

**メインループ [H] のデータフロー（各タイムステップ）:**

```
1. robot.set_joint_position_target(q_des[i])  ← Isaac Lab API
2. robot.write_data_to_sim()
3. sim.step()
4. robot.update(dt)
5. q  = robot.data.joint_pos[0].cpu().numpy()
6. dq = robot.data.joint_vel[0].cpu().numpy()
7. ddq = (dq - prev_dq) / dt                  ← 有限差分
8. A_k = kin.compute_regressor(q, dq, ddq, gravity)  ← (6, 10)
9. wrench = sensor.measure(q, dq, ddq, t)     ← SimulatedForceSensor（ノイズ付き逆動力学）
10. y_k = wrench.wrench                        ← (6,)
11. phi_hat = rtls.update(A_k, y_k)            ← RTLS オンライン更新
12. data_buffer.add_sample(SensorData(...))     ← バッチ推定用に蓄積
13. convergence_log に記録
```

**シミュレーション後:**
- DataBuffer.get_stacked_data(kin) でバッチデータを構築
- BatchLeastSquares, BatchTotalLeastSquares で推定
- RTLS 結果と合わせて比較テーブルを出力

**CLI 引数:**
- `--headless` / GUI（AppLauncher 標準）
- `--noise-force` (default: 0.5 N)
- `--noise-torque` (default: 0.05 Nm)
- `--trajectory` (JSON パス、default: optimized_trajectory.json)
- `--save-data` (NPZ 保存パス)

**再利用するクラス・関数:**
- `CuboidPayload`: [run_identification_test.py:79-168](src/iparam_identification/scripts/run_identification_test.py#L79-L168) からコピー
- `create_payload_prim`: [run_identification_test.py:298-353](src/iparam_identification/scripts/run_identification_test.py#L298-L353) からコピー
- `attach_payload_to_tool0`: [run_identification_test.py:356-399](src/iparam_identification/scripts/run_identification_test.py#L356-L399) からコピー
- `load_optimized_trajectory`: [run_identification_test.py:255-289](src/iparam_identification/scripts/run_identification_test.py#L255-L289) を参考に修正
- `generate_safe_trajectory`: [run_identification_test.py:171-252](src/iparam_identification/scripts/run_identification_test.py#L171-L252) をフォールバック用にコピー
- `UR5E_CFG`: [spawn_ur5e.py:65-106](scripts/spawn_ur5e.py#L65-L106) からコピー
- GUI クローズパターン: [spawn_ur5e.py](scripts/spawn_ur5e.py) から流用
- `SimulatedForceSensor`: [contact_sensor.py:348-414](src/iparam_identification/src/sensor/contact_sensor.py#L348-L414)
- `RecursiveTotalLeastSquares`: [rtls.py](src/iparam_identification/src/estimation/rtls.py)
- `BatchLeastSquares`, `BatchTotalLeastSquares`: [batch_ls.py](src/iparam_identification/src/estimation/batch_ls.py), [batch_tls.py](src/iparam_identification/src/estimation/batch_tls.py)
- `DataBuffer`, `SensorData`: [data_buffer.py](src/iparam_identification/src/sensor/data_buffer.py), [data_types.py](src/iparam_identification/src/sensor/data_types.py)

**注意: API の違い**
- 既存 `IsaacSimStateCollector` は旧 API（`isaacsim.core.api.robots.Robot`）前提
- 新スクリプトは Isaac Lab API（`Articulation`）を使用 → 状態取得は直接行う
- `IsaacSimStateCollector` は使わず、ロボット状態取得を直接記述する

### 3. `docker/requirements.txt` — Pinocchio 追加

```
pin
```

## 実装順序

1. **docker/requirements.txt** に `pin` を追加
2. **kinematics.py** に `from_urdf_path()` を追加
3. **run_isaac_identification.py** を段階的に実装:
   - Phase A: ブートストラップ + シーン構築（ロボット + ペイロードが GUI で見える）
   - Phase B: 軌道読み込み + 実行（ロボットが動く）
   - Phase C: RTLS オンライン推定 + バッチ比較 + 結果出力

## 検証方法

```bash
# 1. Pinocchio のインポート確認
isaac-python -c "import pinocchio; print(pinocchio.__version__)"

# 2. URDF 読み込み確認
isaac-python -c "
from kinematics import PinocchioKinematics
kin = PinocchioKinematics.from_urdf_path('/isaac-sim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e/ur5e.urdf')
print(f'Joints: {kin.n_joints}')
"

# 3. Headless モードで統合テスト
isaac-python src/iparam_identification/scripts/run_isaac_identification.py --headless

# 4. GUI モードで目視確認
isaac-python src/iparam_identification/scripts/run_isaac_identification.py
```

**期待される結果:**
- 質量推定誤差 < 5%（ノイズあり）
- RTLS が 20-30 サンプル後に収束
- OLS / TLS / RTLS の3手法比較テーブルが出力される

## 実装結果（2026-02-08）

### テスト環境

- Isaac Sim 4.5 + Isaac Lab（Docker コンテナ内）
- Pinocchio 2.7.0（Isaac Sim Python 環境にプリインストール済み）
- Headless モード（`--headless`）

### 再現手順

```bash
# コンテナ内で実行（/workspace がマウント済み前提）

# 1. Pinocchio の確認（追加インストール不要だった）
isaac-python -c "import pinocchio; print(pinocchio.__version__)"
# => 2.7.0

# 2. URDF 読み込み確認
isaac-python -c "
import sys; sys.path.insert(0, '/workspace/src/kinematics/src')
from kinematics import PinocchioKinematics
kin = PinocchioKinematics.from_urdf_path(
    '/isaac-sim/exts/isaacsim.robot_motion.motion_generation/'
    'motion_policy_configs/universal_robots/ur5e/ur5e.urdf'
)
print(f'Joints: {kin.n_joints}, Frames: {kin.model.nframes}')
"
# => Joints: 6, Frames: 23

# 3. Headless 統合テスト（デフォルト設定）
PYTHONUNBUFFERED=1 isaac-python \
    src/iparam_identification/scripts/run_isaac_identification.py --headless

# 4. GUI モードで目視確認
isaac-python \
    src/iparam_identification/scripts/run_isaac_identification.py

# 5. カスタムノイズ設定での実行例
PYTHONUNBUFFERED=1 isaac-python \
    src/iparam_identification/scripts/run_isaac_identification.py \
    --headless --noise-force 1.0 --noise-torque 0.1
```

### Headless テスト結果

**実行条件:**
- ペイロード: アルミ直方体 (10×15×20 cm, 真値 m=8.1 kg)
- 軌道: `optimized_trajectory.json`（5 高調波、基本周波数 0.1 Hz、10 秒、100 fps）
- ノイズ: 力 0.5 N, トルク 0.05 Nm（デフォルト）
- PD 制御: stiffness=800, damping=40

**推定結果:**

| Method | Mass [kg] | Error [%] | Residual |
|--------|-----------|-----------|----------|
| OLS    | 8.1000    | 0.00      | 27.49    |
| TLS    | 8.1007    | 0.01      | 3.35     |
| RTLS   | 8.1007    | 0.01      | 3.35     |

**条件数:** κ = 6.03

**RTLS 収束:**
- 初回推定 (t=0.10s): mass = 0.0000 kg（誤差 100.00%）
- 最終推定 (t=10.00s): mass = 8.1007 kg（誤差 0.01%）

### 実装上の注意点

1. **`PYTHONUNBUFFERED=1`**: Headless モードでは Python の stdout バッファリングにより途中経過が表示されないため、環境変数の設定を推奨。

2. **`tool0` prim のフォールバック**: Isaac Sim の USD ステージ上で `tool0` prim が見つからない場合、`wrist_3_link` にフォールバックしてペイロードを取り付ける。Pinocchio 側の `tool0` フレームは正常に使用される（推定精度に影響なし）。

3. **ROS2 依存の回避**: `trajectories` パッケージの `__init__.py` は `rclpy` に依存する `follower_node` をインポートするため、`_register_package(..., init=False)` でスキップし、必要なサブモジュールのみ直接インポートしている。

4. **float64 / float32 の型不一致**: Isaac Lab の `ArticulationCfg.InitialStateCfg` は float32 を期待するため、numpy の float64 値は `float()` でラップする必要がある。

### 今後の拡張

- **Phase 2**: Isaac Sim 物理エンジンからの力/トルク計測（現在は逆動力学ベースの `SimulatedForceSensor`）
- ノイズレベルを変化させたロバスト性検証
- 異なるペイロード形状・質量での追加テスト
- GUI モードでの目視確認と収束プロットの可視化
