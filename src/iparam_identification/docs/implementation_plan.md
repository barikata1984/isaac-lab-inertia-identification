# Inertial Parameter Identification - Implementation Plan

## Overview

UR5eロボットによる把持物体の慣性パラメータ同定システムの実装計画。
Kubus et al. 2008 の Recursive Total Least-Squares (RTLS) 手法に基づく。

### 目標

- 把持物体の10個の慣性パラメータを推定:
  - 質量 $m$
  - 重心位置 $mc_x, mc_y, mc_z$ (質量×重心座標)
  - 慣性テンソル $I_{xx}, I_{xy}, I_{xz}, I_{yy}, I_{yz}, I_{zz}$

### 前提条件

- **センサー**: Isaac Sim Contact Sensor
- **推定方式**: オフライン優先（バッチ処理）
- **対象**: 把持物体のみ（ロボットリンクは既知）
- **検証**: 既知構成物体 → objファイル付き物体

---

## 1. Background Knowledge (これまでの知見)

### 1.1 推定方程式 (Kubus et al. 2008, Eq. 4-6)

Newton-Euler方程式に基づく推定式:

$$\begin{pmatrix} f \\ \tau \end{pmatrix} = A(a, \alpha, \omega, g) \cdot \varphi$$

- $f, \tau$: 力/トルクセンサー測定値 (6×1)
- $A$: 回帰行列（データ行列）(6×10)
- $\varphi$: 慣性パラメータベクトル (10×1)

### 1.2 既存実装 (kinematics パッケージ)

| 機能 | ファイル | 状態 |
|------|----------|------|
| 順運動学 | `kinematics.py` | ✅ 実装済み |
| 速度・加速度計算 | `kinematics.py` | ✅ 実装済み |
| 回帰行列計算 | `kinematics.py:compute_regressor()` | ✅ 実装済み |
| フーリエ軌道 | `trajectories/fourier.py` | ✅ 実装済み |

### 1.3 推定アルゴリズムの比較 (論文 Section IV)

| 手法 | 誤差モデル | 特徴 |
|------|-----------|------|
| **RLS** | $y + e = A\varphi$ | データ行列Aを誤差なしと仮定。実際には不適切 |
| **RIV** | 同上 + instrumental variables | 相関ノイズに対応可能だがAの誤差は無視 |
| **RTLS** | $y + e = (A + E)\varphi$ | AとEの両方の誤差を考慮。最も高精度 |

### 1.4 データ行列の信号源 (論文 Section IV-A)

加速度・角速度の取得方法:
1. **加速度センサー直接測定**: ノイズあり、実際の動きを反映
2. **関節角度セットポイントから導出**: ノイズなし、理想軌道
3. **実関節角度から導出（カルマンフィルタ）**: ノイズ低減、実際の動きを反映

論文の結論: **線形加速度はセンサー、角加速度はフィルタリング**の組み合わせが最良

---

## 2. Implementation Phases

### Phase 1: センサー統合

**目的**: Isaac Sim から力/トルクデータと状態データを取得

#### 2.1 Contact Sensor Integration

```
src/sensor/
├── __init__.py
├── contact_sensor.py     # Isaac Sim Contact Sensor インターフェース
├── data_types.py         # データ型定義
└── data_buffer.py        # 時系列データバッファ
```

##### 2.1.1 contact_sensor.py

**機能**:
- Isaac Sim Contact Sensor API のラッパー
- tool0 (エンドエフェクタ) に作用する力/トルクの取得
- 座標変換（ワールド座標 → センサー座標）

**主要クラス**:
```python
class IsaacContactSensor:
    def __init__(self, prim_path: str, sensor_frame: str = "tool0"):
        """Isaac Sim Contact Sensor の初期化"""

    def get_wrench(self) -> Tuple[np.ndarray, np.ndarray]:
        """力/トルクを取得 (f, τ) in sensor frame"""

    def get_wrench_with_timestamp(self) -> WrenchStamped:
        """タイムスタンプ付き力/トルク"""
```

**検討事項**:
- [ ] Isaac Sim の Contact Sensor API の調査
- [ ] センサーフレームと tool0 フレームの関係確認
- [ ] サンプリングレートの確認（論文では1000Hz）

##### 2.1.2 data_types.py

**データ構造**:
```python
@dataclass
class SensorData:
    """単一時刻のセンサーデータ"""
    timestamp: float
    q: np.ndarray       # 関節角度 (6,)
    dq: np.ndarray      # 関節角速度 (6,)
    ddq: np.ndarray     # 関節角加速度 (6,)
    force: np.ndarray   # 力 (3,)
    torque: np.ndarray  # トルク (3,)

@dataclass
class EstimationData:
    """推定用データ（回帰行列形式）"""
    A: np.ndarray       # 回帰行列 (6, 10)
    y: np.ndarray       # 観測ベクトル [f; τ] (6,)
```

##### 2.1.3 data_buffer.py

**機能**:
- 時系列データの蓄積
- データの同期処理
- バッチ推定用のデータエクスポート

**主要クラス**:
```python
class DataBuffer:
    def __init__(self, max_samples: int = 10000):
        """データバッファの初期化"""

    def add_sample(self, data: SensorData) -> None:
        """サンプルを追加"""

    def get_estimation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """推定用データを取得 (A_stacked, y_stacked)"""

    def save_to_file(self, path: str) -> None:
        """データをファイルに保存"""

    def load_from_file(self, path: str) -> None:
        """ファイルからデータを読み込み"""
```

#### 2.2 センサーオフセット補償 (論文 Eq. 7-9)

**背景**: 力/トルクセンサーは時間変動するオフセットを持つ

**実装方法**:
1. 初期姿勢で重力ベクトル $g_{init}$ を記録
2. オフセット補正行列 $A_{ginit}$ を計算
3. 推定時に補正: $A_{offs} = A + A_{ginit}$

```python
def compute_offset_matrix(g_init: np.ndarray) -> np.ndarray:
    """オフセット補正行列を計算 (論文 Eq. 7)"""
```

#### 2.3 テスト計画

- [ ] Contact Sensor からのデータ取得テスト
- [ ] データバッファの動作確認
- [ ] 既知重量物体での力測定検証

---

### Phase 2: 推定アルゴリズム

**目的**: オフラインバッチ推定の実装、後にオンライン化

```
src/estimation/
├── __init__.py
├── base_estimator.py     # 推定器基底クラス
├── batch_ls.py           # バッチ最小二乗法
├── batch_tls.py          # バッチ全最小二乗法
├── rls.py                # 再帰最小二乗法
├── rtls.py               # 再帰全最小二乗法
└── svd_update.py         # インクリメンタルSVD
```

#### 2.4 バッチ推定（オフライン優先）

##### 2.4.1 batch_ls.py - 最小二乗法

**数式**:
$$\hat{\varphi} = (A^T A)^{-1} A^T y$$

**実装**:
```python
def batch_least_squares(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """バッチ最小二乗法による推定

    Args:
        A: スタックされた回帰行列 (N*6, 10)
        y: スタックされた観測ベクトル (N*6,)

    Returns:
        φ: 推定された慣性パラメータ (10,)
    """
```

##### 2.4.2 batch_tls.py - 全最小二乗法

**数式** (論文 Eq. 23):
$$y + e = (A + E)\varphi$$

**実装手順**:
1. 拡張行列 $[A | y]$ を構成
2. SVD分解: $[A | y] = U S V^T$
3. 最小特異値に対応する右特異ベクトルから解を計算

```python
def batch_total_least_squares(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    """バッチ全最小二乗法による推定"""
```

#### 2.5 再帰推定（オンライン用、Phase 2後半）

##### 2.5.1 rls.py - 再帰最小二乗法 (論文 Eq. 17-19)

```python
class RecursiveLeastSquares:
    def __init__(self, n_params: int = 10, Lambda: np.ndarray = None):
        """RLS推定器の初期化

        Args:
            n_params: パラメータ数
            Lambda: 測定ノイズ共分散行列 (6, 6)
        """
        self.phi_hat = np.zeros(n_params)  # パラメータ推定値
        self.Sigma = np.eye(n_params) * 1e6  # パラメータ共分散

    def update(self, A_k: np.ndarray, y_k: np.ndarray) -> np.ndarray:
        """1ステップ更新

        Args:
            A_k: 回帰行列 (6, 10)
            y_k: 観測ベクトル (6,)

        Returns:
            φ_hat: 更新後のパラメータ推定値
        """
```

##### 2.5.2 rtls.py - 再帰全最小二乗法 (論文の主要貢献)

**アルゴリズム概要** (論文 Section IV-D):
1. 初期SVDを計算（2-3サンプル）
2. 新データで SVD を増分更新（Brand [18]）
3. TLS解を計算

```python
class RecursiveTotalLeastSquares:
    def __init__(self, n_params: int = 10, weighting: np.ndarray = None):
        """RTLS推定器の初期化"""
        self.U = None  # 左特異ベクトル
        self.S = None  # 特異値
        self.initialized = False

    def initialize(self, A_init: np.ndarray, y_init: np.ndarray) -> None:
        """初期SVDを計算（2-3サンプル必要）"""

    def update(self, A_k: np.ndarray, y_k: np.ndarray) -> np.ndarray:
        """インクリメンタルSVD更新とTLS解計算"""
```

##### 2.5.3 svd_update.py - インクリメンタルSVD (Brand 2002)

**参考**: Brand, M. (2002). Incremental singular value decomposition of uncertain data with missing values.

```python
def incremental_svd_update(
    U: np.ndarray,
    S: np.ndarray,
    new_data: np.ndarray,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """SVDの増分更新

    Args:
        U: 現在の左特異ベクトル
        S: 現在の特異値（対角行列）
        new_data: 新しいデータ列
        eps: 数値精度閾値

    Returns:
        U_new: 更新後の左特異ベクトル
        S_new: 更新後の特異値
    """
```

#### 2.6 推定結果の後処理

```python
@dataclass
class EstimationResult:
    """推定結果"""
    mass: float
    center_of_mass: np.ndarray  # (3,) [cx, cy, cz]
    inertia_matrix: np.ndarray  # (3, 3) 対称行列

    # 生データ
    phi: np.ndarray  # (10,) 生の推定パラメータ

    # 品質指標
    condition_number: float
    residual_norm: float

    @classmethod
    def from_phi(cls, phi: np.ndarray) -> "EstimationResult":
        """φベクトルから構造化結果を生成"""
```

#### 2.7 テスト計画

- [ ] バッチLS/TLSの数値検証（合成データ）
- [ ] RTLSとバッチTLSの結果比較
- [ ] ノイズ耐性の評価

---

### Phase 3: 補助機能

**目的**: 推定精度向上のための軌道設計・最適化

```
src/trajectory/
├── __init__.py
├── excitation_trajectory.py  # 励起軌道設計
├── condition_optimizer.py    # 条件数最適化
└── jerk_limiter.py           # ジャーク制限
```

#### 2.8 励起軌道設計 (論文 Section III)

##### 2.8.1 条件数と推定精度

**重要な関係**:
- 相関行列: $\Upsilon = A_\Xi^T A_\Xi$
- 条件数 $\kappa(\Upsilon)$ が小さいほど推定が安定

##### 2.8.2 excitation_trajectory.py

```python
class ExcitationTrajectory:
    """パラメータ励起用の最適化軌道"""

    def __init__(
        self,
        n_joints: int = 6,
        n_harmonics: int = 5,
        base_freq: float = 0.5,
    ):
        """励起軌道の初期化

        Args:
            n_joints: 関節数
            n_harmonics: フーリエ級数の次数
            base_freq: 基本周波数 [Hz]
        """

    def optimize_coefficients(
        self,
        kinematics: PinocchioKinematics,
        joint_limits: np.ndarray,
        n_iterations: int = 100,
    ) -> None:
        """条件数を最小化する係数を探索"""

    def compute_condition_number(
        self,
        kinematics: PinocchioKinematics,
        duration: float,
        dt: float,
    ) -> float:
        """軌道の条件数を計算"""
```

##### 2.8.3 jerk_limiter.py (論文 Eq. 14-15)

**目的**: 軌道開始・終了時のジャーク制限

```python
def apply_jerk_limit(
    trajectory: np.ndarray,
    t_blend: float,
    dt: float,
) -> np.ndarray:
    """6次多項式によるジャーク制限の適用"""
```

#### 2.9 テスト計画

- [ ] 条件数計算の検証
- [ ] 最適化前後の条件数比較
- [ ] ジャーク制限軌道の連続性確認

---

## 3. Validation Strategy

### 3.1 Phase 1: 既知構成物体での検証 ✅ 完了

**手順**:
1. Isaac Sim で単純形状（直方体、円柱）を生成
2. 既知の質量・慣性パラメータを設定
3. 推定結果と真値を比較

**評価指標** (論文 Eq. 37):
$$e_{rel}(\hat{x}) = \left| \frac{\hat{x} - x_{th}}{x_{th}} \right| \times 100\%$$

**実施結果** (2026-02-04):
- テスト物体: アルミニウム直方体 (10×15×20cm, 8.1kg)
- 質量誤差: **0.01%** (OLS/TLS両方)
- 慣性誤差: **< 0.1%** (全成分)
- 条件数: 4.57（良好）
- 詳細: `implementation_log.md` 参照

### 3.2 Phase 2: objファイル物体での検証

**手順**:
1. 提供されたobjファイルと慣性パラメータを使用
2. Isaac Simにインポート
3. 推定結果と提供パラメータを比較

**ステータス**: 未実施（Phase 1で十分な精度が確認されたため、必要に応じて実施）

---

## 4. Package Structure (Current)

```
iparam_identification/
├── package.xml
├── setup.py
├── pytest.ini
├── resource/
│   └── iparam_identification
├── scripts/                          # ✅ 追加
│   ├── run_test.sh                   # テスト実行スクリプト
│   └── run_identification_test.py    # Isaac Sim統合テスト
├── src/
│   ├── __init__.py
│   ├── sensor/                       # ✅ 完了
│   │   ├── __init__.py
│   │   ├── contact_sensor.py
│   │   ├── data_types.py
│   │   └── data_buffer.py
│   ├── estimation/                   # ✅ 完了
│   │   ├── __init__.py
│   │   ├── base_estimator.py
│   │   ├── batch_ls.py
│   │   ├── batch_tls.py
│   │   ├── rls.py
│   │   ├── rtls.py
│   │   └── svd_update.py
│   └── trajectory/                   # 🔄 未実装（任意）
│       ├── __init__.py
│       ├── excitation_trajectory.py
│       ├── condition_optimizer.py
│       └── jerk_limiter.py
├── test/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_sensor.py                # 36件
│   └── test_estimation.py            # 37件
└── notes/
    ├── implementation_plan.md
    ├── implementation_log.md
    └── references.md
```

---

## 5. Dependencies

### ROS 2 パッケージ
- `kinematics`: 回帰行列計算、運動学（Pinocchio使用）
- `trajectories`: フーリエ軌道生成
- `ur`: ロボットスポーン

### Python ライブラリ
- `numpy`: 数値計算
- `scipy`: SVD、最適化

### Isaac Sim 環境専用
- `pinocchio` (pin==2.7.0): 運動学計算（Isaac Sim Python 3.11にインストール）
- `isaacsim`: シミュレーションAPI

### 注意事項
- Pinocchioは**Isaac Sim の Python環境**に直接インストールが必要
  ```bash
  /isaac-sim/python.sh -m pip install pin==2.7.0
  ```
- ROS側のapt版pinocchioとは共存しない（削除推奨）

---

## 6. Timeline

| Phase | 内容 | 主要成果物 | 状態 |
|-------|------|-----------|------|
| **1** | センサー統合 | `contact_sensor.py`, `data_buffer.py` | ✅ 完了 |
| **2a** | バッチ推定 | `batch_ls.py`, `batch_tls.py` | ✅ 完了 |
| **2b** | 再帰推定 | `rls.py`, `rtls.py`, `svd_update.py` | ✅ 完了 |
| **V** | Isaac Sim検証 | `run_identification_test.py` | ✅ 完了 |
| **3** | 軌道最適化 | `excitation_trajectory.py` | 🔄 任意 |

---

## 7. Open Questions → 解決済み

1. **Isaac Sim Contact Sensor** ✅
   - 手首に取り付けた力/トルクセンサーの代わりに使用可能か？
   - **回答**: シミュレーションでは逆動力学から力/トルクを計算。`y = A @ phi_true`として測定値を合成。

2. **座標変換** ✅
   - Contact Sensor の出力座標系は？
   - tool0 フレームへの変換が必要か？
   - **回答**: `kinematics.compute_regressor()` が tool0 フレームで出力するためそのまま使用可能。

3. **サンプリングレート** ✅
   - Isaac Sim の物理シミュレーションレートは？
   - データ収集の最適な頻度は？
   - **回答**: 物理レート 240Hz、軌道サンプリング 100Hz で十分な精度を達成。

4. **Python環境衝突** ✅ (新規発見・解決)
   - Isaac Sim (Python 3.11) と ROS 2 (Python 3.12) の共存
   - **解決策**: Pinocchio を Isaac Sim Python に直接インストール、URDFキャッシュ機能を実装

---

## References

1. Kubus, D., Kroger, T., & Wahl, F. M. (2008). On-line estimation of inertial parameters using a recursive total least-squares approach. IEEE/RSJ IROS.

2. Brand, M. (2002). Incremental singular value decomposition of uncertain data with missing values. ECCV.

3. Golub, G. H., & Van Loan, C. F. (1990). Matrix Computations. Johns Hopkins University Press.
