# Isaac Sim / Lab Development Container

NVIDIA Isaac Sim 5.1.0 + Isaac Lab の開発用コンテナ環境。

## ディレクトリ構成

```
.
├── .devcontainer/
│   └── devcontainer.json      # VS Code Dev Containers 設定 (docker-compose に委譲)
├── docker/
│   ├── Dockerfile             # Isaac Sim ベースイメージ + Isaac Lab + 開発ツール
│   ├── docker-compose.yml     # コンテナオーケストレーション (単体でも使用可)
│   ├── entrypoint.sh          # ユーザー環境セットアップ
│   └── requirements.txt       # 追加 pip パッケージ (Isaac Sim Python 用)
├── .env                       # 環境変数 (UID/GID, バージョン等)
├── .dockerignore
├── setup.sh                   # 初回セットアップスクリプト
└── README.md
```

## 使い方

### 1. 初回セットアップ

```bash
chmod +x setup.sh
./setup.sh
```

これにより `.env` にホスト側の UID/GID が自動設定され、ビルドを実行できます。

### 2a. VS Code Dev Containers で開く

1. VS Code で `Dev Containers: Open Folder in Container...` を実行
2. このプロジェクトディレクトリを選択
3. 自動的にビルド＆接続されます

### 2b. Docker Compose のみで運用

```bash
# ビルド
docker compose -f docker/docker-compose.yml build

# 起動
docker compose -f docker/docker-compose.yml up -d

# コンテナに入る
docker compose -f docker/docker-compose.yml exec isaac-lab-inertia-identification zsh

# 停止
docker compose -f docker/docker-compose.yml down
```

### 3. Isaac Sim / Lab の実行

コンテナ内では以下のエイリアスが使えます：

```bash
# Isaac Sim の Python を使う
isaac-python my_script.py

# pip install する場合
isaac-pip install some-package

# Isaac Lab スクリプトの実行
isaaclab -p scripts/tutorials/00_sim/create_empty.py
```

## 追加パッケージの管理

`docker/requirements.txt` を編集し、コンテナを再ビルドしてください。
このファイルは Isaac Sim の内蔵 Python に対してインストールされます。

## 設計方針

- **devcontainer への依存最小化**: `devcontainer.json` は `docker-compose.yml` に委譲するだけ。Docker 単体で完全に動作する。
- **ホストユーザーとのパーミッション一致**: ビルド時にホスト側の UID/GID でコンテナ内ユーザーを作成。マウントされたファイルの所有権が一致する。
- **全ハードウェアアクセス**: `privileged: true` + `/dev` マウントにより、USB カメラ・シリアル・GPU すべてにアクセス可能。
- **ホスト設定の引き継ぎ**: `.ssh`, `.gitconfig` をリードオンリーでマウント。
