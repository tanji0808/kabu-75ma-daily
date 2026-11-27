# 自動売買（三菱UFJ eスマート証券・kabuステーションAPI）の始め方

毎日の判定はこれまでどおりGitHubで自動で動き、夕方に `data/auto_orders.json`（翌営業日の注文の一覧）と
確認用ページ `auto.html` を作ります。Windowsのパソコン（またはWindowsサーバー）で `kabus_bridge.py` を動かすと、
その一覧を読んで kabuステーションAPI に注文を出します。

## 1. 口座と設定（最初に1回）
1. 三菱UFJ eスマート証券の口座を開く。**信用取引口座も開く**（APIの利用条件）。特定口座（源泉徴収あり）にする。
2. Windowsのパソコン（またはWindowsサーバー）に kabuステーション を入れてログインする。
3. kabuステーションの「システム設定」→「API」タブで
   - APIのパスワードを決める
   - ソフトリミット（1回の注文の上限）を低め（例：30万円）にしておく
4. 電子契約の画面で、API の利用に同意する。

## 2. このプログラムの準備
1. Python 3.10 以上を入れる（https://www.python.org/ ）。追加のライブラリはいりません。
2. このフォルダ（bridge）をパソコンにコピーする。
3. 環境変数を設定する（Windowsの「環境変数を編集」）
   - `KABUS_API_PASSWORD` … 手順1で決めたAPIのパスワード
   - `KABUS_TRADE_PASSWORD` … 取引パスワード（verify / live のときだけ使う）
   - `KABUS_SIGN_KEY` … 注文ファイルの署名の鍵。長いでたらめな文字列（例：パスワード管理アプリで40文字以上を作る）を、
     **GitHub の Settings → Secrets and variables → Actions に `AUTO_SIGN_KEY` として同じものを登録**する。
     チャットやメールには書かない。verify / live ではこれがないと動きません。

## 3. 動かす時刻（タスク スケジューラ・平日）
| 時刻 | コマンド | すること |
|---|---|---|
| 8:30 | `python kabus_bridge.py --job morning` | 寄りの買い・リバウンドの指値・保有株の損切り（逆指値） |
| 8:52 | `python kabus_bridge.py --job gapdown` | 窓あけ下げの対象期間なら、気配を見て寄指 |
| 9:10 | `python kabus_bridge.py --job after` | 寄りで買えた株に損切りの逆指値 |
| 9:15 | `python kabus_bridge.py --job watch` | 15:20まで、利確の値段に届いたら売る |
| 14:45 | `python kabus_bridge.py --job close` | 時間で手じまう日の株を引成で売る |

kabuステーションは 8:20 までに起動・ログインしておく（自動ログインの設定を使う）。

## 4. 進め方（大事）
1. **練習（dry）**：初期値。注文は出さず、出すはずの注文を `bridge_log.txt` に書くだけ。1〜2週間、ページ（auto.html）と見比べる。
2. **検証環境（--mode verify）**：kabuステーションの検証用につないで、本物のお金を使わずに注文を試す。項目名やコードの間違いはここで直す。
3. **本番（--mode live）**：少ない金額から。最初は 1回の上限・ソフトリミットを低くしておく。

止めたい時は、このフォルダに `STOP` という名前の空のファイルを置く（または環境変数 `KABUS_STOP=1`）。
