"""過去10年版：採用ルールのバックテスト → 資金管理の比較（ランク6以上／ランクで絞らない）。結果は backtest10/ に保存。"""
import os
import subprocess
import sys

os.environ.update(BT_PERIOD="11y", BT_OUT="backtest10", BT_VARIANTS="trail_calm5_rsi60")
import backtest  # noqa: E402  （環境変数を読んでから読み込む）

backtest.main()
for rank, name in ((6, "rank6"), (0, "all")):
    env = dict(os.environ, P2_TRADES="backtest10/trades_trail_calm5_rsi60.csv", P2_PERIOD="11y",
               P2_MIN_RANK=str(rank), P2_SPLIT="2021-01-01", P2_OUT=f"backtest10/portfolio2_{name}.json")
    subprocess.run([sys.executable, "portfolio2.py"], env=env, check=True)
