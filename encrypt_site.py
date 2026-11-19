#!/usr/bin/env python3
"""
公開フォルダ（_site）の中身をパスワードで暗号化します。

- data.json（全銘柄の株価・財務）と cards.html（候補カード）をまとめて gzip で圧縮し、
  パスワードから作った鍵（PBKDF2-SHA256）で AES-256-GCM 暗号化して data.bin にします。
- 平文の data.json・cards.html・CSV は公開フォルダから削除します。
- パスワードは GitHub の Secrets「SITE_PASSWORD」から受け取ります（コードやログには残りません）。

data.bin の形式: b"KB1" + 反復回数(4バイト) + salt(16) + iv(12) + 暗号文
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import secrets
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITERATIONS = 310_000


def main() -> None:
    site = sys.argv[1] if len(sys.argv) > 1 else "_site"
    pw = os.environ.get("SITE_PASSWORD", "")
    if len(pw) < 4:
        sys.exit("パスワード（GitHub の Secrets「SITE_PASSWORD」）が登録されていません。"
                 "安全のため、暗号化しないままのページは公開しません。")

    with open(os.path.join(site, "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    cards_path = os.path.join(site, "cards.html")
    cards = open(cards_path, encoding="utf-8").read() if os.path.exists(cards_path) else ""

    bundle = json.dumps({"data": data, "cards": cards}, ensure_ascii=False, separators=(",", ":")).encode()
    plain = gzip.compress(bundle, 9)
    salt, iv = secrets.token_bytes(16), secrets.token_bytes(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS).derive(pw.encode())
    ct = AESGCM(key).encrypt(iv, plain, None)
    with open(os.path.join(site, "data.bin"), "wb") as f:
        f.write(b"KB1" + ITERATIONS.to_bytes(4, "big") + salt + iv + ct)

    # 平文のファイルは公開しない
    for p in [os.path.join(site, "data.json"), cards_path] + glob.glob(os.path.join(site, "*.csv")) \
             + glob.glob(os.path.join(site, "75日線ブレイク候補_*.html")):
        if os.path.exists(p):
            os.remove(p)
    print(f"暗号化しました: data.bin（{(len(ct) + 35) / 1e6:.1f}MB）")


if __name__ == "__main__":
    main()
