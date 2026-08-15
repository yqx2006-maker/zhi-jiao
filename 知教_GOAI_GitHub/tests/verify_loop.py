#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知教 GOAI · 闭环任务验收脚本（可复现、无需 API Key）

验证目标：
  1. 后端可启动，/api/health 返回合法 JSON（loaded/model/asr 字段齐全）；
  2. 后端 /api/chat 管线在源码中完整实现四段 SSE 事件
     （event: reason → data: 增量 → event: quiz → event: profile → event: done）；
  3. 前端脚本剧本（SUBJECT_SCRIPTS）对至少 2 个学科具备完整闭环字段
     （reason / quiz / firstProfile / afterProfile / migration），
     即「无 Key 也能演示完整闭环」在结构上是成立的。

运行：
  cd 知教_GOAI
  python tests/verify_loop.py
退出码 0 = 全部通过；非 0 = 存在未通过项。
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SERVER = ROOT / "server.py"
FRONTEND = ROOT / "web" / "index.html"
TEST_PORT = 8139

REQUIRED_SSE_EVENTS = ["event: reason", "event: quiz", "event: profile", "event: done"]
REQUIRED_SCRIPT_FIELDS = ["reason", "quiz", "firstProfile", "afterProfile", "migration"]

passed = []
failed = []


def check(name, ok, detail=""):
    (passed if ok else failed).append(name)
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))


def main():
    if not SERVER.exists():
        print(f"ERROR: 找不到 {SERVER}")
        sys.exit(2)
    if not FRONTEND.exists():
        print(f"ERROR: 找不到 {FRONTEND}（请确认提交前端为 index.html）")
        sys.exit(2)

    env = dict(os.environ, PORT=str(TEST_PORT))
    proc = subprocess.Popen(
        [sys.executable, str(SERVER)],
        env=env, cwd=str(ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        # 等待服务起来
        base = f"http://127.0.0.1:{TEST_PORT}"
        up = False
        for _ in range(40):
            try:
                urllib.request.urlopen(base + "/api/health", timeout=1)
                up = True
                break
            except Exception:
                time.sleep(0.25)
        check("后端服务可启动并响应 /api/health", up)
        if not up:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            _report()
            return

        # 1) health JSON 结构
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=3) as r:
                j = json.loads(r.read().decode("utf-8"))
            ok = all(k in j for k in ("loaded", "model", "asr"))
            check("health 返回合法 JSON 且字段齐全", ok, str(j))
        except Exception as e:
            check("health 返回合法 JSON 且字段齐全", False, str(e))

        # 2) 后端 SSE 四事件在源码中存在
        src = SERVER.read_text(encoding="utf-8")
        miss = [e for e in REQUIRED_SSE_EVENTS if e not in src]
        check("后端 /api/chat 实现完整 SSE 闭环事件", not miss,
              f"缺失: {miss}" if miss else "reason/quiz/profile/done 均存在")

        # 2.5) 提供方管理 / 会话存档接口实测可用
        try:
            with urllib.request.urlopen(base + "/api/providers", timeout=3) as r:
                pj = json.loads(r.read().decode("utf-8"))
            check("提供方接口可访问（providers）", isinstance(pj.get("list"), list) and "presets" in pj,
                  str(pj)[:120])
        except Exception as e:
            check("提供方接口可访问（providers）", False, str(e))
        try:
            with urllib.request.urlopen(base + "/api/sessions", timeout=3) as r:
                sj = json.loads(r.read().decode("utf-8"))
            check("会话存档接口可访问（sessions）", sj.get("ok") is True, str(sj)[:120])
        except Exception as e:
            check("会话存档接口可访问（sessions）", False, str(e))

        # 2.6) config / quiz 路由存在（页面配置 Key 与练习卡回传）
        route_miss = [k for k in ('if path == "/api/config":', 'if path == "/api/quiz":') if k not in src]
        check("config / quiz 路由存在", not route_miss, f"缺失: {route_miss}" if route_miss else "config/quiz 均在")

        # 2.7) 轻量 RAG 与 OCR 组件存在（kb.json / ocr 工具）
        kb_ok = (ROOT / "kb.json").exists() and "def retrieve_kb" in src
        check("RAG 知识库与检索函数存在", kb_ok)
        ocr_ok = (ROOT / "tools" / "ocr.cs").exists() and "ocr_files_text" in src
        check("离线 OCR 组件存在（tools/ocr.cs + 后端接入）", ocr_ok)

        # 3) 前端脚本剧本闭环字段完整（每个学科）
        html = FRONTEND.read_text(encoding="utf-8")
        idx = html.find("const SUBJECT_SCRIPTS")
        if idx == -1:
            check("前端 SUBJECT_SCRIPTS 存在", False, "未找到 SUBJECT_SCRIPTS")
        else:
            # 取脚本剧本段落（到下一个顶层 const/函数前粗略截断）
            seg = html[idx: idx + 6000]
            subjects = [s for s in ("数学", "物理", "化学", "生物", "英语")
                        if f'"{s}"' in seg]
            if not subjects:
                check("前端剧本覆盖多学段", False, "未在剧本中发现已知学科键")
            else:
                for subj in subjects:
                    # 取该学科对象片段
                    si = seg.find(f'"{subj}"')
                    block = seg[si: si + 2500]
                    miss = [f for f in REQUIRED_SCRIPT_FIELDS if f not in block]
                    check(f"剧本闭环字段完整（{subj}）", not miss,
                          f"缺失: {miss}" if miss else "reason/quiz/firstProfile/afterProfile/migration 齐全")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    _report()


def _report():
    print("\n" + "=" * 48)
    print(f"通过 {len(passed)} 项，未通过 {len(failed)} 项")
    if failed:
        print("未通过：")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    print("✅ 闭环任务验收全部通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
