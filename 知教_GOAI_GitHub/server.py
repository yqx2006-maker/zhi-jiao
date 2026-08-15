#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知教 GOAI 后端服务 —— 零第三方依赖（仅使用 Python 标准库）

功能：
  - 静态托管 web/index.html（浏览器访问 http://localhost:8000 即可）
  - GET  /api/health  模型配置健康检查
  - POST /api/chat    SSE 流式对话（DeepSeek 开源模型接入）
  - POST /api/reset   清空服务端会话上下文（可选 JSON body {subject} 仅清空指定学科，缺省全清）
  - POST /api/asr     离线语音转写（body 为 16k WAV，调用 Windows SAPI，零联网零密钥）

对话管线（对应前端的"教学推理链"可视化）：
  1. 诊断调用   -> event: reason   {observe, strategy}
  2. 流式回复   -> data: 增量文本
  3. 验证练习   -> event: quiz     {q, opts, ok, no, hint, reveal}
  4. 画像调用   -> event: profile  {diff, basis, strategy, next, topic, stages}
  5. event: done

运行：python server.py   （可用环境变量 PORT 指定端口，默认 8000）
"""
import json
import gzip
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

# 控制台编码兜底：避免在英文系统(cp1252)下打印中文日志时崩溃（UnicodeEncodeError）
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
PORT = int(os.environ.get("PORT", "8000"))

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
# 模型：DeepSeek V4 系列（deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用，自动迁移）
_DEPRECATED_MODELS = {"deepseek-chat": "deepseek-v4-flash", "deepseek-reasoner": "deepseek-v4-pro"}
MODEL = (os.environ.get("DEEPSEEK_MODEL") or "").strip() or "deepseek-v4-flash"
if MODEL in _DEPRECATED_MODELS:
    _new = _DEPRECATED_MODELS[MODEL]
    print("[config] 模型 %s 已停用（2026-07-24 起），自动改用 %s" % (MODEL, _new))
    MODEL = _new

MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}

# 可 gzip 压缩的文本类型（按 Content-Type 基础类型匹配）
COMPRESSIBLE = {"text/html", "text/css", "text/javascript", "application/json", "text/markdown", "image/svg+xml"}
_GZ_CACHE = {}  # 路径 -> (mtime_ns, gzip字节)：同文件避免重复压缩


# ---------------- 环境变量 ----------------
def load_env():
    """从脚本目录及上级目录读取 .env（DEEPSEEK_API_KEY），环境变量优先。"""
    for p in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
        try:
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if not v:  # 模板占位符或空值视为未配置
                    continue
                os.environ.setdefault(k.strip(), v)
        except OSError:
            continue


load_env()
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
if API_KEY and API_KEY.startswith("your_"):  # 模板占位值视为未配置
    API_KEY = ""

# ---------------- 模型提供方管理（页面可配置：预设/自定义；OpenAI 兼容或 Anthropic 兼容；落盘 providers.json） ----------------
PROVIDERS_FILE = BASE_DIR / "providers.json"
_PRESET_PROVIDERS = [
    {"id": "deepseek",  "name": "DeepSeek",          "base": "https://api.deepseek.com/v1",                       "protocol": "openai",    "model": "deepseek-v4-flash"},
    {"id": "dashscope", "name": "通义千问（阿里云）", "base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "protocol": "openai",    "model": "qwen-plus"},
    {"id": "moonshot",  "name": "Moonshot Kimi",     "base": "https://api.moonshot.cn/v1",                        "protocol": "openai",    "model": "moonshot-v1-8k"},
    {"id": "zhipu",     "name": "智谱 GLM",          "base": "https://open.bigmodel.cn/api/paas/v4",              "protocol": "openai",    "model": "glm-4-flash"},
    {"id": "ollama",    "name": "Ollama（本机）",    "base": "http://localhost:11434/v1",                         "protocol": "openai",    "model": "qwen2.5:7b", "api_key": ""},
]
_providers = {}
_active_provider_id = None


def load_providers():
    global _active_provider_id
    _providers.clear()
    try:
        data = json.loads(PROVIDERS_FILE.read_text(encoding="utf-8"))
        for p in data.get("providers", []):
            if isinstance(p, dict) and p.get("id"):
                _providers[p["id"]] = {
                    "id": p["id"], "name": p.get("name") or p["id"],
                    "base": (p.get("base") or "").strip(),
                    "protocol": (p.get("protocol") or "openai").lower(),
                    "model": p.get("model") or "",
                    "api_key": (p.get("api_key") or "").strip(),
                    "vision": bool(p.get("vision")),
                }
        _active_provider_id = data.get("active") if data.get("active") in _providers else None
    except Exception:
        pass


def save_providers():
    try:
        PROVIDERS_FILE.write_text(json.dumps({"providers": list(_providers.values()), "active": _active_provider_id},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as e:
        print("[providers] 保存失败：%s" % str(e)[:120])


def active_provider():
    p = _providers.get(_active_provider_id)
    if p:
        return p
    # 回退：未配置自建提供方时使用环境变量 DeepSeek
    return {"id": "env", "name": "DeepSeek（环境变量）", "base": "https://api.deepseek.com/v1",
            "protocol": "openai", "model": MODEL, "api_key": API_KEY}


def effective_key():
    return (active_provider().get("api_key") or "").strip()


load_providers()


# ---------------- 教学法提示词 ----------------
SYSTEM_TEACH = """你是“知教”，一名面向中国中小学生的 AI 辅导智能体。你的工作方式是一条闭环：观察提问 → 追问定位卡点 → 用图像/实例引导验证 → 依据证据更新学习画像。
教学规则（必须遵守）：
1. 不直接给出最终答案，先用一个具体问题确认学生卡在哪里。
2. 追问时给学生更多选择：给出两到三个具体的候选方向（如“A……还是 B……”或列出 2-3 个选项），学生可以直接选；最后可自然地补一句开放邀请，如“或者，你直接向我提问也行”，让学生有更多回应方式。
3. 语言自然、口语化、贴合学生年级；选项与邀请句表述顺畅、不重复、不生硬；**整体控制在 60~100 字，宁短勿长，禁止寒暄与复述学生原话**。
4. 优先用生活类比（爬山、温度、速度）和图像对比来解释抽象概念。
5. 学生答错时不否定，指出其回答中正确的部分，再引导下一步。
6. 学生表达不清时，给出两个可选说法让其确认，而不是判定“不会”。
7. 布置“试一试/小任务”时，必须当场给出具体可做的题目：真实数字、完整算式或明确条件（例如“我们做这道：−2×3+(−5)=？写出过程”或给出改数后的完整题目），让学生能立刻动笔；严禁让学生自己去课本/资料里找题。
8. 回复以引导性问题或小任务收尾，让学生能立刻接话。
9. 若在对话中直接给出 A/B/C 选项题，必须保证**有且仅有一个选项在数学上正确**，其余选项明确错误（不得出现多个选项都成立的多解题；拿不准就改成“计算唯一数值”类题）。
10. 当一个知识点的关键结论已由学生验证掌握后，用一两句话做小结（如“符号定方向、|k| 定陡缓”），并告知“学习画像已更新”，然后收尾。
11. **验证节奏要克制**：一个知识点用**一个新例子验证一次即可**，确认后立即小结推进；**严禁用多个雷同例子反复验证同一件事**（例如“k=2、k=1、k=-2 连番验证同一结论”属于过度，避免）。学生主动要求再看一例时才给第二个例子。
12. **进阶/补充知识点做成可选项**：想补充的内容（如 |k| 与倾角、负斜率方向、应用题迁移）以“要不要我再补充……？”的形式问一句，**把选择权交给学生**；学生没要求就不要自动铺开。学生答错时，及时指出并补上关键一步（见规则 5）。"""

SYSTEM_DIAG = """你是知教的诊断模块。请阅读当前对话历史与学生最新发言，输出一个 JSON 对象（不要任何其它文字、不要 markdown 代码块），格式：
{"observe":"25字内：学生的困难或理解状态（证据不足则如实说明）","strategy":"25字内：你选择的教学策略及理由"}"""

SYSTEM_PROFILE = """你是知教的学习画像模块。根据对话历史输出一个 JSON 对象（不要任何其它文字、不要 markdown 代码块），格式：
{"diff":"25字内：当前学习困难","basis":"40字内：判断依据（引用学生原话或行为证据）","strategy":"25字内：当前教学策略","next":"25字内：下一步学习任务","topic":"8字内：学习主题",
"stages":[{"t":"已互动","done":true,"now":false},{"t":"定位卡点","done":false,"now":true},{"t":"引导验证","done":false,"now":false},{"t":"巩固迁移","done":false,"now":false}]}
规则：证据不足时使用“初步观察/待验证”措辞，不得编造学生未表现出的行为；stages 的 done/now 依据对话进展如实标记；对话中出现『自主验证作答：…→ 正确』表示该验证环节已通过，stages 应相应推进（例如：已互动/定位卡点 done、引导验证或巩固迁移 now），不得停留在更早阶段。"""

SYSTEM_QUIZ = """你是知教的验证练习模块，服务于闭环中的“自主验证”环节。根据对话历史中正在教学的知识点，出一道供学生自主验证的小选择题。输出一个 JSON 对象（不要任何其它文字、不要 markdown 代码块），格式：
{"q":"题干（一行，简洁具体）","opts":[{"t":"选项文本","ok":true},{"t":"选项文本","ok":false},{"t":"选项文本","ok":false}],"ok":"答对时的鼓励与一句原理确认（40字内）","no":"答错时的温和提示（30字内）","hint":"首次答错后的针对性提示（30字内）","reveal":"两次答错后揭示正确答案的讲解（50字内）"}
规则：恰好 3 个选项且只有 1 个 ok=true；题目只验证当前正在讨论的概念，难度要低、学生跳一跳够得着；选项不得出现“以上都是/都不是”；语言贴合学生年级。
出题前必须自查唯一性：逐项自问“这一项在数学上是否同样成立/同样是合法做法”，若存在不止一个选项在数学上成立（例如移项题里多个选项都是合法的第一步），必须改写题目或选项，直到**有且仅有一个选项成立、其余选项明确错误**；拿不准就只出“计算出一个确定数值”这类唯一解题目，宁可不考“做法是否合法”。"""

QUIZ_CHECK_SYS = """你是数学选择题质检员。学生年级：{grade}。检查下面这道题，只输出一个词，不要解释：
UNIQUE_OK：有且仅有一个正确选项，且难度适合该年级（不超纲、不过于简单）
AMBIGUOUS：存在不止一个选项在数学上也成立/合法（多解）
TOO_HARD：难度超出该年级（超纲或需要更高年级知识）
TOO_EASY：过于简单，与该年级不匹配"""


def quiz_quality(qz, grade):
    """语义质检：唯一性 + 年级难度。返回 UNIQUE_OK / AMBIGUOUS / TOO_HARD / TOO_EASY；校验失败保守放行。"""
    try:
        opts = [str(o.get("t", ""))[:60] for o in qz.get("opts", [])]
        body = "题目：%s\n选项：%s" % (str(qz.get("q", ""))[:120], " / ".join(opts))
        out = (ds_json_call(QUIZ_CHECK_SYS.format(grade=grade or "初二"),
                            [{"role": "user", "content": body}], max_tokens=10) or "").strip().upper()
        for kw in ("AMBIGUOUS", "TOO_HARD", "TOO_EASY", "UNIQUE_OK"):
            if kw in out:
                return kw
        return "UNIQUE_OK"
    except Exception:
        return "UNIQUE_OK"  # 校验失败时保守放行


QUIZ_AMBIGUOUS_PS = """你是知教。刚才给学生出的选择题出现了“多个选项在数学上都成立（多解）”的情况。用一两句话（不超过60字）承认这一点，并引导学生选自己习惯的方法继续学；不要重新出题、不要逐项点评。"""


def quiz_ambiguous_ack(qz):
    """多解时的兜底引导语（模型生成；失败用模板）。"""
    try:
        opts = "；".join(str(o.get("t", ""))[:40] for o in qz.get("opts", []))
        body = "题目：%s\n选项：%s" % (str(qz.get("q", ""))[:100], opts)
        return (ds_json_call(QUIZ_AMBIGUOUS_PS, [{"role": "user", "content": body}], max_tokens=80) or "").strip()
    except Exception:
        return "（这道题其实不止一种合法解法——移项变号或两边同减都是可行的第一步。你更习惯哪一种？我们按你的方法练一道巩固题。）"

QUIZ_ACK_PS = """你是知教。学生刚完成一道自主验证选择题。用一两句话回应（不超过60字）：答对就肯定，并紧接着给出下一步的具体任务或一个追问（让学生能立刻接话、闭环继续）；答错就安抚，并说明下一步换什么角度巩固。"""

# ---------------- 未成年人内容安全与教学法防护（零依赖，精简规则） ----------------
SENSITIVE_WORDS = [
    "自杀", "自残", "轻生", "割腕", "跳楼",
    "色情", "裸照", "裸聊", "约炮", "援交",
    "赌博", "博彩", "赌场", "下注",
    "毒品", "冰毒", "海洛因", "摇头丸", "大麻",
    "枪支", "杀人", "持刀伤人", "炸学校",
]
def sensitive_hit(text):
    """基础敏感词命中检测（高危词表，仅作演示级拦截，不等同于正式内容审核服务）"""
    if not text:
        return False
    for w in SENSITIVE_WORDS:
        if w in text:
            return True
    return False

BLOCKED_MSG = "（为保护未成年人，该内容已按安全规则拦截。请换个问题，我们可以继续正常的学习辅导。）"

def looks_like_direct_answer(text):
    """启发式判断教学回复是否“直接给答案”且无引导：无问句/无任务词 + 出现答案型措辞。"""
    t = (text or "").strip()
    if not t:
        return False
    no_question = all(ch not in t for ch in ("？", "?", "试试", "告诉我", "选一个", "来说说", "做一做", "想一想", "算一算", "写一写"))
    answerish = any(k in t for k in ("答案是", "答案就是", "答案应为", "答案为", "结果等于", "结果就是", "等于", "所以是"))
    return no_question and answerish


# ---------------- 会话（按学科隔离，各学科独立上下文） ----------------
session_lock = threading.Lock()
sessions = {}  # subject -> [user/assistant 教学对话消息列表]


def reset_session(subject=None):
    with session_lock:
        if subject:
            sessions.pop(subject, None)
        else:
            sessions.clear()


# ---------------- 离线语音转写（Windows SAPI，零联网零密钥） ----------------
# 前端录音编码为 16k/16bit/单声道 WAV -> 本模块调用系统自带中文识别引擎转写。
ASR_PS1 = Path(tempfile.gettempdir()) / "zhijiao_asr.ps1"
# 说明：System.Speech 的 InstalledRecognizers 在部分 Win10/11 机器上枚举为空（内部校验失败），
# 但原生 SAPI COM 令牌可用。故用 COM 枚举令牌 + 反射构造内部 RecognizerInfo，走托管同步 Recognize()。
ASR_PS1_SRC = r'''
param([string]$wav, [string]$out)
Add-Type -AssemblyName System.Speech
$diag = "start"
$text = ""
try {
  # 优先托管引擎：InstalledRecognizers 可直接使用时最可靠（选中文化引擎）
  $diag = "managed"
  $eng = $null
  try {
    $rlist = [System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers()
    for ($i = 0; $i -lt $rlist.Count; $i++) {
      try { if ($rlist[$i].Culture.Name -eq 'zh-CN') { $eng = New-Object System.Speech.Recognition.SpeechRecognitionEngine($rlist[$i].Id); break } } catch {}
    }
  } catch {}
  if ($eng -eq $null) {
    # 托管枚举为空/不可用：回退原生 COM 令牌 + 反射构造（仅接受中文 804 令牌）
    $diag = "com"
    $cat = New-Object -ComObject SAPI.SpObjectTokenCategory
    $cat.SetId('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Recognizers')
    $tokens = $cat.EnumerateTokens()
    $comToken = $null
    for ($i = 0; $i -lt $tokens.Count; $i++) {
      $t = $tokens.Item($i)
      try { if ($t.GetAttribute('Language') -eq '804') { $comToken = $t; break } } catch {}
    }
    if ($comToken -ne $null) {
      $asm = [System.Speech.Recognition.SpeechRecognitionEngine].Assembly
      $otType = $asm.GetType('System.Speech.Internal.ObjectTokens.ObjectToken')
      $open = ($otType.GetMethods([System.Reflection.BindingFlags]'NonPublic,Public,Static') |
               Where-Object { $_.Name -eq 'Open' -and $_.GetParameters()[0].ParameterType.Name -eq 'ISpObjectToken' })[0]
      $token = $open.Invoke($null, @($comToken))
      $riCtor = [System.Speech.Recognition.RecognizerInfo].GetConstructors([System.Reflection.BindingFlags]'NonPublic,Instance')[0]
      $culture = [System.Globalization.CultureInfo]::GetCultureInfo('zh-CN')
      $ri = $riCtor.Invoke(@($token, $culture))
      $eng = New-Object System.Speech.Recognition.SpeechRecognitionEngine($ri)
    }
  }
  if ($eng -eq $null) {
    $diag = "no_zh_recognizer"
  } else {
    $diag = "engine"
    $eng.SetInputToWaveFile($wav)
    $eng.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar))
    $eng.InitialSilenceTimeout = [TimeSpan]::FromSeconds(5)
    $eng.BabbleTimeout = [TimeSpan]::FromSeconds(3)
    $eng.EndSilenceTimeout = [TimeSpan]::FromMilliseconds(500)
    $diag = "recognize"
    $res = $eng.Recognize()
    if ($res) { $text = $res.Text; $diag = "ok" } else { $diag = "no_result" }
    $eng.Dispose()
  }
} catch { $diag = "err:" + $_.Exception.Message }
[System.IO.File]::WriteAllText($out, $diag + "`n" + $text, (New-Object System.Text.UTF8Encoding($false)))
'''


def wav_diag(data):
    """解析 WAV 头与采样，返回时长/音量峰值，用于定位“没识别上”是无声还是引擎问题。"""
    try:
        if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return {"stage": "bad_wav"}
        sr, secs, peak = 0, 0.0, 0
        i = 12
        while i + 8 <= len(data):
            cid, sz = data[i:i + 4], int.from_bytes(data[i + 4:i + 8], "little")
            body = data[i + 8:i + 8 + sz]
            if cid == b"fmt " and len(body) >= 16:
                sr = int.from_bytes(body[4:6], "little")
            elif cid == b"data" and sr:
                secs = len(body) / 2.0 / sr
                for k in range(0, len(body) - 1, 2):
                    v = int.from_bytes(body[k:k + 2], "little", signed=True)
                    a = v if v >= 0 else -v
                    if a > peak:
                        peak = a
            i += 8 + sz + (sz & 1)
        return {"seconds": round(secs, 1), "peak": round(peak / 32768.0, 3), "rate": sr}
    except Exception:
        return {}


def wav_amplify(data, target=0.85, max_gain=6):
    """音频偏轻时整体放大（原地改写 data 块），提升离线引擎识别率。"""
    try:
        if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            return data
        i = 12
        while i + 8 <= len(data):
            cid, sz = data[i:i + 4], int.from_bytes(data[i + 4:i + 8], "little")
            if cid == b"data":
                body = bytearray(data[i + 8:i + 8 + sz])
                peak = 0
                for k in range(0, len(body) - 1, 2):
                    v = int.from_bytes(body[k:k + 2], "little", signed=True)
                    a = v if v >= 0 else -v
                    if a > peak:
                        peak = a
                if peak < 100:
                    return data
                gain = min(max_gain, target * 32768.0 / peak)
                if gain > 1.05:
                    for k in range(0, len(body) - 1, 2):
                        v = int.from_bytes(body[k:k + 2], "little", signed=True)
                        v = int(max(-32768, min(32767, v * gain)))
                        body[k:k + 2] = v.to_bytes(2, "little", signed=True)
                    return bytes(data[:i + 8]) + bytes(body) + bytes(data[i + 8 + sz:])
                return data
            i += 8 + sz + (sz & 1)
    except Exception:
        pass
    return data


_SAPI_PROBE = None  # None=未探测；缓存 sapi_usable() 结果
_sapi_lock = threading.Lock()


def sapi_usable():
    """探测 System.Speech 是否真的能构造中文识别引擎（结果缓存）。

    注意：必须实测“构造引擎”，不能只看 InstalledRecognizers 的枚举结果——
    部分 Win10/11 机器能枚举出中文引擎（如 MS-2052-80-DESK），但构造时抛
    “No recognizer of the required ID found”/E_ACCESSDENIED（桌面语音识别
    功能未启用或运行时受限）。health 据此如实上报，前端在 asr=false 时会
    自动降级浏览器 Web Speech，而不是把用户卡在“录音成功但识别报错”的中间态。
    """
    global _SAPI_PROBE
    if os.name != "nt":
        return False
    if _SAPI_PROBE is not None:
        return _SAPI_PROBE
    with _sapi_lock:  # 防止并发首次探测重复拉起 PowerShell
        if _SAPI_PROBE is not None:
            return _SAPI_PROBE
        ps = (r"$ErrorActionPreference='SilentlyContinue'; Add-Type -AssemblyName System.Speech; "
          r"$n=0; try { $r=[System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers(); "
          r"for($i=0;$i -lt $r.Count;$i++){ try { if($r[$i].Culture.Name -eq 'zh-CN'){ $e=New-Object System.Speech.Recognition.SpeechRecognitionEngine($r[$i].Id); $n=1; break } } catch {} } } catch {}; "
          r"if($n -eq 0){ try { $cat=New-Object -ComObject SAPI.SpObjectTokenCategory; "
          r"$cat.SetId('HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Recognizers'); "
          r"$ts=$cat.EnumerateTokens(); $tk=$null; "
          r"for($i=0;$i -lt $ts.Count;$i++){ $t=$ts.Item($i); "
          r"try { if($t.GetAttribute('Language') -eq '804'){ $tk=$t; break } } catch {} }; "
          r"if($tk -ne $null){ $asm=[System.Speech.Recognition.SpeechRecognitionEngine].Assembly; "
          r"$ot=$asm.GetType('System.Speech.Internal.ObjectTokens.ObjectToken'); "
          r"$open=($ot.GetMethods([System.Reflection.BindingFlags]'NonPublic,Public,Static') | Where-Object { $_.Name -eq 'Open' -and $_.GetParameters()[0].ParameterType.Name -eq 'ISpObjectToken' })[0]; "
          r"$tok=$open.Invoke($null,@($tk)); "
          r"$ci=[System.Speech.Recognition.RecognizerInfo].GetConstructors([System.Reflection.BindingFlags]'NonPublic,Instance')[0]; "
          r"$ri=$ci.Invoke(@($tok,[System.Globalization.CultureInfo]::GetCultureInfo('zh-CN'))); "
          r"$e=New-Object System.Speech.Recognition.SpeechRecognitionEngine($ri); $n=1 } } catch {} }; "
          r"if($n -gt 0){ 'OK' } else { 'NONE' }")
        try:
            r = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-Command", ps],
                capture_output=True, timeout=25)
            _SAPI_PROBE = (r.stdout.decode("utf-8", "ignore").strip() == "OK")
        except Exception:
            _SAPI_PROBE = False
        return _SAPI_PROBE


def sapi_ready():
    """health 专用：非阻塞——未探测完成时立即返回 False 并在后台继续探测（避免首个 health 请求被 PowerShell 拖住，
    否则 file:// 前端探测端口会超时并永久缓存“无后端”）。"""
    if os.name != "nt":
        return False
    if _SAPI_PROBE is None:
        threading.Thread(target=sapi_usable, daemon=True).start()
        return False
    return _SAPI_PROBE


def friendly_asr_stage(diag):
    """把 SAPI 诊断码映射为前端可读的稳定阶段（保留原始 diag 在 raw 字段）。"""
    d = (diag or "").strip()
    if d in ("", "ok", "no_result"):
        return d or "no_result"
    if d in ("no_token", "no_zh_recognizer", "no_output"):
        return "no_recognizer"
    if d == "timeout":
        return "timeout"
    if d == "not_windows":
        return "not_windows"
    if d.startswith("err:"):
        low = d.lower()
        if "recognizer" in low or "required id" in low or "culture" in low:
            return "no_recognizer"
        return "engine_err"
    return d


def asr_recognize(wav_bytes):
    """调用 Windows SAPI 对 WAV 做离线转写，返回 (文本, 诊断阶段)；失败/非 Windows 返回 ("", 原因)。"""
    if os.name != "nt":
        return "", "not_windows"
    try:
        ASR_PS1.write_text(ASR_PS1_SRC, encoding="utf-8")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name
        out_path = wav_path + ".txt"
        try:
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive",
                 "-ExecutionPolicy", "Bypass", "-File", str(ASR_PS1),
                 "-wav", wav_path, "-out", out_path],
                capture_output=True, timeout=45)
            if os.path.exists(out_path):
                raw = Path(out_path).read_text(encoding="utf-8-sig")
                diag, _, text = raw.partition("\n")
                return text.strip(), diag.strip()
            return "", "no_output"
        finally:
            for p in (wav_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except Exception as e:
        return "", "err:" + str(e)[:80]


# ---------------- 语音文本 LLM 纠错 ----------------
def asr_correct(raw):
    """用 LLM 修正语音转写中的同音/近音错误（如“我一角而女子一”→“一个直角三角形”）；未配置或失败时原样返回。"""
    raw = (raw or "").strip()
    if not effective_key() or len(raw) < 2:
        return raw
    try:
        out = ds_json_call(
            "你是中文语音识别纠错助手。输入是离线语音识别的原始转写，可能含同音/近音错误、乱序或无意义字词。"
            "请结合 K12 教学辅导场景（常涉及数学/语文/英语学科话题与日常口语），还原为说话人最可能想说的通顺句子。"
            "只输出纠正后的一句话，不要解释、不要引号；若原文已通顺合理则原样输出。",
            [{"role": "user", "content": raw}],
            max_tokens=150,
        )
        out = out.strip().strip('"').strip("'").strip()
        if out and len(out) <= 120:
            return out
    except Exception:
        pass
    return raw


# ---------------- WinRT DNN 实时语音识别（asr_live.exe，Windows 10/11 新一代引擎） ----------------
ASR_LIVE_EXE = BASE_DIR / "tools" / "asr_live.exe"
_live_lock = threading.Lock()
_live = {"proc": None, "out": None, "stop": None, "con": None}
_live_known_bad = False  # 引擎硬失败（spawn/engine_fail）后缓存，避免前端每次点击都重复尝试等待


def live_available():
    return os.name == "nt" and ASR_LIVE_EXE.exists() and not _live_known_bad


def live_start():
    """启动实时识别进程；进程度过启动窗口仍存活即视为会话建立成功。"""
    global _live_known_bad
    with _live_lock:
        if _live["proc"] is not None and _live["proc"].poll() is None:
            return False, "busy"
        if not live_available():
            return False, "unsupported"
        out = tempfile.mktemp(suffix=".txt", prefix="zj_live_out_")
        stop = tempfile.mktemp(suffix=".stop", prefix="zj_live_stop_")
        con = tempfile.mktemp(suffix=".log", prefix="zj_live_con_")
        try:
            con_f = open(con, "w")
            proc = subprocess.Popen([str(ASR_LIVE_EXE), out, stop],
                                    stdout=con_f, stderr=subprocess.STDOUT,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            con_f.close()
        except Exception as e:
            _live_known_bad = True
            return False, "spawn:" + str(e)[:60]
        t0 = time.time()
        ok = False
        while time.time() - t0 < 5:
            if proc.poll() is not None:
                break
            if os.path.exists(out) and time.time() - t0 > 2.5:
                ok = True
                break
            time.sleep(0.2)
        if not ok:
            _live_known_bad = True
            try:
                proc.kill()
            except Exception:
                pass
            for p in (out, stop, con):
                try:
                    os.remove(p)
                except OSError:
                    pass
            return False, "engine_fail"
        _live.update(proc=proc, out=out, stop=stop, con=con)
        return True, ""


def live_stop():
    """通知识别进程收尾并返回拼接后的转写文本。"""
    with _live_lock:
        proc, out, stop, con = _live["proc"], _live["out"], _live["stop"], _live["con"]
        _live.update(proc=None, out=None, stop=None, con=None)
    if proc is None:
        return ""
    try:
        with open(stop, "a"):
            pass
    except OSError:
        pass
    try:
        proc.wait(timeout=6)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    text = ""
    try:
        raw = Path(out).read_text(encoding="utf-8-sig", errors="ignore")
        text = "".join(ln.strip() for ln in raw.splitlines() if ln.strip())
    except OSError:
        pass
    for p in (out, stop, con):
        try:
            if p:
                os.remove(p)
        except OSError:
            pass
    return text


# ---------------- DeepSeek / 模型提供方调用（OpenAI 兼容 + Anthropic 兼容） ----------------
def _resp_text(body, protocol):
    """从非流式响应中提取文本（按协议）。"""
    if protocol == "anthropic":
        try:
            return "".join(b.get("text", "") for b in (body.get("content") or []) if b.get("type") == "text")
        except Exception:
            return ""
    return (body.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def _delta_text(data, protocol):
    """从流式一行 data 中提取文本增量（按协议）。"""
    try:
        obj = json.loads(data)
        if protocol == "anthropic":
            d = obj.get("delta") or {}
            if obj.get("type") == "content_block_delta" and d.get("type") == "text_delta":
                return d.get("text") or ""
            return ""
        return (obj.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
    except Exception:
        return ""


def ds_request(messages, stream=False, max_tokens=600, temperature=0.4, timeout=90, provider_id=None):
    p = _providers.get(provider_id) if provider_id else active_provider()
    if not p:
        p = active_provider()
    key = (p.get("api_key") or "").strip()
    model = (p.get("model") or "").strip() or MODEL
    base = (p.get("base") or "https://api.deepseek.com/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        base = base.rsplit("/chat/completions", 1)[0]
    if base.endswith("/messages"):
        base = base.rsplit("/messages", 1)[0]
    protocol = (p.get("protocol") or "openai").lower()
    if protocol == "anthropic":
        urls = [base + "/v1/messages", base + "/messages"]
        sys_txt = "\n\n".join(m["content"] for m in messages if m.get("role") == "system" and m.get("content"))
        conv = [{"role": m["role"], "content": m["content"]}
                for m in messages if m.get("role") in ("user", "assistant")]
        payload = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": temperature,
                              "system": sys_txt or None, "messages": conv, "stream": stream}).encode("utf-8")
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
    else:  # OpenAI 兼容
        urls = [base + "/chat/completions", base + "/v1/chat/completions"]
        payload = json.dumps({"model": model, "messages": messages, "stream": stream,
                              "max_tokens": max_tokens, "temperature": temperature}).encode("utf-8")
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    urls = list(dict.fromkeys(urls))  # 去重（用户已带 /v1 时两个候选相同）
    last_404 = None
    for url in urls:
        req = urllib.request.Request(url, data=payload, headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 404 and len(urls) > 1:
                last_404 = e
                continue  # 端点 404：尝试另一个候选（常见为 base 缺/多 /v1）
            raise
    raise last_404 if last_404 else urllib.error.URLError("no endpoint")


def ds_json_call(system, messages, max_tokens=400):
    """非流式调用，返回纯文本；429 限流自动退避重试一次。"""
    resp = None
    protocol = (active_provider().get("protocol") or "openai").lower()
    for _attempt in range(2):
        try:
            resp = ds_request([{"role": "system", "content": system}] + messages,
                              stream=False, max_tokens=max_tokens, timeout=60)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 and _attempt == 0:
                print("[api] 429 限流，退避 2.5s 后重试")
                time.sleep(2.5)
                continue
            raise
    body = json.loads(resp.read().decode("utf-8"))
    return _resp_text(body, protocol)


def extract_json(text):
    """宽容解析模型输出中的 JSON：去掉 markdown 围栏，截取首个 {...}。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------- 服务端会话存档（跨浏览器/机器恢复；仅存本机文件） ----------------
SESSIONS_FILE = BASE_DIR / "sessions.json"


def load_sessions_file():
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8") or "{}")
    except Exception:
        return {}


def save_sessions_file(data):
    try:
        if isinstance(data, dict):
            # 容量控制：单会话消息最多 80 条，总存档最多 40 个会话（按 ts 保留最新）
            items = [v for k, v in data.items() if k != "__meta" and isinstance(v, dict)]
            items.sort(key=lambda v: v.get("ts") or "")
            for it in items:
                msgs = it.get("msgs")
                if isinstance(msgs, list) and len(msgs) > 80:
                    it["msgs"] = msgs[-80:]
            if len(items) > 40:
                keep = {id(v) for v in items[-40:]}
                for k in [k for k, v in list(data.items()) if k != "__meta" and isinstance(v, dict) and id(v) not in keep]:
                    data.pop(k, None)
        SESSIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError as e:
        print("[sessions] 保存失败：%s" % str(e)[:120])


# ---------------- 轻量 RAG：学科知识卡检索（零依赖，关键词命中；kb.json 可扩展） ----------------
KB_FILE = BASE_DIR / "kb.json"
_KB_CARDS = []


def load_kb():
    global _KB_CARDS
    try:
        _KB_CARDS = (json.loads(KB_FILE.read_text(encoding="utf-8")) or {}).get("cards", [])
    except Exception:
        _KB_CARDS = []


def retrieve_kb(text, subject, k=2):
    """按关键词命中（学科加权）返回知识卡内容摘要；无命中返回 ''。"""
    if not _KB_CARDS or not text:
        return ""
    scored = []
    for c in _KB_CARDS:
        keys = [x for x in (c.get("keywords") or []) if x and x in text]
        if not keys:
            continue
        score = len(keys) * 2 + (2 if c.get("subject") == subject else 0)
        scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    picks = scored[:k]
    if not picks:
        return ""
    return "学科知识参考（供引用，勿照抄）：" + "；".join(c[1].get("content", "")[:120] for c in picks)


load_kb()


# ---------------- 学习进展：基于对话证据的确定性阶段推进 ----------------
def prog_stages(subject):
    """按证据推进阶段，避免“画像一直卡在定位卡点”的名不副实。
    证据：用户发言轮数（turns）、自主验证作答正确次数（correct）。"""
    msgs = sessions.get(subject, [])
    turns = sum(1 for m in msgs if m.get("role") == "user")
    correct = 0
    for m in msgs:
        content = m.get("content") or ""
        if "自主验证作答" in content and "正确" in content:
            correct += 1
    done = [
        turns >= 1,            # 已互动
        turns >= 2,            # 定位卡点（已至少追问一轮）
        turns >= 4 and correct >= 1,  # 引导验证（完成过验证且答对过）
        turns >= 6 and correct >= 2,  # 巩固迁移
    ]
    now_idx = next((i for i, d in enumerate(done) if not d), 3)
    return [{"t": t, "done": done[i], "now": i == now_idx}
            for i, t in enumerate(["已互动", "定位卡点", "引导验证", "巩固迁移"])]


def merge_profile_stages(prof, subject):
    """合并阶段：取“模型输出”与“证据规则”中推进更远的一组（只前进，不后退）。"""
    mine = prog_stages(subject)
    model = prof.get("stages")
    mine_done = sum(1 for s in mine if s.get("done"))
    model_done = sum(1 for s in (model or []) if s.get("done")) if model else 0
    if not model or model_done < mine_done:
        prof["stages"] = mine
    return prof


# ---------------- 图片离线 OCR（Windows.Media.Ocr，零依赖；失败自动降级为口述题意） ----------------
OCR_EXE = BASE_DIR / "tools" / "ocr.exe"


def ocr_image_text(img_path, out_path):
    """调用 ocr.exe 提取图片文字；返回文本或 ''。"""
    try:
        r = subprocess.run([str(OCR_EXE), img_path, out_path], capture_output=True, timeout=45)
        if r.returncode != 0:
            return ""
        return Path(out_path).read_text(encoding="utf-8-sig", errors="ignore").strip()
    except Exception as e:
        print("[ocr] 工具异常：%s" % str(e)[:120])
        return ""


def ocr_files_text(files):
    """对图片附件做离线 OCR，返回拼装文本；无可用结果返回 ''。"""
    if not OCR_EXE.exists():
        return ""
    parts = []
    for f in files:
        data = f.get("data")
        name = f.get("name") or ""
        if not data or not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
            continue
        tmp_path, out_path = "", ""
        try:
            with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            out_path = tmp_path + ".txt"
            txt = ocr_image_text(tmp_path, out_path)
            if txt:
                parts.append("图片《%s》OCR 提取：\n%s" % (name, txt[:600]))
        except Exception as e:
            print("[ocr] 处理失败：%s" % str(e)[:120])
        finally:
            for p in (tmp_path, out_path):
                try:
                    if p:
                        os.remove(p)
                except OSError:
                    pass
    if not parts:
        return ""
    return ("（以下为题目图片的离线 OCR 提取结果，可能不准确，请结合学生口述理解题意：\n"
            + "\n---\n".join(parts) + "\n）")


# ---------------- multipart/form-data 解析（标准库实现） ----------------
def parse_multipart(body, content_type):
    fields, files = {}, []
    m = re.search(r'boundary=("?)([^";]+)\1', content_type or "")
    if not m:
        return fields, files
    boundary = ("--" + m.group(2)).encode()
    for part in body.split(boundary):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        head, content = part.split(b"\r\n\r\n", 1)
        head_s = head.decode("utf-8", "ignore")
        name_m = re.search(r'name="([^"]*)"', head_s)
        file_m = re.search(r'filename="([^"]*)"', head_s)
        if not name_m:
            continue
        if file_m:
            if file_m.group(1):
                # 保留字节：图片可用于离线 OCR（题目文字提取）
                files.append({"name": file_m.group(1), "size": len(content), "data": content})
        else:
            fields.setdefault(name_m.group(1), content.decode("utf-8", "ignore"))
    return fields, files


# ---------------- HTTP 处理 ----------------
class Handler(BaseHTTPRequestHandler):
    server_version = "ZhiJiaoGOAI/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("[知教] %s\n" % (fmt % args))

    # ---- 通用响应 ----
    def send_json(self, obj, status=200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")  # 允许 file:// 直开页面调用
        self.end_headers()
        self.wfile.write(data)

    def sse_send(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    # ---- GET ----
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/health":
            self.send_json({
                "loaded": bool(effective_key()),
                "model": (active_provider().get("model") or MODEL),
                "provider": active_provider().get("name") or "未配置",
                "backend": "python-stdlib",
                "key_source": "provider" if _active_provider_id else ("env" if API_KEY else "none"),
                "asr": sapi_ready(),
                "asr_live": live_available(),
                "ocr": bool(OCR_EXE.exists()),
                "message": "模型提供方：%s" % (active_provider().get("name")
                                             or "未配置（可在页面「配置模型」添加，或编辑 .env 后重启）"),
            })
            return
        if path == "/api/providers":
            self.send_json({
                "list": [{"id": p["id"], "name": p["name"], "base": p["base"], "protocol": p["protocol"],
                          "model": p["model"], "key_set": bool(p.get("api_key")), "vision": bool(p.get("vision"))}
                         for p in _providers.values()],
                "active": _active_provider_id,
                "presets": _PRESET_PROVIDERS,
            })
            return
        if path == "/api/sessions":
            # 服务端会话存档（跨浏览器/机器恢复）
            self.send_json({"ok": True, "data": load_sessions_file()})
            return
        self.serve_static(path)

    # ---- OPTIONS（CORS 预检：file:// 直开页面调用 /api/asr 等需要） ----
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    # ---- POST ----
    def do_POST(self):
        global _active_provider_id
        path = self.path.split("?")[0]
        if path == "/api/reset":
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                subj = (json.loads(body.decode("utf-8") or "{}") or {}).get("subject")
            except (json.JSONDecodeError, UnicodeDecodeError):
                subj = None
            reset_session(subj)
            self.send_json({"ok": True, "subject": subj or "all"})
            return
        if path == "/api/asr":
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > 10 * 1024 * 1024:  # 防超大上传拖垮标准库服务器
                self.send_json({"ok": False, "error": "音频过大（上限 10MB）"}, 413)
                return
            wav = self.rfile.read(length) if length else b""
            if not wav.startswith(b"RIFF"):
                self.send_json({"ok": False, "error": "音频格式无效（需 WAV）"})
                return
            wav = wav_amplify(wav)  # 偏轻录音整体放大，提升识别率
            try:  # 留存最后一次录音，便于离线分析
                (Path(tempfile.gettempdir()) / "zhijiao_asr_last.wav").write_bytes(wav)
            except OSError:
                pass
            text, stage = asr_recognize(wav)
            if text:
                text = asr_correct(text)
            resp = {"ok": bool(text), "text": text}
            if not text:
                resp["debug"] = dict(wav_diag(wav), stage=friendly_asr_stage(stage), raw=stage)
                print("[asr] 未识别出文本，诊断：", resp["debug"])
            self.send_json(resp)
            return
        if path == "/api/asr_live/start":
            ok, err = live_start()
            self.send_json({"ok": ok, "error": err})
            return
        if path == "/api/asr_live/stop":
            text = asr_correct(live_stop())
            self.send_json({"ok": bool(text), "text": text})
            return
        if path == "/api/config":
            # 兼容旧「配置模型 Key」：把 Key 写入内置 DeepSeek 提供方并设为当前
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            key = (data.get("api_key") or "").strip()
            if key and not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                # 非法字符（粘贴时混入空格/中文等）会在 Authorization 头 latin-1 编码时报错，这里直接拦截并提示
                self.send_json({
                    "ok": False,
                    "loaded": bool(effective_key()),
                    "key_source": "provider" if _active_provider_id else ("env" if API_KEY else "none"),
                    "msg": "Key 包含非法字符（粘贴时可能混入了空格或中文）：请只复制密钥本身（仅字母/数字/-/_）后重试。",
                })
                return
            preset = next((x for x in _PRESET_PROVIDERS if x["id"] == "deepseek"),
                          {"id": "deepseek", "name": "DeepSeek", "base": "https://api.deepseek.com/v1",
                           "protocol": "openai", "model": MODEL})
            p = dict(_providers.get("deepseek") or preset)
            p["api_key"] = key
            _providers["deepseek"] = p
            _active_provider_id = "deepseek"
            save_providers()
            self.send_json({
                "ok": True,
                "loaded": bool(effective_key()),
                "key_source": "provider",
                "msg": "已启用页面配置的 Key" if key else "已清除 Key（未配置时将回退剧本演示模式）",
            })
            return
        if path == "/api/providers":
            # 新增/更新提供方
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            pr = data.get("provider") or {}
            pid = (pr.get("id") or "").strip()
            base = (pr.get("base") or "").strip().rstrip("/")
            if not pid or not base:
                self.send_json({"ok": False, "msg": "缺少 Provider ID 或 API 地址"})
                return
            _active_provider_id = pid  # 「添加并使用」：添加后立即切换为该提供方
            _providers[pid] = {
                "id": pid,
                "name": (pr.get("name") or pid).strip(),
                "base": base,
                "protocol": (pr.get("protocol") or "openai").lower(),
                "model": (pr.get("model") or "").strip(),
                "api_key": (pr.get("api_key") or "").strip(),
                "vision": bool(pr.get("vision")),
            }
            save_providers()
            self.send_json({"ok": True, "active": _active_provider_id})
            return
        if path == "/api/providers/select":
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            pid = (data.get("id") or "").strip()
            if pid not in _providers:
                self.send_json({"ok": False, "msg": "提供方不存在"})
                return
            _active_provider_id = pid
            save_providers()
            self.send_json({"ok": True, "active": pid})
            return
        if path == "/api/providers/remove":
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            pid = (data.get("id") or "").strip()
            _providers.pop(pid, None)
            if _active_provider_id == pid:
                _active_provider_id = next(iter(_providers), None)
            save_providers()
            self.send_json({"ok": True, "active": _active_provider_id})
            return
        if path == "/api/providers/test":
            # 连接测试：用最小请求探测该提供方，返回精确错误（不改变当前提供方）
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            pid = (data.get("id") or "").strip()
            if pid not in _providers:
                self.send_json({"ok": False, "error": "提供方不存在"})
                return
            p = _providers[pid]
            url_hint = (p.get("base") or "").rstrip("/") + ("/v1/messages" if p.get("protocol") == "anthropic" else "/chat/completions")
            try:
                resp = ds_request([{"role": "user", "content": "你好"}], stream=False,
                                  max_tokens=1, timeout=20, provider_id=pid)
                body = json.loads(resp.read().decode("utf-8"))
                txt = _resp_text(body, (p.get("protocol") or "openai").lower())
                self.send_json({"ok": True, "text": (txt or "")[:50],
                                "url": url_hint, "model": p.get("model") or MODEL})
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8", "ignore")[:200]
                except Exception:
                    detail = ""
                self.send_json({"ok": False, "error": "HTTP %s" % e.code, "detail": detail, "url": url_hint})
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)[:200], "url": url_hint})
            return
        if path == "/api/sessions":
            # 服务端会话存档（跨浏览器/机器恢复）
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            save_sessions_file(data.get("data") or {})
            self.send_json({"ok": True})
            return
        if path == "/api/quiz":
            # 真实模式练习卡作答回传：把验证结果写入会话上下文并按证据更新画像，闭环不中断
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                data = json.loads(body.decode("utf-8") or "{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = {}
            subj = (data.get("subject") or "数学").strip() or "数学"
            grade = (data.get("grade") or "初二").strip() or "初二"
            q = (data.get("q") or "").strip()[:120]
            chosen = (data.get("chosen") or "").strip()[:120]
            correct = bool(data.get("correct"))
            ev = "（自主验证作答：题目「%s」→ 选「%s」→ %s）" % (q, chosen, "正确" if correct else "错误")
            ack, prof = "", None
            with session_lock:
                hist = sessions.setdefault(subj, [])
                hist.append({"role": "user", "content": ev})
            if effective_key():
                try:
                    ack = (ds_json_call(QUIZ_ACK_PS, hist, max_tokens=120) or "").strip()
                    if sensitive_hit(ack):
                        ack = ""
                    if ack:
                        with session_lock:
                            sessions.setdefault(subj, []).append({"role": "assistant", "content": ack})
                except Exception:
                    ack = ""
                try:
                    prof = extract_json(ds_json_call(SYSTEM_PROFILE, sessions.setdefault(subj, []), max_tokens=350))
                except Exception:
                    prof = None
            if not prof or "diff" not in prof:
                prof = {"diff": "验证作答：%s" % ("初步掌握" if correct else "待巩固"),
                        "basis": ev[:80], "strategy": "继续按闭环推进。",
                        "next": "完成下一个引导任务。", "topic": "%s · 进行中" % subj}
            merge_profile_stages(prof, subj)  # 学习进展按证据推进
            resp = {"ok": True,
                    "text": ack or ("答对了，已记录你的验证结果，我们继续。" if correct else "没关系，已记录。下一步换个角度帮你巩固。"),
                    "profile": prof}
            # 答对后自动生成“下一步”练习卡，让闭环真正延续（如：斜率方向→倾斜程度）；失败自动重试一次
            if correct and effective_key():
                for _attempt in range(2):
                    try:
                        q2 = extract_json(ds_json_call(SYSTEM_QUIZ, sessions.setdefault(subj, []), max_tokens=350))
                        if q2 and isinstance(q2.get("q"), str) and isinstance(q2.get("opts"), list):
                            opts2 = [o for o in q2["opts"] if isinstance(o, dict) and isinstance(o.get("t"), str)][:4]
                            if len(opts2) >= 2 and sum(1 for o in opts2 if o.get("ok")) == 1 \
                                    and not sensitive_hit(q2["q"]) and not any(sensitive_hit(o.get("t") or "") for o in opts2):
                                nq = {"q": q2["q"], "opts": opts2,
                                      "ok": q2.get("ok") or "答对了！",
                                      "no": q2.get("no") or "再想想。",
                                      "hint": q2.get("hint") or "",
                                      "reveal": q2.get("reveal") or ""}
                                if quiz_quality(nq, grade) == "UNIQUE_OK":  # 语义+年级质检：不合格则放弃下发
                                    resp["next_quiz"] = nq
                                else:
                                    print("[quiz] 下一步练习卡质检未通过，放弃")
                                break
                    except Exception:
                        pass
            self.send_json(resp)
            return
        if path == "/api/chat":
            self.handle_chat()
            return
        self.send_json({"error": "not found"}, 404)

    # ---- 对话主流程 ----
    def handle_chat(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > 20 * 1024 * 1024:  # 防超大附件拖垮标准库服务器（前端已限 8MB/文件，此为底线）
            self.send_json({"error": "请求体过大（上限 20MB）"}, 413)
            return
        body = self.rfile.read(length) if length else b""
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" in ctype:
            fields, files = parse_multipart(body, ctype)
        else:
            try:
                fields = json.loads(body.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                fields = {}
            files = []

        text = (fields.get("text") or "").strip()[:3000]  # 单条输入上限：防超长文本拖垮上下文/接口
        grade = fields.get("grade") or "初二"
        subject = fields.get("subject") or "数学"
        profile = (fields.get("profile") or "").strip()
        cross = (fields.get("cross") or "") == "1"  # 跨学科联合提问开关（前端左栏切换）

        # SSE 响应头
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")  # 允许 file:// 直开页面调用
        self.end_headers()

        if not effective_key():
            self.sse_send("event: error\ndata: 未配置 DeepSeek API Key：可在页面「配置模型 Key」填入，或编辑 .env 后重启服务\n\n")
            self.sse_send("event: done\n\n")
            return

        # 未成年人安全：输入命中敏感词直接拦截，不进入模型
        if sensitive_hit(text):
            self.sse_send("event: error\ndata: 该提问可能涉及敏感内容，为保护未成年人已拦截，请换个问题。\n\n")
            self.sse_send("event: done\n\n")
            return

        # 组装学生输入：图片优先离线 OCR 提取文字；无 OCR 或失败则如实请学生口述题意
        user_content = text if text else "[学生上传了题目材料但未附带文字]"
        vision_content = None
        if files:
            names = "、".join(f["name"] for f in files[:3])
            ocr_note = ocr_files_text(files) if any(f.get("data") for f in files) else ""
            if ocr_note:
                user_content += "\n" + ocr_note
            else:
                user_content += "\n（学生同时上传了 %d 份材料：%s。请先请学生口述题意与已知条件。）" % (len(files), names)
            # 多模态：若当前提供方支持图片输入，把图片以 base64 随消息发送（OpenAI 兼容 content 数组）
            if active_provider().get("vision"):
                import base64 as _b64
                img_parts = []
                for f in files:
                    data = f.get("data")
                    if not data:
                        continue
                    nm = (f.get("name") or "").lower()
                    if not nm.endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        continue
                    mime = "image/png" if nm.endswith(".png") else "image/jpeg"
                    img_parts.append({"type": "image_url",
                                      "image_url": {"url": "data:%s;base64,%s" % (mime, _b64.b64encode(data).decode("ascii"))}})
                if img_parts:
                    vision_content = [{"type": "text", "text": user_content}] + img_parts

        ctx_note = "学生学段年级：%s；当前学科：%s。" % (grade, subject)
        if profile:
            ctx_note += "学生自述学习背景：%s。" % profile
        if cross:
            ctx_note += "本次为跨学科联合提问：以当前学科为主线，自然带入至少一个其他学科的视角做类比或联系。"
        # 轻量 RAG：注入学科知识参考（关键词命中，零依赖）
        kb_note = retrieve_kb(text, subject)
        if kb_note:
            ctx_note += "\n" + kb_note

        try:
            with session_lock:
                history = list(sessions.setdefault(subject, []))

            # ---- 第 1 步：诊断（教学推理链可视化） ----
            diag_user_msg = {"role": "user",
                             "content": vision_content or (ctx_note + "\n学生说：" + user_content)}
            diag_text = ds_json_call(
                SYSTEM_DIAG,
                history + [diag_user_msg],
                max_tokens=200,
            )
            reason = extract_json(diag_text) or {
                "observe": "正在收集更多信息以定位卡点。",
                "strategy": "先通过追问确认学生的理解起点。",
            }
            self.sse_send("event: reason\ndata: %s\n\n" %
                          json.dumps(reason, ensure_ascii=False, separators=(",", ":")))

            # ---- 第 2 步：流式教学回复 ----
            teach_messages = (
                [{"role": "system", "content": SYSTEM_TEACH},
                 {"role": "system", "content": ctx_note + "\n诊断结果：%s\n请据此生成对学生的回复。"
                  % json.dumps(reason, ensure_ascii=False)}]
                + history
                + [{"role": "user", "content": vision_content or user_content}]
            )
            try:
                resp = ds_request(teach_messages, stream=True, max_tokens=800)
            except urllib.error.HTTPError as _e429:
                if _e429.code == 429:  # 限流：退避一次再试
                    print("[chat] 429 限流，退避 2.5s 后重试")
                    time.sleep(2.5)
                    resp = ds_request(teach_messages, stream=True, max_tokens=800)
                else:
                    raise
            _proto = (active_provider().get("protocol") or "openai").lower()
            full_reply = ""
            sse_buf = ""  # 聚合小分片再 flush，前端渲染更平滑
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]" or (_proto == "anthropic" and "message_stop" in data):
                    break
                delta = _delta_text(data, _proto)
                if delta:
                    full_reply += delta
                    sse_buf += delta.replace("\n", "\\n")
                    if len(sse_buf) >= 10:
                        self.sse_send("data: %s\n\n" % sse_buf)
                        sse_buf = ""
            if sse_buf:
                self.sse_send("data: %s\n\n" % sse_buf)
            # 教学法防护：疑似“直接给答案且无引导”时，追加一句引导追问
            if full_reply and looks_like_direct_answer(full_reply):
                print("[chat] 教学回复疑似直答，追加引导追问")
                guide = "\n\n先别急着看答案——你来说说：解这道题，第一步该想什么？"
                full_reply += guide
                self.sse_send("data: %s\n\n" % guide.replace("\n", "\\n"))
            if not full_reply:
                # 流式输出为空（思考模式可能吞掉输出/偶发空响应）：非流式重试一次，避免“未返回内容”死路
                print("[chat] 教学流式返回为空，尝试非流式重试一次")
                try:
                    resp2 = ds_request(teach_messages, stream=False, max_tokens=800, timeout=60)
                    body2 = json.loads(resp2.read().decode("utf-8"))
                    full_reply = _resp_text(body2, _proto)
                    if full_reply:
                        self.sse_send("data: %s\n\n" % full_reply.replace("\n", "\\n"))
                        print("[chat] 非流式重试成功")
                except Exception as e2:
                    print("[chat] 非流式重试失败：%s" % str(e2)[:150])
                    full_reply = ""
            # 未成年人安全：回复命中敏感词 → 前端以安全提示替换气泡，不入历史、不出题
            if full_reply and sensitive_hit(full_reply):
                print("[chat] 教学回复命中敏感词，已拦截")
                self.sse_send("event: blocked\ndata: %s\n\n" % BLOCKED_MSG)
                self.sse_send("event: done\n\n")
                return
            if not full_reply:
                self.sse_send("event: error\ndata: 模型未返回内容\n\n")
                self.sse_send("event: done\n\n")
                return

            with session_lock:
                msgs = sessions.setdefault(subject, [])
                msgs.append({"role": "user", "content": user_content})
                msgs.append({"role": "assistant", "content": full_reply})
                # 控制上下文长度：保留最近 12 轮
                if len(msgs) > 24:
                    del msgs[:-24]
                history_full = list(msgs)

            # ---- 第 2.5 步：自主验证练习卡（前端以 event: quiz 渲染，闭环「自主验证」环节）----
            try:
                quiz_text = ds_json_call(
                    SYSTEM_QUIZ,
                    teach_messages + [{"role": "assistant", "content": full_reply}],
                    max_tokens=350,
                )
                qz = extract_json(quiz_text)
                quiz = None
                if qz and isinstance(qz.get("q"), str) and isinstance(qz.get("opts"), list):
                    opts = [o for o in qz["opts"]
                            if isinstance(o, dict) and isinstance(o.get("t"), str)][:4]
                    # 严格校验：至少 2 个选项且恰好 1 个正确，否则不出题，绝不出错题；命中敏感词也不出题
                    if len(opts) >= 2 and sum(1 for o in opts if o.get("ok")) == 1 \
                            and not sensitive_hit(qz["q"]) and not any(sensitive_hit(o["t"]) for o in opts):
                        quiz = {"q": qz["q"], "opts": opts,
                                "ok": qz.get("ok") or "答对了！",
                                "no": qz.get("no") or "再想想。",
                                "hint": qz.get("hint") or "",
                                "reveal": qz.get("reveal") or ""}
                if quiz:
                    v = quiz_quality(quiz, grade)
                    if v != "UNIQUE_OK":
                        print("[chat] 练习卡质检未通过（%s），尝试重出一次" % v)
                        if v == "AMBIGUOUS":
                            ackm = quiz_ambiguous_ack(quiz)
                            if ackm:
                                self.sse_send("data: %s\n\n" % ackm.replace("\n", "\\n"))
                        quiz = None
                        try:
                            # 带“年级难度 + 唯一性”约束重出一次
                            extra = [{"role": "system", "content":
                                      "注意：上一道题质检未通过（难度与%s不匹配或存在多个正确选项）。请重新出一道：难度严格匹配%s（不超纲、不过于简单），有且仅有唯一一个正确选项，其余选项必须明确错误。" % (grade, grade)}]
                            qz2 = extract_json(ds_json_call(SYSTEM_QUIZ, teach_messages + [{"role": "assistant", "content": full_reply}] + extra, max_tokens=350))
                        except Exception:
                            qz2 = None
                        if qz2 and isinstance(qz2.get("q"), str) and isinstance(qz2.get("opts"), list):
                            opts2b = [o for o in qz2["opts"] if isinstance(o, dict) and isinstance(o.get("t"), str)][:4]
                            if len(opts2b) >= 2 and sum(1 for o in opts2b if o.get("ok")) == 1 \
                                    and not sensitive_hit(qz2["q"]) and not any(sensitive_hit(o.get("t") or "") for o in opts2b):
                                quiz = {"q": qz2["q"], "opts": opts2b,
                                        "ok": qz2.get("ok") or "答对了！",
                                        "no": qz2.get("no") or "再想想。",
                                        "hint": qz2.get("hint") or "",
                                        "reveal": qz2.get("reveal") or ""}
                                if quiz_quality(quiz, grade) != "UNIQUE_OK":
                                    print("[chat] 重出练习卡仍未通过质检，放弃下发")
                                    quiz = None
                if quiz:
                    self.sse_send("event: quiz\ndata: %s\n\n" %
                                  json.dumps(quiz, ensure_ascii=False, separators=(",", ":")))
            except Exception:
                pass  # 练习卡生成失败不影响主回复与画像流程

            # ---- 第 3 步：学习画像更新 ----
            prof_text = ds_json_call(SYSTEM_PROFILE, history_full, max_tokens=350)
            prof = extract_json(prof_text)
            if prof and "diff" in prof and sensitive_hit((prof.get("diff") or "") + (prof.get("basis") or "")
                                                         + (prof.get("strategy") or "") + (prof.get("next") or "")
                                                         + (prof.get("topic") or "")):
                prof = None  # 画像字段命中敏感词：按默认画像兜底，不渲染风险内容
            if not prof or "diff" not in prof:
                prof = {
                    "diff": "正在互动中，卡点待进一步确认。",
                    "basis": "学生说：" + (text or "[图片/材料]") + "；知教已回复并引导下一步。",
                    "strategy": reason.get("strategy", "追问定位卡点。"),
                    "next": "完成知教提出的引导问题或小任务。",
                    "topic": "%s · 进行中" % subject,
                    "stages": [
                        {"t": "已互动", "done": True, "now": False},
                        {"t": "定位卡点", "done": False, "now": True},
                        {"t": "引导验证", "done": False, "now": False},
                        {"t": "巩固迁移", "done": False, "now": False},
                    ],
                }
            # 规范化画像字段：模型输出不稳定时保证前端渲染不崩
            for k in ("diff", "basis", "strategy", "next", "topic"):
                if not isinstance(prof.get(k), str) or not prof.get(k):
                    prof[k] = "（待补充）"
            if not isinstance(prof.get("stages"), list) or not prof["stages"]:
                prof["stages"] = [
                    {"t": "已互动", "done": True, "now": False},
                    {"t": "定位卡点", "done": False, "now": True},
                    {"t": "引导验证", "done": False, "now": False},
                    {"t": "巩固迁移", "done": False, "now": False},
                ]
            merge_profile_stages(prof, subject)  # 学习进展按证据确定性推进（只前进不后退）
            self.sse_send("event: profile\ndata: %s\n\n" %
                          json.dumps(prof, ensure_ascii=False, separators=(",", ":")))
            self.sse_send("event: done\n\n")

        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "ignore")[:200]
            except Exception:
                detail = str(e)
            hint = {401: "（Key 无效或已过期：请在页面「配置模型」重新填入有效 Key）",
                    402: "（账户余额不足：请到对应平台充值）",
                    429: "（请求过于频繁，请稍等片刻再试）",
                    400: "（请求参数异常，常见于模型名不对：请在「配置模型」里检查模型名，或先用「测试连接」定位）"}.get(e.code, "")
            pname = active_provider().get("name") or "模型"
            msg = "模型请求失败 HTTP %s（提供方：%s）：%s%s" % (e.code, pname, detail, hint)
            print("[chat] " + msg)
            self.sse_send("event: error\ndata: %s\n\n" % msg)
            self.sse_send("event: done\n\n")
        except urllib.error.URLError:
            # 网络不可达（断网/防火墙/代理）：给出可操作的提示，而不是裸报错
            pname = active_provider().get("name") or "模型"
            msg = "无法连接模型服务（提供方：%s，网络不可达），请检查本机网络/代理设置，或用「配置模型 → 测试连接」定位" % pname
            print("[chat] " + msg)
            self.sse_send("event: error\ndata: 服务异常：%s\n\n" % msg)
            self.sse_send("event: done\n\n")
        except Exception as e:  # 网络超时/解析异常等，如实暴露
            msg = str(e)[:200]
            if "codec can't encode" in msg or "latin-1" in msg:
                msg += "（多为 API Key 包含非法字符，请在页面「配置模型 Key」重新粘贴干净的密钥）"
            print("[chat] 服务异常：%s" % msg)
            self.sse_send("event: error\ndata: 服务异常：%s\n\n" % msg)
            self.sse_send("event: done\n\n")

    # ---- 静态文件 ----
    def serve_static(self, path):
        if path == "/":
            path = "/index.html"
        path = urllib.parse.unquote(path)  # self.path 为百分号编码原始串：中文文件名（如 .md 文档链接）必须解码
        rel = Path(path.lstrip("/"))
        target = (WEB_DIR / rel).resolve()
        try:
            target.relative_to(WEB_DIR.resolve())
        except ValueError:
            self.send_json({"error": "forbidden"}, 403)
            return
        if not target.is_file():
            # 允许项目根目录的 Markdown 文档（README/部署/合规/提交说明）供页脚与「关于」弹窗链接：
            # 仅放行"根目录下的 .md 文件"，杜绝向任意根目录文件开放静态读取
            if rel.suffix.lower() == ".md":
                doc = (BASE_DIR / rel).resolve()
                try:
                    doc.relative_to(BASE_DIR.resolve())
                except ValueError:
                    doc = None
                if doc and doc.parent == BASE_DIR and doc.is_file():
                    target = doc
        if not target.is_file():
            if path.startswith("/api/"):
                self.send_json({"error": "not found"}, 404)
            else:
                # 非 API 的缺失资源返回友好 HTML 404（避免浏览器把 404 当文件下载）
                body = ("<!DOCTYPE html><html lang='zh-CN'><meta charset='utf-8'>"
                        "<title>404 · 知教</title>"
                        "<body style='font-family:system-ui,sans-serif;background:#eef1f6;color:#243042;"
                        "display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
                        "<div style='text-align:center'><h1 style='margin:0 0 6px'>404</h1>"
                        "<p style='margin:0 0 14px'>页面不存在。</p>"
                        "<a href='/'>返回首页</a></div></body></html>")
                data = body.encode("utf-8")
                self.send_response(404)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            return
        data = target.read_bytes()
        ctype = MIME.get(target.suffix.lower(), "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-cache")
        # gzip：文本类资源按客户端 Accept-Encoding 压缩（标准库，带 mtime 缓存避免重复压缩）
        base_ct = ctype.split(";")[0].strip()
        gz = None
        if "gzip" in self.headers.get("Accept-Encoding", "") and base_ct in COMPRESSIBLE:
            mtime = target.stat().st_mtime_ns
            cached = _GZ_CACHE.get(rel.as_posix())
            if cached and cached[0] == mtime:
                gz = cached[1]
            else:
                gz = gzip.compress(data, 6)
                _GZ_CACHE[rel.as_posix()] = (mtime, gz)
            if gz and len(gz) >= len(data):
                gz = None
        if gz:
            data = gz
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    # 启动时后台预热 SAPI 探测，避免首个 /api/health 请求被 PowerShell 探测阻塞
    threading.Thread(target=sapi_usable, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("=" * 56)
    print(" 知教 GOAI 后端已启动")
    print(" 前端地址：http://localhost:%d" % PORT)
    print(" 模型：%s" % MODEL)
    if API_KEY or _providers:
        print(" 模型提供方：已配置（%s）" % (active_provider().get("name") or "DeepSeek"))
    else:
        print(" 模型提供方：未配置（可在页面「配置模型」添加，或编辑 .env 后重启；前端自动回退剧本演示模式）")
    print("=" * 56)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[知教] 服务已停止")


if __name__ == "__main__":
    main()
