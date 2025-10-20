import os
import json
from typing import List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from openai import OpenAI
import openai as _openai
import traceback

print("OpenAI SDK on server:", getattr(_openai, "__version__", "unknown"))

load_dotenv()

# ---------- Pydantic Models ----------
BeatName = Literal["フック", "事実", "ズレ", "展開", "オチ", "余韻"]
SlideKind = Literal["TITLE", "BULLETS", "PUNCHLINE"]

class EpisodeIn(BaseModel):
    when: str
    where: str
    who: str
    what: str
    emotion: str
    target: Literal["飲み会", "面接", "自己紹介", "YouTube", "司会", "配信", "その他"] = "飲み会"
    tone: Literal["爽やか", "自虐", "毒弱め", "ノーマル"] = "ノーマル"
    duration_sec: int = Field(..., ge=30, le=600)
    ng: List[str] = Field(default_factory=list)
    embellish_rate: int = Field(20, ge=0, le=100, description="脚色率 0–100")

class Beat(BaseModel):
    id: str
    name: BeatName
    seconds: int
    summary: str

class ScriptLine(BaseModel):
    beat_id: str
    seconds: int
    text: str
    pause: float = 0.0
    alternatives: List[str] = Field(default_factory=list)

class Slide(BaseModel):
    kind: SlideKind
    title: str
    bullets: List[str] = Field(default_factory=list, max_items=5)
    note: Optional[str] = None

class EpisodeOut(BaseModel):
    anonymization_level: int = Field(..., ge=0, le=2)
    warnings: List[str] = Field(default_factory=list)
    beats: List[Beat]
    script: List[ScriptLine]
    slides: List[Slide]

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False}
    )

# ---------- OpenAI Client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------- FastAPI ----------
app = FastAPI(title="Episode Talk Maker API", version="0.4.0")

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"service": "Episode Talk Maker API", "health": "/health", "docs": "/docs"}

@app.get("/health")
def health():
    return {"ok": True, "openai_key_configured": bool(OPENAI_API_KEY)}

# ---------- Prompts ----------
CHARS_PER_SEC = float(os.getenv("CHARS_PER_SEC", "6.2"))  # 尺×6.2 をデフォルト

def build_system_prompt():
    return (
        "あなたは日本語の“間”と弱毒ユーモアに最適化された放送作家です。"
        "ユーザーの出来事を、飲み会/面接/配信など状況に応じたトーンで、"
        "フック→事実→ズレ→展開→オチ→余韻の構成に割り当て、"
        "1行80字以内の口語台本を生成します。"
        "総尺は指定±15%以内。"
        "差別/誹謗中傷は弱毒化し、NGワードは出力に含めません。"
        "各パート秒数、スライド（TITLE/BULLETS/PUNCHLINE）も出力します。"
        "出力は必ず与えたJSONスキーマに完全準拠してください。"
    )

def build_user_prompt(ep: EpisodeIn) -> Tuple[str, int, int, int]:
    ng_join = ", ".join(ep.ng) if ep.ng else "（なし）"
    target_chars = int(ep.duration_sec * CHARS_PER_SEC)
    # 許容レンジ（±15%）
    min_chars = int(target_chars * 0.85)
    max_chars = int(target_chars * 1.15)
    emb = ep.embellish_rate

    prompt = (
        f"入力:\n"
        f"- いつ: {ep.when}\n"
        f"- どこ: {ep.where}\n"
        f"- 誰と: {ep.who}\n"
        f"- 何が: {ep.what}\n"
        f"- 感情: {ep.emotion}\n"
        f"- ターゲット: {ep.target}\n"
        f"- トーン: {ep.tone}\n"
        f"- 尺(秒): {ep.duration_sec}\n"
        f"- 脚色率(0-100): {emb}\n"
        f"- NGワード: {ng_join}\n\n"
        f"要件:\n"
        f"【要点（=骨子）】\n"
        f"- フック/事実/ズレ/展開/オチ/余韻 の6項目。\n"
        f"- 各項目は **短く一文**、上限30文字/項目。事実の核のみ。比喩や擬音は使わない。脚色は行わない。\n"
        f"- 各項目に推奨秒数を割当（合計は尺の±15%）。\n\n"
        f"【台本】\n"
        f"- 口語、自然な一人語り調。1行80字以内で改行。\n"
        f"- 総文字数は 約 {target_chars} 文字（許容レンジ {min_chars}–{max_chars} 文字）。\n"
        f"- [間x.xs] 等の表記は付けない。\n"
        f"- 脚色率 {emb}% に応じて演出度を調整：\n"
        f"  ・0%: 事実重視。誇張なし。具体的描写は控えめ。\n"
        f"  ・50%: 比喩・擬音・テンポの工夫で“面白いけど現実感”。\n"
        f"  ・100%: 誇張/比喩/擬音を大胆に。筋は変えずに演出強化。\n\n"
        f"【スライド】\n"
        f"- TITLE/BULLETS/PUNCHLINE の3–6枚。\n"
        f"- BULLETSは短文で要点のみ。\n"
        f"\n出力は与えたJSONスキーマに厳密準拠すること。"
    )
    return prompt, target_chars, min_chars, max_chars

def build_adjust_prompt(current_texts: List[str], min_chars: int, max_chars: int) -> str:
    joined = "\n".join(current_texts)
    return (
        "次の台本テキスト群は、総文字数が目標レンジ外です。"
        f"総文字数が {min_chars}〜{max_chars} の範囲に入るよう、"
        "構成（ビート数）・要点・スライドは変えずに、台本の text のみを増減して再出力してください。"
        "JSON スキーマは同一で、script の text を中心に調整すること。\n\n"
        f"【現在の台本テキスト】\n{joined}\n"
    )

# ---------- JSON Schema ----------
def output_json_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "anonymization_level": {"type": "integer", "minimum": 0, "maximum": 2},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "beats": {
                "type": "array",
                "minItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string", "enum": ["フック", "事実", "ズレ", "展開", "オチ", "余韻"]},
                        "seconds": {"type": "integer", "minimum": 1},
                        "summary": {"type": "string"}
                    },
                    "required": ["id", "name", "seconds", "summary"]
                }
            },
            "script": {
                "type": "array",
                "minItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "beat_id": {"type": "string"},
                        "seconds": {"type": "integer", "minimum": 1},
                        "text": {"type": "string"},
                        "pause": {"type": "number", "minimum": 0},
                        "alternatives": { "type": "array", "items": {"type": "string"}, "maxItems": 0 }
                    },
                    "required": ["beat_id", "seconds", "text", "pause", "alternatives"]
                }
            },
            "slides": {
                "type": "array",
                "minItems": 3,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {"type": "string", "enum": ["TITLE", "BULLETS", "PUNCHLINE"]},
                        "title": {"type": "string"},
                        "bullets": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
                        "note": {"type": "string"}
                    },
                    "required": ["kind", "title", "bullets", "note"]
                }
            }
        },
        "required": ["anonymization_level", "warnings", "beats", "script", "slides"]
    }

# ---------- Helpers ----------
def count_script_chars(script: List[ScriptLine]) -> int:
    return sum(len(s.text) for s in script)

def call_chat(messages, schema):
    # 従属性を高めるため温度を低めに、タイムアウトを長めに
    return client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=4096,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "EpisodeOut",
                "schema": schema,
                "strict": True,
            },
        },
        timeout=90,
    )

# ---------- Endpoint ----------
@app.post("/generate", response_model=EpisodeOut)
def generate(ep: EpisodeIn):
    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured on server")

    try:
        system_prompt = build_system_prompt()
        user_prompt, target_chars, min_chars, max_chars = build_user_prompt(ep)
        schema = output_json_schema()

        # 1st try
        chat = call_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            schema=schema
        )
        raw = chat.choices[0].message.content
        if not raw:
            raise RuntimeError("Empty content from OpenAI (1st try)")

        data = json.loads(raw)
        out = EpisodeOut(**data)

        total = count_script_chars(out.script)
        if min_chars <= total <= max_chars:
            return out  # 目標レンジ内 → そのまま返す

        # 2nd try: 調整プロンプトで増量/圧縮を依頼（1回だけ）
        current_texts = [s.text for s in out.script]
        adjust_prompt = build_adjust_prompt(current_texts, min_chars, max_chars)
        chat2 = call_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "user", "content": adjust_prompt},
            ],
            schema=schema
        )
        raw2 = chat2.choices[0].message.content
        if not raw2:
            # 調整が返らなければ初回結果を返す（最低限成立させる）
            return out

        data2 = json.loads(raw2)
        out2 = EpisodeOut(**data2)
        # 調整結果で OK ならそれを返す。ダメでも out を返す。
        total2 = count_script_chars(out2.script)
        return out2 if min_chars <= total2 <= max_chars else out

    except Exception as e:
        print("ERROR in /generate:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))