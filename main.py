import os
import json
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI
import openai as _openai

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

    class Config:
        # 👈 これを追加！
        extra = "forbid"  # additionalProperties: false 相当
        schema_extra = {
            "additionalProperties": False}

# ---------- OpenAI Client ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- FastAPI ----------
app = FastAPI(title="Episode Talk Maker API", version="0.2.0")

origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def build_system_prompt():
    return (
        "あなたは日本語の“間”と弱毒ユーモアに最適化された放送作家です。"
        "ユーザーの出来事を、飲み会/面接/配信など状況に応じたトーンで、"
        "フック→事実→ズレ→展開→オチ→余韻の構成に割り当て、"
        "1行80字以内の口語台本（[間1.0s]等の演出、擬音、比喩、コールバック、三段オチ）を生成します。"
        "総尺は指定±15%以内。実名/会社名は自動で匿名化（例: 友人A/会社B）。"
        "差別/誹謗中傷は弱毒化し、NGワードは出力に含めません。"
        "各パート秒数、スライド（TITLE/BULLETS/PUNCHLINE）も出力します。"
        "出力は必ず与えたJSONスキーマに完全準拠してください。"
    )

def build_user_prompt(ep: EpisodeIn):
    ng_join = ", ".join(ep.ng) if ep.ng else "（なし）"
    return (
        f"入力:\n"
        f"- いつ: {ep.when}\n"
        f"- どこ: {ep.where}\n"
        f"- 誰と: {ep.who}\n"
        f"- 何が: {ep.what}\n"
        f"- 感情: {ep.emotion}\n"
        f"- ターゲット: {ep.target}\n"
        f"- トーン: {ep.tone}\n"
        f"- 尺(秒): {ep.duration_sec}\n"
        f"- NGワード: {ng_join}\n\n"
        f"要件:\n"
        f"1) 構成：フック→事実→ズレ→展開→オチ→余韻（各パート秒数配分）。\n"
        f"2) 台本：口語。1行80字以内。各行に秒数目安と[間x.xs]等の演出。言い換え候補を1-2個。\n"
        f"3) スライド：TITLE/BULLETS/PUNCHLINEで、最小3〜最大6枚。\n"
        f"4) 総尺は±15%以内。冗長さは圧縮。\n"
        f"5) 匿名化：固有名詞を友人A/会社Bなどに変換し、匿名化レベルを0〜2で評価。\n"
        f"6) NGワードを含めない。含む恐れがある場合はwarningsに説明。\n"
    )

def output_json_schema():
    # Chat Completions の strict:true では、すべての object に additionalProperties:false が必要
    # かつ properties にある全キーを required に列挙する必要がある
    return {
        "type": "object",
        "additionalProperties": False,  # ★ ルート必須
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
                    "required": ["id", "name", "seconds", "summary"]  # ★ 全キー
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
                        "alternatives": {"type": "array", "items": {"type": "string"}, "maxItems": 2}
                    },
                    # ★ 全キー必須（strict:true要件）
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
                    # ★ 全キー必須（strict:true要件）
                    "required": ["kind", "title", "bullets", "note"]
                }
            }
        },
        # ★ ルートも全キー必須（warnings を追加）
        "required": ["anonymization_level", "warnings", "beats", "script", "slides"]
    }



@app.post("/generate", response_model=EpisodeOut)
def generate(ep: EpisodeIn):
    try:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(ep)

        # --- Chat Completions + JSON Schema（SDK2.x安定対応） ---
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "EpisodeOut",
                    "schema": output_json_schema(),
                    "strict": True,
                },
            },
            timeout=60,
        )

        raw = chat.choices[0].message.content
        if not raw:
            raise RuntimeError("Empty content from OpenAI")

        data = json.loads(raw)
        return EpisodeOut(**data)

    except Exception as e:
        print("ERROR in /generate:", repr(e))
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))