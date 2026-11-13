"""特征列 schema：首跑生成，之后严格校验，防止列漂移。"""
import json

from chan_notes.config import DATA

SCHEMA_PATH = DATA / "schema_v1.json"


def load() -> list[str] | None:
    if SCHEMA_PATH.exists():
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return None


def ensure(columns) -> list[str]:
    cols = sorted(columns)
    cur = load()
    if cur is None:
        SCHEMA_PATH.write_text(json.dumps(cols, ensure_ascii=False, indent=1), encoding="utf-8")
        return cols
    if set(cols) != set(cur):
        miss = sorted(set(cur) - set(cols))
        extra = sorted(set(cols) - set(cur))
        raise RuntimeError(f"特征 schema 漂移：缺失 {miss[:10]} / 新增 {extra[:10]}")
    return cur


def align(row: dict) -> dict:
    cols = load()
    if cols is None:
        return row
    return {c: row.get(c, 0.0) for c in cols}
