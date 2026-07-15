# -*- coding: utf-8 -*-
"""
Vector DB 비교 라이트 벤치 — Chroma vs FAISS vs Qdrant (benchmark.py 단일 파일).

목적: 발표 슬라이드 반쪽용 비교표. 실험 A(검색 품질 Recall@5) + 실험 B(규모별 성능).

[환경 결정 사항 — 이 클러스터(team_a2 SLURM) 실측 기준]
- 온프레미스, 외부 API 금지. 임베딩은 sentence-transformers 로컬 모델 BAAI/bge-m3
  (STEP7/8과 동일 모델, HF 캐시 오프라인 로드, 1024차원, CPU).
- 텔레메트리: Chroma anonymized_telemetry=False, Qdrant는 로컬(파일) 모드라 원천 없음
  (서버 모드용 docker-compose.yml에는 QDRANT__TELEMETRY_DISABLED=true 명시).
- ★ 이 클러스터는 docker/sudo가 없어 Qdrant 서버(localhost:6333)를 띄울 수 없다 —
  스크립트는 6333 접속을 먼저 시도하고, 실패하면 STEP7/8 실서비스와 동일한
  embedded(로컬 파일) 모드로 폴백한다. 어느 모드로 측정됐는지 results.csv에 기록.
- 랜덤 시드 42 고정 (random/numpy).

[데이터]
- data/raw/ 의 .md/.txt 문서를 500자 청크(overlap 100)로 분할.
- 메타데이터: category(파일명 규칙 — '보고서' 포함=report, '계획'=plan,
  '가이드'/'체크리스트'=guide, 그 외=note), year(파일 mtime 연도).
  ※ 원 템플릿의 '부서'를 이 프로젝트 문서 체계에 맞게 category로 치환.
- 청크 ID: "<상대경로>#<3자리 순번>" — questions.csv의 정답청크ID가 이걸 가리킨다.
- 임베딩은 1회 계산 후 embeddings.npy(+chunks.json) 캐시, L2 정규화, 3 DB 동일 주입.

실행:
    .venv(STEP7_DB)/bin/python3 benchmark.py --dry-run     # 문서 10개·1k 스케일만
    .venv(STEP7_DB)/bin/python3 benchmark.py --dump-chunks # 청크 ID 목록(질문 작성용)
    .venv(STEP7_DB)/bin/python3 benchmark.py               # 본 실험
출력: results.csv, slide_table.md, scaling_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
DATA_RAW = HERE / "data" / "raw"
EMB_NPY = HERE / "embeddings.npy"
CHUNKS_JSON = HERE / "chunks.json"
QUESTIONS_CSV = HERE / "questions.csv"
RESULTS_CSV = HERE / "results.csv"
SLIDE_MD = HERE / "slide_table.md"
CURVE_PNG = HERE / "scaling_curve.png"

EMB_MODEL = "BAAI/bge-m3"   # STEP7/8과 동일(로컬 캐시)
DIM = 1024
CHUNK_SIZE, CHUNK_OVERLAP = 500, 100
TOP_K = 5
SCALES = [1_000, 10_000, 100_000]
N_WARMUP, N_MEASURE = 10, 100
CATEGORIES = ["report", "plan", "guide", "note"]

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chroma 구버전 환경변수 겸용

KOREAN_FONT = "/usr/share/fonts/google-noto-sans-mono-cjk-vf-fonts/NotoSansMonoCJK-VF.ttc"


# ──────────────────────────────────────────────────────────────────
# 1. 청킹 + 메타데이터
# ──────────────────────────────────────────────────────────────────
def category_of(filename: str) -> str:
    """파일명 규칙으로 카테고리 부여(원 템플릿의 '부서' 파싱을 문서 체계에 맞게 치환)."""
    if "보고서" in filename:
        return "report"
    if "계획" in filename:
        return "plan"
    if "가이드" in filename or "체크리스트" in filename:
        return "guide"
    return "note"


def build_chunks(limit_docs: int | None = None) -> list[dict]:
    files = sorted(p for p in DATA_RAW.rglob("*") if p.suffix in (".md", ".txt"))
    if limit_docs:
        files = files[:limit_docs]
    assert files, f"data/raw 에 문서 없음: {DATA_RAW}"
    chunks = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        rel = str(p.relative_to(DATA_RAW))
        year = time.localtime(p.stat().st_mtime).tm_year
        cat = category_of(p.name)
        i, n = 0, 0
        while i < len(text):
            piece = text[i:i + CHUNK_SIZE].strip()
            if len(piece) >= 50:  # 잔부스러기 제외
                chunks.append({"id": f"{rel}#{n:03d}", "text": piece,
                               "category": cat, "year": year})
                n += 1
            i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ──────────────────────────────────────────────────────────────────
# 2. 임베딩 (1회 계산 → npy 캐시, L2 정규화)
# ──────────────────────────────────────────────────────────────────
def embed_chunks(chunks: list[dict], force: bool = False) -> np.ndarray:
    if EMB_NPY.exists() and CHUNKS_JSON.exists() and not force:
        cached = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        if [c["id"] for c in cached] == [c["id"] for c in chunks]:
            print(f"[emb] 캐시 재사용: {EMB_NPY}")
            return np.load(EMB_NPY)
    from sentence_transformers import SentenceTransformer
    print(f"[emb] {EMB_MODEL} 로드(CPU) 후 {len(chunks)}청크 임베딩...")
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    vecs = model.encode([c["text"] for c in chunks], batch_size=16,
                        show_progress_bar=True, normalize_embeddings=True)
    vecs = np.asarray(vecs, dtype=np.float32)
    np.save(EMB_NPY, vecs)
    CHUNKS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    return vecs


def embed_queries(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMB_MODEL, device="cpu")
    return np.asarray(model.encode(texts, normalize_embeddings=True), dtype=np.float32)


# ──────────────────────────────────────────────────────────────────
# 3. DB 어댑터 3종 — 동일 인터페이스(insert / search / filter_search)
# ──────────────────────────────────────────────────────────────────
class ChromaDB:
    name = "Chroma"
    mode = "in-process"

    def __init__(self, tag: str):
        import chromadb
        from chromadb.config import Settings
        # 텔레메트리 비활성 필수 + 매 스케일 새 인메모리 클라이언트(측정 오염 방지)
        self.client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
        self.col = self.client.create_collection(f"bench_{tag}", metadata={"hnsw:space": "ip"})

    def insert(self, ids, vecs, metas):
        B = 5000  # chroma 1회 add 상한 회피
        for s in range(0, len(ids), B):
            self.col.add(ids=list(ids[s:s+B]), embeddings=vecs[s:s+B].tolist(),
                         metadatas=list(metas[s:s+B]))

    def search(self, qvec, k=TOP_K):
        r = self.col.query(query_embeddings=[qvec.tolist()], n_results=k)
        return r["ids"][0]

    def filter_search(self, qvec, flt: dict, k=TOP_K):
        # {"category": "report", "year_gte": 2026} → chroma where 문법
        where = self._where(flt)
        r = self.col.query(query_embeddings=[qvec.tolist()], n_results=k, where=where)
        return r["ids"][0]

    @staticmethod
    def _where(flt):
        conds = []
        if "category" in flt:
            conds.append({"category": {"$eq": flt["category"]}})
        if "year_gte" in flt:
            conds.append({"year": {"$gte": flt["year_gte"]}})
        return conds[0] if len(conds) == 1 else {"$and": conds}


class FaissDB:
    name = "FAISS"
    mode = "in-process"
    # ── 메타데이터 필터 "직접 구현" 블록 — 슬라이드 근거용으로 줄 수를 셈 ──
    # FAISS는 벡터 인덱스일 뿐 메타데이터 저장/필터 기능이 없어, 조건에 맞는 id 배열을
    # 우리가 직접 만들어 IDSelectorArray로 후보를 제한해야 한다(사전 필터링).
    FILTER_IMPL_FIRST = "  # <filter-impl-start>"

    def __init__(self, tag: str):
        import faiss
        self.faiss = faiss
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
        self.metas: dict[int, dict] = {}
        self.ids: list[str] = []

    def insert(self, ids, vecs, metas):
        base = len(self.ids)
        int_ids = np.arange(base, base + len(ids), dtype=np.int64)
        self.index.add_with_ids(vecs, int_ids)
        for j, m in enumerate(metas):
            self.metas[base + j] = m
        self.ids.extend(ids)

    def search(self, qvec, k=TOP_K):
        _, I = self.index.search(qvec[None, :], k)
        return [self.ids[i] for i in I[0] if i >= 0]

    def filter_search(self, qvec, flt: dict, k=TOP_K):
        # <filter-impl-start>
        # 1) 메타데이터 사전 스캔으로 조건에 맞는 내부 id 배열을 직접 구성
        allowed = np.array(
            [iid for iid, m in self.metas.items()
             if ("category" not in flt or m["category"] == flt["category"])
             and ("year_gte" not in flt or m["year"] >= flt["year_gte"])],
            dtype=np.int64)
        if allowed.size == 0:
            return []
        # 2) IDSelectorArray + SearchParameters 로 후보 제한 검색
        sel = self.faiss.IDSelectorArray(allowed)
        params = self.faiss.SearchParameters(sel=sel)
        _, I = self.index.search(qvec[None, :], k, params=params)
        return [self.ids[i] for i in I[0] if i >= 0]
        # <filter-impl-end>

    @classmethod
    def filter_impl_lines(cls) -> int:
        """<filter-impl-start>~end 사이 실코드 줄 수(빈 줄·주석 제외) — 결과에 기록."""
        src = Path(__file__).read_text(encoding="utf-8").splitlines()
        s = next(i for i, l in enumerate(src) if "<filter-impl-start>" in l and "FIRST" not in l)
        e = next(i for i, l in enumerate(src) if "<filter-impl-end>" in l)
        return sum(1 for l in src[s+1:e] if l.strip() and not l.strip().startswith("#"))


class QdrantDB:
    name = "Qdrant"

    def __init__(self, tag: str):
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
        self.qm = qm
        self.colname = f"bench_{tag}"
        # 서버(docker, localhost:6333) 우선 → 실패 시 embedded 폴백(이 클러스터는 docker
        # 불가라 실측은 embedded — STEP7/8 실서비스와 동일 모드라 오히려 실사용 대표성 있음)
        try:
            self.client = QdrantClient(url="http://localhost:6333", timeout=3)
            self.client.get_collections()
            self.mode = "server(:6333)"
        except Exception:
            self._tmp = HERE / f"_qdrant_bench_{tag}"
            if self._tmp.exists():
                shutil.rmtree(self._tmp)
            self.client = QdrantClient(path=str(self._tmp))
            self.mode = "embedded"
        self.client.create_collection(
            self.colname, vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE))
        self.ids: list[str] = []

    def insert(self, ids, vecs, metas):
        qm = self.qm
        base = len(self.ids)
        B = 1000
        for s in range(0, len(ids), B):
            pts = [qm.PointStruct(id=base + s + j, vector=vecs[s + j].tolist(),
                                  payload=metas[s + j])
                   for j in range(min(B, len(ids) - s))]
            self.client.upsert(self.colname, pts, wait=True)
        self.ids.extend(ids)
        # 인덱싱 완료(green) 대기 — 서버 모드에서 의미(HNSW 빌드), embedded도 동일 API
        for _ in range(600):
            info = self.client.get_collection(self.colname)
            if str(info.status).lower().endswith("green"):
                break
            time.sleep(0.5)

    def _hits(self, res):
        return [self.ids[h.id] for h in res]

    def search(self, qvec, k=TOP_K):
        res = self.client.query_points(self.colname, query=qvec.tolist(), limit=k).points
        return self._hits(res)

    def filter_search(self, qvec, flt: dict, k=TOP_K):
        qm = self.qm
        must = []
        if "category" in flt:
            must.append(qm.FieldCondition(key="category", match=qm.MatchValue(value=flt["category"])))
        if "year_gte" in flt:
            must.append(qm.FieldCondition(key="year", range=qm.Range(gte=flt["year_gte"])))
        res = self.client.query_points(self.colname, query=qvec.tolist(), limit=k,
                                       query_filter=qm.Filter(must=must)).points
        return self._hits(res)

    def cleanup(self):
        try:
            self.client.delete_collection(self.colname)
            self.client.close()
            if hasattr(self, "_tmp") and self._tmp.exists():
                shutil.rmtree(self._tmp)
        except Exception:
            pass


DB_CLASSES = [ChromaDB, FaissDB, QdrantDB]


# ──────────────────────────────────────────────────────────────────
# 4. 실험 A — 검색 품질 (Recall@5, questions.csv 기반)
# ──────────────────────────────────────────────────────────────────
RECALL_KS = (1, 5, 10)  # 2026-07-12: Recall@1/5/10 동시 산출(top-10 한 번 검색해 접두 판정)


def experiment_a(chunks, vecs, rows: list[dict]):
    assert QUESTIONS_CSV.exists(), f"questions.csv 없음: {QUESTIONS_CSV}"
    with QUESTIONS_CSV.open(encoding="utf-8") as f:
        questions = [q for q in csv.DictReader(f) if q["질문"].strip()]
    print(f"[A] 질문 {len(questions)}건 로드")
    qvecs = embed_queries([q["질문"] for q in questions])

    ids = [c["id"] for c in chunks]
    metas = [{"category": c["category"], "year": c["year"]} for c in chunks]
    kmax = max(RECALL_KS)
    recalls = {}
    for cls in DB_CLASSES:
        db = cls("expA")
        db.insert(ids, vecs, metas)
        hits = {k: 0 for k in RECALL_KS}
        for q, qv in zip(questions, qvecs):
            flt = json.loads(q["필터조건JSON"]) if q["필터조건JSON"].strip() else None
            got = db.filter_search(qv, flt, k=kmax) if flt else db.search(qv, k=kmax)
            for k in RECALL_KS:
                ok = q["정답청크ID"] in got[:k]
                hits[k] += ok
                rows.append({"exp": "A", "db": db.name, "scale": len(ids),
                             "metric": f"hit@{k}", "question": q["질문"][:40],
                             "expected": q["정답청크ID"], "value": int(ok), "mode": db.mode})
        recalls[db.name] = {k: hits[k] / len(questions) for k in RECALL_KS}
        print(f"[A] {db.name:<7} " + " ".join(
            f"R@{k}={hits[k]}/{len(questions)}={recalls[db.name][k]:.2f}" for k in RECALL_KS)
            + f" ({db.mode})")
        if hasattr(db, "cleanup"):
            db.cleanup()
    rows.append({"exp": "A", "db": "FAISS", "scale": len(ids), "metric": "filter_impl_lines",
                 "question": "", "expected": "", "value": FaissDB.filter_impl_lines(), "mode": ""})
    return recalls


# ──────────────────────────────────────────────────────────────────
# 5. 실험 B — 규모별 성능 (삽입 시간, 검색/필터 p95)
# ──────────────────────────────────────────────────────────────────
def p95(xs):
    return float(np.percentile(np.array(xs) * 1000.0, 95))  # ms


def experiment_b(scales, rows: list[dict]):
    results = {}
    for n in scales:
        # 같은 차원 랜덤 벡터 + 랜덤 메타데이터(시드 고정 — DB 간 동일 데이터)
        rng = np.random.default_rng(SEED)
        vecs = rng.standard_normal((n, DIM), dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        cats = rng.choice(CATEGORIES, size=n)
        years = rng.integers(2020, 2027, size=n)
        ids = [f"r{i:06d}" for i in range(n)]
        metas = [{"category": str(cats[i]), "year": int(years[i])} for i in range(n)]
        qv = rng.standard_normal((N_WARMUP + N_MEASURE, DIM)).astype(np.float32)
        qv /= np.linalg.norm(qv, axis=1, keepdims=True)

        for cls in DB_CLASSES:
            db = cls(f"expB{n}")
            t0 = time.perf_counter()
            db.insert(ids, vecs, metas)
            ins = time.perf_counter() - t0

            for w in range(N_WARMUP):
                db.search(qv[w])
            lat = []
            for i in range(N_MEASURE):
                t = time.perf_counter(); db.search(qv[N_WARMUP + i]); lat.append(time.perf_counter() - t)
            flat = []
            for i in range(N_MEASURE):
                flt = {"category": CATEGORIES[i % 4]}
                t = time.perf_counter(); db.filter_search(qv[N_WARMUP + i], flt); flat.append(time.perf_counter() - t)

            r = {"insert_s": round(ins, 2), "p95_ms": round(p95(lat), 2),
                 "filter_p95_ms": round(p95(flat), 2), "mode": db.mode}
            results[(db.name, n)] = r
            print(f"[B] {db.name:<7} n={n:>7,} 삽입 {r['insert_s']}s | p95 {r['p95_ms']}ms | "
                  f"필터 p95 {r['filter_p95_ms']}ms ({db.mode})")
            for metric in ("insert_s", "p95_ms", "filter_p95_ms"):
                rows.append({"exp": "B", "db": db.name, "scale": n, "metric": metric,
                             "question": "", "expected": "", "value": r[metric], "mode": db.mode})
            if hasattr(db, "cleanup"):
                db.cleanup()
    return results


# ──────────────────────────────────────────────────────────────────
# 6. 출력 — results.csv / slide_table.md / scaling_curve.png
# ──────────────────────────────────────────────────────────────────
def write_outputs(recalls, bres, scales, rows):
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp", "db", "scale", "metric", "question",
                                          "expected", "value", "mode"])
        w.writeheader()
        w.writerows(rows)

    big = max(scales)
    impl_lines = FaissDB.filter_impl_lines()
    md = ["| | Chroma | FAISS | **Qdrant** |", "|---|---|---|---|"]
    for k in RECALL_KS:
        md.append(f"| Recall@{k} (동일 임베딩) | " + " | ".join(
            f"{recalls[d][k]:.2f}" if d in recalls else "n/a"
            for d in ("Chroma", "FAISS", "Qdrant")) + " |")
    md.append(f"| 검색 p95, {big:,}건 | " + " | ".join(
        f"{bres[(d, big)]['p95_ms']} ms" for d in ("Chroma", "FAISS", "Qdrant")) + " |")
    md.append(f"| 필터 검색 p95, {big:,}건 | " + " | ".join(
        f"{bres[(d, big)]['filter_p95_ms']} ms" for d in ("Chroma", "FAISS", "Qdrant")) + " |")
    md.append(f"| 삽입 시간, {big:,}건 | " + " | ".join(
        f"{bres[(d, big)]['insert_s']} s" for d in ("Chroma", "FAISS", "Qdrant")) + " |")
    md.append(f"| 메타데이터 필터 | 기본 지원($eq·$and·범위) | 직접 구현 필요({impl_lines}줄) | "
              "강력 지원(중첩 must/should·범위, payload 인덱스) |")
    # 2026-07-12 2차 수정(검수 반영): 구행 "서버형/동시접속 ✗(인프로세스 전용)"은 Chroma에
    # HttpClient 서버 모드가 실존하므로 사실 오류였음 → '프로덕션 기능' 비교 행으로 교체.
    # FAISS는 벡터 양자화(PQ/SQ) 자체는 있으나 인증·스냅샷·서버 운영 기능이 없어 ✗ — 각주로 명시.
    md.append("| 프로덕션 기능 (인증·스냅샷 백업·양자화) | △ (서버 모드 시 인증 지원, 백업·양자화 제한) | "
              "✗ (라이브러리 — 운영 기능 없음)* | ✓ (API key 인증·스냅샷·스칼라/프로덕트 양자화) |")
    md.append("")
    md.append(f"※ Qdrant 측정 모드: {bres[(('Qdrant'), big)]['mode']} — STEP7/8 실서비스와 동일한 "
              "embedded 모드로 측정(= 실배포와 동일 조건). 서버 모드 배포 시 docker-compose.yml 사용, "
              "그 경우 latency는 별도 재측정 필요(embedded와 특성이 다름).")
    md.append("※* FAISS는 벡터 양자화(PQ/SQ) 알고리즘은 지원하나 인증·백업·서버 운영 계층이 없다"
              "(라이브러리이므로) — '프로덕션 기능' 관점의 ✗.")
    md.append("")
    md.append("**결론**: 검색 품질 동등, 속도 차이는 ms 단위 → 필터 검색 표현력(FAISS는 직접 구현 "
              "필요)과 프로덕션 기능·무변경 서버 전환 경로 기준으로 **Qdrant 선정**")
    SLIDE_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))

    # 스케일링 커브 (X: 벡터 수 로그, Y: p95 ms)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    if Path(KOREAN_FONT).exists():
        fm.fontManager.addfont(KOREAN_FONT)
        plt.rcParams["font.family"] = fm.FontProperties(fname=KOREAN_FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 4))
    for d, marker in (("Chroma", "o"), ("FAISS", "s"), ("Qdrant", "^")):
        ax.plot(scales, [bres[(d, n)]["p95_ms"] for n in scales], marker=marker, label=d)
    ax.set_xscale("log")
    ax.set_xlabel("벡터 수 (로그 스케일)")
    ax.set_ylabel("검색 p95 latency (ms)")
    ax.set_title("Vector DB 규모별 검색 p95 (top-5, 동일 랜덤 벡터)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_PNG, dpi=300)
    print(f"[OK] {RESULTS_CSV.name} / {SLIDE_MD.name} / {CURVE_PNG.name} 저장")


# ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="문서 10개·1k 스케일만 빠르게")
    ap.add_argument("--dump-chunks", action="store_true", help="청크 ID+미리보기 출력(질문 작성용)")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--reuse-b", action="store_true",
                    help="실험 B는 기존 results.csv 값 재사용(A만 재실행 — Recall K 변경 등)")
    args = ap.parse_args()

    chunks = build_chunks(limit_docs=10 if args.dry_run else None)
    print(f"[chunk] {len(chunks)}개 (문서 {len(set(c['id'].split('#')[0] for c in chunks))}건, "
          f"카테고리 분포: { {c: sum(1 for x in chunks if x['category']==c) for c in CATEGORIES} })")
    if args.dump_chunks:
        for c in chunks:
            print(f"{c['id']}\t[{c['category']}/{c['year']}]\t{c['text'][:60].replace(chr(10),' ')}")
        return

    vecs = embed_chunks(chunks)
    assert vecs.shape == (len(chunks), DIM), f"임베딩 shape 이상: {vecs.shape}"

    rows: list[dict] = []
    scales = [1_000] if args.dry_run else SCALES
    # --reuse-b: 기존 results.csv의 실험 B 행을 그대로 승계(덮어쓰기 전에 로드)
    bres, reused_rows = {}, []
    if args.reuse_b and RESULTS_CSV.exists():
        with RESULTS_CSV.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["exp"] != "B":
                    continue
                reused_rows.append(r)
                key = (r["db"], int(r["scale"]))
                bres.setdefault(key, {"mode": r["mode"]})[r["metric"]] = float(r["value"])
        scales = sorted({s for _, s in bres})
        print(f"[B] 기존 results.csv에서 재사용: {len(reused_rows)}행, scales={scales}")

    recalls = {} if args.skip_a else experiment_a(chunks, vecs, rows)
    if not args.skip_b and not args.reuse_b:
        bres = experiment_b(scales, rows)
    rows.extend(reused_rows)
    if recalls and bres:
        write_outputs(recalls, bres, scales, rows)
    # 스모크 원칙: 빈 결과로 exit 0 금지
    assert rows, "측정 결과 0건"


if __name__ == "__main__":
    main()
