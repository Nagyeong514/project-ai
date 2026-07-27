# -*- coding: utf-8 -*-
"""
benchmark_v3.py — Vector DB 비교 벤치 (확정 지표 3종)

지표:
  ① 메모리 사용량 (MB)  — 데이터 로드 완료 후 프로세스 RSS 증가분
  ② 콜드 스타트 (s)     — 새 프로세스에서 디스크의 저장 데이터 로드 → 첫 검색 성공까지
  ③ 처리량 QPS          — 단일 스레드, 워밍업 10회 후 100회 검색 (p95 ms 참고 병기)

측정 구조 (정확한 콜드 스타트를 위한 2단계 프로세스 분리):
  [build 프로세스]  합성 벡터 삽입 → 디스크 persist → 종료
  [measure 프로세스] 새로 기동 → 로드 시간(=콜드 스타트) → RAM → QPS 측정
  DB×규모 조합마다 두 프로세스 모두 새로 띄워 RSS 오염 없이 격리 측정.

조건(기존 실험과 동일): 시드 42, L2 정규화 cosine, Chroma 텔레메트리 오프,
  Qdrant는 실배포(STEP7/8)와 동일한 embedded 모드(QdrantClient(path=...)).

실행:
  python benchmark_v3.py              # 1천/1만/10만 전체
  python benchmark_v3.py --dry-run    # 1천만 빠르게
산출: results_v3.csv, slide_table_v3.md
"""
import argparse, csv, gc, json, os, pickle, shutil, subprocess, sys, time
import numpy as np

SEED = 42
SCALES = [1000, 10000, 100000]
DBS = ["chroma", "faiss", "qdrant"]
DIM = 1024           # 임베딩 차원 — 기존 실험(bge-m3, benchmark.py)과 동일 (2026-07-12: 384→1024)
N_QUERY, N_WARMUP = 100, 10
OUT_CSV, OUT_TABLE = "results_v3.csv", "slide_table_v3.md"

def synthetic(n: int, dim: int):
    rng = np.random.default_rng(SEED)
    v = rng.standard_normal((n, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [f"syn_{i:06d}" for i in range(n)], v

def rss_mb():
    import psutil
    return psutil.Process().memory_info().rss / 1e6

def wdir(db, scale):
    return f"_bench_{db}_{scale}"

# ──────────────────────────────────────────────────────────────
# build 단계: 삽입 + persist 후 종료 (콜드 스타트 측정을 위해 프로세스 분리)
# ──────────────────────────────────────────────────────────────
def build(db, scale):
    ids, vecs = synthetic(scale, DIM)
    d = wdir(db, scale)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

    if db == "chroma":
        import chromadb
        from chromadb.config import Settings
        cli = chromadb.PersistentClient(path=os.path.join(d, "store"),
                                        settings=Settings(anonymized_telemetry=False))
        col = cli.create_collection("bench", metadata={"hnsw:space": "cosine"})
        for s in range(0, scale, 5000):
            col.add(ids=ids[s:s+5000], embeddings=vecs[s:s+5000].tolist())

    elif db == "faiss":
        import faiss
        index = faiss.IndexFlatIP(DIM)
        index.add(vecs)
        faiss.write_index(index, os.path.join(d, "index.faiss"))
        pickle.dump(ids, open(os.path.join(d, "ids.pkl"), "wb"))

    elif db == "qdrant":
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        cli = QdrantClient(path=os.path.join(d, "store"))  # embedded — 실배포 동일
        cli.create_collection("bench", vectors_config=VectorParams(
            size=DIM, distance=Distance.COSINE))
        pts = [PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vecs)]
        for s in range(0, scale, 2000):
            cli.upsert("bench", pts[s:s+2000], wait=True)
        cli.close()  # 파일 락 해제 — measure 프로세스가 열 수 있도록
    print("BUILD_OK")

# ──────────────────────────────────────────────────────────────
# measure 단계: 새 프로세스에서 콜드 스타트 → RAM → QPS
# ──────────────────────────────────────────────────────────────
def measure(db, scale):
    d = wdir(db, scale)
    rng = np.random.default_rng(SEED + 7)
    qs = rng.standard_normal((N_WARMUP + N_QUERY, DIM)).astype("float32")
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    gc.collect(); ram0 = rss_mb()

    # ── ① 콜드 스타트: 클라이언트 기동 + 데이터 로드 + 첫 검색 성공까지 ──
    t0 = time.perf_counter()
    if db == "chroma":
        import chromadb
        from chromadb.config import Settings
        cli = chromadb.PersistentClient(path=os.path.join(d, "store"),
                                        settings=Settings(anonymized_telemetry=False))
        col = cli.get_collection("bench")
        col.query(query_embeddings=[qs[0].tolist()], n_results=5)   # 첫 검색
        name = "Chroma"
        def do_search(q): col.query(query_embeddings=[q.tolist()], n_results=5)
    elif db == "faiss":
        import faiss
        index = faiss.read_index(os.path.join(d, "index.faiss"))
        _ids = pickle.load(open(os.path.join(d, "ids.pkl"), "rb"))
        index.search(qs[0].reshape(1, -1), 5)                        # 첫 검색
        name = "FAISS"
        def do_search(q): index.search(q.reshape(1, -1), 5)
    elif db == "qdrant":
        from qdrant_client import QdrantClient
        cli = QdrantClient(path=os.path.join(d, "store"))            # embedded 로드
        cli.query_points("bench", query=qs[0].tolist(), limit=5)     # 첫 검색
        name = "Qdrant(embedded)"
        def do_search(q): cli.query_points("bench", query=q.tolist(), limit=5)
    cold_s = time.perf_counter() - t0

    # ── ② 메모리: 로드 완료 후 RSS 증가분 ──
    gc.collect(); ram = rss_mb() - ram0

    # ── ③ QPS: 워밍업 후 100회 단일 스레드 ──
    for q in qs[1:N_WARMUP]:
        do_search(q)
    lat = []
    for q in qs[N_WARMUP:]:
        t1 = time.perf_counter(); do_search(q)
        lat.append((time.perf_counter() - t1) * 1000)
    lat = np.array(lat)

    print("RESULT " + json.dumps({
        "db": name, "scale": scale,
        "ram_mb": round(ram, 1),
        "cold_start_s": round(cold_s, 3),
        "qps": round(N_QUERY / (lat.sum() / 1000), 1),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
    }, ensure_ascii=False))

# ──────────────────────────────────────────────────────────────
# 드라이버: build → measure 서브프로세스 순차 실행 → CSV/표
# ──────────────────────────────────────────────────────────────
def spawn(*args):
    p = subprocess.run([sys.executable, __file__, *args],
                       capture_output=True, text=True)
    return p.stdout, p.stderr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--phase", choices=["build", "measure"])
    ap.add_argument("--db"); ap.add_argument("--scale", type=int)
    args = ap.parse_args()

    if args.phase == "build":
        build(args.db, args.scale); return
    if args.phase == "measure":
        measure(args.db, args.scale); return

    scales = [1000] if args.dry_run else SCALES
    results = []
    for db in DBS:
        for sc in scales:
            print(f"=== {db} × {sc:,} ===", flush=True)
            out, err = spawn("--phase", "build", "--db", db, "--scale", str(sc))
            if "BUILD_OK" not in out:
                print(f"[build 실패]\n{err[-600:]}"); continue
            out, err = spawn("--phase", "measure", "--db", db, "--scale", str(sc))
            line = [l for l in out.splitlines() if l.startswith("RESULT ")]
            if not line:
                print(f"[measure 실패]\n{err[-600:]}"); continue
            r = json.loads(line[0][7:]); print(r, flush=True); results.append(r)
            shutil.rmtree(wdir(db, sc), ignore_errors=True)  # 디스크 정리

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    # 슬라이드용 표: 규모별 3행 × 지표 3종
    names = ["Chroma", "FAISS", "Qdrant(embedded)"]
    R = {(r["db"], r["scale"]): r for r in results}
    lines = ["| 지표 (규모) | Chroma | FAISS | **Qdrant(embedded)** |", "|---|---|---|---|"]
    for label, key, fmt in [("메모리 사용량 MB", "ram_mb", "{:,}"),
                            ("콜드 스타트 s", "cold_start_s", "{}"),
                            ("QPS (단일 스레드)", "qps", "{:,}")]:
        for sc in scales:
            cells = [fmt.format(R[(n, sc)][key]) if (n, sc) in R else "-" for n in names]
            lines.append(f"| {label} ({sc:,}건) | " + " | ".join(cells) + " |")
    lines += ["",
        "※ Qdrant는 실배포(STEP7/8)와 동일한 embedded 모드로 측정. 서버 모드는 별도 재측정 필요.",
        "※ 콜드 스타트 = 프로세스 재시작 후 디스크 데이터 로드 → 첫 검색 성공까지 (빌드·측정 프로세스 분리).",
        "※ 메모리 = 데이터 로드 완료 후 프로세스 RSS 증가분 (조합마다 새 프로세스로 격리 측정).",
        "※ QPS = 단일 스레드 순차 검색 기준 (워밍업 10회 후 100회). p95는 results_v3.csv 참고.",
    ]
    open(OUT_TABLE, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n완료 → {OUT_CSV}, {OUT_TABLE}")

if __name__ == "__main__":
    main()
