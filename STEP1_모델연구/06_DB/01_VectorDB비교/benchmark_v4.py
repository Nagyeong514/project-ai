# -*- coding: utf-8 -*-
"""
benchmark_v4.py — Vector DB 비교 벤치 (확정 지표 4종, 측정 쿼리 100회)

지표:
  ① Insert Time (s)   — 벌크 삽입 전체 소요 시간 (Qdrant는 인덱싱 대기 포함)
  ② Latency (ms)      — 검색 p50 / p95 (워밍업 10회 후 100회 측정)
  ③ Cold Start (s)    — 새 프로세스에서 디스크 데이터 로드 → 첫 검색 성공까지
  ④ QPS               — 단일 스레드 처리량 (= 100회 / 총 소요시간)

측정 구조:
  [build 프로세스]  삽입(①측정) → persist → insert_time 기록 → 종료
  [measure 프로세스] 새 프로세스 기동 → ③콜드 스타트 → ②④ latency/QPS
  DB×규모 조합마다 프로세스를 새로 띄워 격리 측정.

조건(기존 실험 동일): 시드 42, L2 정규화 cosine, Chroma 텔레메트리 오프,
  Qdrant embedded 모드(실배포 STEP7/8과 동일).

실행: python benchmark_v4.py [--dry-run] [--n-query 100]
산출: results_v4.csv, slide_table_v4.md
"""
import argparse, csv, json, os, pickle, shutil, subprocess, sys, time
import numpy as np

SEED = 42
# 2026-07-12 지시 반영: 10만 건 스케일 취소 → 100건 추가(실서비스 규모 대표: 현 DB 수십 건).
SCALES = [100, 1000, 10000]
DBS = ["chroma", "faiss", "qdrant"]
DIM = 1024                    # 기존 실험(bge-m3, benchmark.py)과 동일 차원 (원본 384→1024)
N_WARMUP = 10
OUT_CSV, OUT_TABLE = "results_v4.csv", "slide_table_v4.md"

def synthetic(n, dim):
    rng = np.random.default_rng(SEED)
    v = rng.standard_normal((n, dim)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return [f"syn_{i:06d}" for i in range(n)], v

def queries(n, dim):
    rng = np.random.default_rng(SEED + 7)
    q = rng.standard_normal((n, dim)).astype("float32")
    return q / np.linalg.norm(q, axis=1, keepdims=True)

def wdir(db, scale):
    return f"_bench_{db}_{scale}"

# ── build: 삽입 시간 측정 + persist ─────────────────────────────
def build(db, scale):
    ids, vecs = synthetic(scale, DIM)
    d = wdir(db, scale)
    shutil.rmtree(d, ignore_errors=True); os.makedirs(d)

    t0 = time.perf_counter()
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
        cli = QdrantClient(path=os.path.join(d, "store"))  # embedded
        cli.create_collection("bench", vectors_config=VectorParams(
            size=DIM, distance=Distance.COSINE))
        pts = [PointStruct(id=i, vector=v.tolist()) for i, v in enumerate(vecs)]
        for s in range(0, scale, 2000):
            cli.upsert("bench", pts[s:s+2000], wait=True)  # wait=인덱싱 반영 포함
        cli.close()  # 파일 락 해제
    insert_s = time.perf_counter() - t0

    json.dump({"insert_s": round(insert_s, 2)},
              open(os.path.join(d, "build_info.json"), "w"))
    print("BUILD_OK")

# ── measure: 콜드 스타트 → latency/QPS ──────────────────────────
def measure(db, scale, n_query):
    d = wdir(db, scale)
    qs = queries(N_WARMUP + n_query, DIM)

    t0 = time.perf_counter()
    if db == "chroma":
        import chromadb
        from chromadb.config import Settings
        cli = chromadb.PersistentClient(path=os.path.join(d, "store"),
                                        settings=Settings(anonymized_telemetry=False))
        col = cli.get_collection("bench")
        col.query(query_embeddings=[qs[0].tolist()], n_results=5)
        name = "Chroma"
        def do_search(q): col.query(query_embeddings=[q.tolist()], n_results=5)
    elif db == "faiss":
        import faiss
        index = faiss.read_index(os.path.join(d, "index.faiss"))
        _ids = pickle.load(open(os.path.join(d, "ids.pkl"), "rb"))
        index.search(qs[0].reshape(1, -1), 5)
        name = "FAISS"
        def do_search(q): index.search(q.reshape(1, -1), 5)
    elif db == "qdrant":
        from qdrant_client import QdrantClient
        cli = QdrantClient(path=os.path.join(d, "store"))
        cli.query_points("bench", query=qs[0].tolist(), limit=5)
        name = "Qdrant(embedded)"
        def do_search(q): cli.query_points("bench", query=q.tolist(), limit=5)
    cold_s = time.perf_counter() - t0            # ③ 콜드 스타트

    for q in qs[1:N_WARMUP]:                     # 워밍업
        do_search(q)
    lat = []
    for q in qs[N_WARMUP:]:                      # ②④ 측정 100회
        t1 = time.perf_counter(); do_search(q)
        lat.append((time.perf_counter() - t1) * 1000)
    lat = np.array(lat)

    insert_s = json.load(open(os.path.join(d, "build_info.json")))["insert_s"]
    print("RESULT " + json.dumps({
        "db": name, "scale": scale,
        "insert_s": insert_s,                                    # ①
        "p50_ms": round(float(np.percentile(lat, 50)), 2),       # ②
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "cold_start_s": round(cold_s, 3),                        # ③
        "qps": round(n_query / (lat.sum() / 1000), 1),           # ④
    }, ensure_ascii=False))

# ── 드라이버 ────────────────────────────────────────────────────
def spawn(*args):
    p = subprocess.run([sys.executable, __file__, *args],
                       capture_output=True, text=True)
    return p.stdout, p.stderr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-query", type=int, default=100, help="측정 쿼리 수 (기본 100)")
    ap.add_argument("--scales", type=str, default=None,
                    help="측정할 스케일 콤마 목록(예: 100000). 기존 results_v4.csv와 병합됨")
    ap.add_argument("--phase", choices=["build", "measure"])
    ap.add_argument("--db"); ap.add_argument("--scale", type=int)
    args = ap.parse_args()

    if args.phase == "build":
        build(args.db, args.scale); return
    if args.phase == "measure":
        measure(args.db, args.scale, args.n_query); return

    scales = [1000] if args.dry_run else (
        [int(s) for s in args.scales.split(",")] if args.scales else SCALES)
    results = []
    # 부분 재측정 병합: 기존 results_v4.csv에서 이번에 안 도는 스케일의 행을 승계
    if os.path.exists(OUT_CSV):
        for r in csv.DictReader(open(OUT_CSV, encoding="utf-8")):
            if int(r["scale"]) not in scales:
                results.append({k: (float(v) if k not in ("db", "scale") else
                                    int(v) if k == "scale" else v) for k, v in r.items()})
    for db in DBS:
        for sc in scales:
            print(f"=== {db} × {sc:,} ===", flush=True)
            out, err = spawn("--phase", "build", "--db", db, "--scale", str(sc))
            if "BUILD_OK" not in out:
                print(f"[build 실패]\n{err[-600:]}"); continue
            out, err = spawn("--phase", "measure", "--db", db,
                             "--scale", str(sc), "--n-query", str(args.n_query))
            line = [l for l in out.splitlines() if l.startswith("RESULT ")]
            if not line:
                print(f"[measure 실패]\n{err[-600:]}"); continue
            r = json.loads(line[0][7:]); print(r, flush=True); results.append(r)
            shutil.rmtree(wdir(db, sc), ignore_errors=True)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)

    names = ["Chroma", "FAISS", "Qdrant(embedded)"]
    R = {(r["db"], int(r["scale"])): r for r in results}
    scales = sorted({int(r["scale"]) for r in results})  # 병합 후 전체 스케일로 표 재구성
    lines = ["| 지표 (규모) | Chroma | FAISS | **Qdrant(embedded)** |", "|---|---|---|---|"]
    for label, key in [("Insert Time (s)", "insert_s"),
                       ("Latency p50 / p95 (ms)", None),
                       ("Cold Start (s)", "cold_start_s"),
                       ("QPS", "qps")]:
        for sc in scales:
            if key is None:  # latency는 p50/p95 병기
                cells = [f'{R[(n,sc)]["p50_ms"]} / {R[(n,sc)]["p95_ms"]}'
                         if (n, sc) in R else "-" for n in names]
            else:
                cells = [f'{R[(n,sc)][key]:,}' if (n, sc) in R else "-" for n in names]
            lines.append(f"| {label} ({sc:,}건) | " + " | ".join(cells) + " |")
    lines += ["",
        "※ Qdrant는 실배포(STEP7/8)와 동일한 embedded 모드로 측정. 서버 모드는 별도 재측정 필요.",
        "※ Insert Time: 벌크 삽입 전체 소요 (Qdrant는 wait=True로 인덱싱 반영 포함).",
        "※ Latency/QPS: 워밍업 10회 후 측정 쿼리 100회, 단일 스레드.",
        "※ Cold Start: 프로세스 재시작 후 디스크 로드 → 첫 검색 성공까지 (빌드·측정 프로세스 분리).",
    ]
    open(OUT_TABLE, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n완료 → {OUT_CSV}, {OUT_TABLE}")

if __name__ == "__main__":
    main()
