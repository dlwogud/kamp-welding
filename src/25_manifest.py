"""25. 산출물 진위 검사 — 보고서의 모든 수치가 이번 실행에서 나왔는지 확인한다.

왜 필요한가
    outputs/ 에는 과거 실행의 파일이 남을 수 있다. 특히 부록 A(외부 검증)는 외부 데이터가
    없으면 건너뛰므로, 아무 조치가 없으면 **"실행은 성공했는데 일부 표는 과거 파일"** 인
    상태가 가능하다. 심사위원이 새 환경에서 재현할 때 이 구분이 보이지 않으면 안 된다.

무엇을 하는가
    1) 보고서 본문이 인용한 모든 파일명(tN_*.csv / fN_*.png)을 추출한다.
       → 기대 산출물 목록을 사람이 관리하지 않고 **보고서에서 직접 유도**한다.
    2) 각 파일이 (a) 존재하는가 (b) 이번 실행에서 생성되었는가(mtime >= .run_start)를 본다.
    3) SHA-256 해시를 기록한다.
    4) 결과를 outputs/run_manifest.json 과 표로 남기고, 누락·과거파일이 있으면
       0 이 아닌 코드로 종료한다. 즉 **종료 조건**이다.

상태 정의
    FRESH    이번 실행이 생성 — 보고서 수치의 근거로 유효
    STALE    존재하지만 이번 실행이 생성하지 않음 — 과거 파일
    MISSING  없음
    SKIPPED  부록 A 산출물이며 외부 데이터가 없어 의도적으로 건너뜀 (본문 결론 무영향)

    .run_start 가 없으면(단독 실행) 신선도를 판정할 근거가 없으므로 상태를 만들지 않고
    즉시 실패 처리한다. '판정 불가'를 'PASS' 로 보고하는 것이 가장 위험하기 때문이다.

    검사기(25·27)가 만드는 산출물 3개는 검사 대상에서 제외한다(common.AUDIT_ARTIFACTS).
    포함하면 신규 실행에서 분류 시점에 아직 없어 MISSING 으로 잡히고, 보고서가 이들을
    인용하는 순간 검사 대상 개수가 바뀌는 자기참조가 생긴다.

임시폴더 교체 방식을 쓰지 않은 이유
    outputs/ 를 임시폴더에 만들고 성공 후 교체하는 방식도 가능하나, 본 파이프라인은
    스크립트 간 산출물 의존(예: 16 -> 17)이 있어 중간 단계에서 실제 경로가 필요하다.
    대신 **상태를 숨기지 않고 드러내는** 방식을 택했다. 부분 실패 시 run_all.sh 는
    set -e 로 중단되고, 그때 남은 혼합 상태는 본 스크립트가 STALE 로 보고한다.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

from common import AUDIT_ARTIFACTS, EXTERNAL_FILES, OUT, save_table

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "결과보고서_본문.md"
MANIFEST = OUT / "run_manifest.json"
RUN_START = OUT / ".run_start"

# 부록 A 산출물 목록은 common.EXTERNAL_FILES 에 단일 정의되어 있다.


def run_start_ts() -> float | None:
    """run_all.sh 가 기록한 실행 시작 시각. 단독 실행 시에는 없다."""
    if not RUN_START.exists():
        return None
    try:
        return float(RUN_START.read_text().strip())
    except ValueError:
        return None


def cited_files() -> list[str]:
    """보고서가 인용한 산출물 파일명을 추출한다. 기대 목록의 단일 출처."""
    if not REPORT.exists():
        return []
    text = REPORT.read_text(encoding="utf-8")
    names = re.findall(r"`((?:t|f)\d+[a-z]?_[\w\.]+\.(?:csv|png))`", text)
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def locate(name: str) -> Path:
    return (OUT / "figures" / name) if name.endswith(".png") else (OUT / "tables" / name)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(name: str, start: float) -> dict:
    """main() 이 start is None 을 미리 걸러내므로 여기서는 float 만 받는다."""
    p = locate(name)
    is_ext = name in EXTERNAL_FILES
    if not p.exists():
        return {"file": name, "status": "SKIPPED" if is_ext else "MISSING",
                "bytes": None, "sha256": None, "mtime": None, "appendix": is_ext}
    st = p.stat()
    if st.st_mtime >= start - 1:
        status = "FRESH"
    else:
        status = "SKIPPED" if is_ext else "STALE"
    return {"file": name, "status": status, "bytes": st.st_size,
            "sha256": sha256(p), "mtime": int(st.st_mtime), "appendix": is_ext}


def main() -> int:
    print("=" * 70)
    print("  25. 산출물 진위 검사")
    print("=" * 70)

    start = run_start_ts()
    if start is None:
        print("\n  오류: outputs/.run_start 가 없습니다. 신선도를 검증할 수 없습니다.")
        print("  전체 검사는 `bash run_all.sh` 로 실행하십시오.")
        return 1

    names = [n for n in cited_files() if n not in AUDIT_ARTIFACTS]
    if not names:
        print("\n  보고서를 찾을 수 없어 인용 목록을 만들 수 없습니다. 검사를 실패 처리합니다.")
        print(f"  (기대 경로: {REPORT})")
        return 1

    recs = [classify(n, start) for n in names]
    df = pd.DataFrame(recs)
    counts = df["status"].value_counts().to_dict()

    print(f"\n  보고서가 인용한 산출물: {len(recs)} 개")
    for k in ("FRESH", "SKIPPED", "STALE", "MISSING", "UNKNOWN"):
        if k in counts:
            print(f"    {k:8s} {counts[k]:3d}")
    bad = df[df["status"].isin(["STALE", "MISSING", "UNKNOWN"])]
    if len(bad):
        print("\n  ── 문제가 있는 산출물 ──")
        for _, r in bad.iterrows():
            why = ("이번 실행이 생성하지 않은 과거 파일" if r["status"] == "STALE"
                   else "파일이 없음")
            print(f"    [{r['status']}] {r['file']} — {why}")

    skipped = df[df["status"] == "SKIPPED"]
    if len(skipped):
        print(f"\n  부록 A(외부 검증) 산출물 {len(skipped)} 개가 SKIPPED 입니다.")
        print("  외부 데이터가 없을 때의 정상 동작이며, 제1~6장 결론에는 영향이 없습니다.")
        print("  해당 표·그림의 수치는 이번 실행에서 재생성되지 않았습니다.")

    save_table(df, "t63_run_manifest", index=False)
    MANIFEST.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_start": start,
        "report": str(REPORT.relative_to(ROOT)),
        "cited_total": len(recs),
        "counts": counts,
        "verdict": "PASS" if len(bad) == 0 else "FAIL",
        "artifacts": recs,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  saved -> {MANIFEST.relative_to(ROOT)}")

    print("\n" + "=" * 70)
    if len(bad) == 0:
        print("  판정: PASS — 보고서가 인용한 본문 산출물 전부가 이번 실행의 결과입니다.")
        print("=" * 70)
        return 0
    print(f"  판정: FAIL — 과거파일 {counts.get('STALE', 0)} 개, 누락 {counts.get('MISSING', 0)} 개.")
    print("  보고서 수치의 근거가 이번 실행과 일치하지 않습니다.")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
