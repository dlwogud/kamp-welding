"""27. 보고서 수치 대조 — 본문 표의 셀을 출처 csv 와 직접 비교한다.

왜 필요한가
    보고서의 각 표에는 출처 파일명이 병기되어 있다(예: `t36_model_by_anomtype.csv`).
    그러나 '파일이 존재하는가' 는 '본문에 옮겨 적은 숫자가 그 파일의 값과 같은가' 를
    보장하지 않는다. 25 번의 진위 검사는 **파일의 신선도**를 보고, 본 스크립트는
    **옮겨 적은 값의 정확성**을 본다. 둘은 서로 다른 종류의 오류를 잡는다.

    실제로 본 과제에서 다음 종류의 오류가 발생했다.
      - 벤치마크가 6유형인데 본문·README·발표자료에 '7유형' 으로 적음
      - 그 값을 5개 문서에 전파한 뒤 '문서 간 일치' 를 확인해 통과 처리함
    **문서 간 일치는 오류를 일관되게 만들 뿐 맞게 만들지 않는다.** 대조의 기준은
    언제나 산출물이어야 한다.

무엇을 하는가
    1) 본문에서 〔표 N〕 ... `파일명.csv` 캡션과 바로 뒤의 Markdown 표를 짝지어 추출한다.
    2) 표의 모든 수치 셀을 모은다.
    3) 출처 csv 의 수치 집합과 비교해, 본문에만 있는 값을 보고한다.
       (본문은 발췌·요약이므로 csv 에만 있는 값은 문제가 아니다. 역방향만 본다.)
    4) 반올림 표기를 허용한다 — csv 값을 본문 셀의 소수 자리수로 맞춰 비교한다.

한계를 명시한다
    본문 표는 열을 줄이거나 단위를 바꾸거나 여러 csv 를 합치기도 한다. 따라서 불일치가
    곧 오류는 아니다. 본 스크립트는 **사람이 확인할 목록을 좁혀 주는 도구**이며,
    '0건' 을 재현성의 증명으로 쓰지 않는다.

    또한 집계 수치(검사 대상 산출물 수, 표·그림 개수, 스크립트 수)의 **정답을
    매니페스트와 파일시스템에서 역산해 표로 남긴다.** 이 수치는 분석이 바뀔 때마다
    흔들리는데 5개 문서에 흩어져 있어 **실제로 세 차례 어긋났다**(이상유형 수,
    스크립트 수, 산출물 수). 문서를 고칠 때는 이 표에서 값을 읽어야 하며,
    문서끼리 베끼면 같은 오류가 반복된다.

    매니페스트의 cited_total 과 보고서 역산값이 어긋나면 **실패 처리**한다. 이는
    추측이 아닌 정확한 검사이고, 보고서가 산출물 인용을 바꾸면 즉시 드러난다.

산출
    t68_report_number_audit   표별 대조 결과
    t69_doc_claim_audit       집계 수치의 정답 (문서를 고칠 때 여기서 읽는다)
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd

from common import AUDIT_ARTIFACTS, EXTERNAL_FILES, OUT, save_table

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "결과보고서_본문.md"
TAB = OUT / "tables"
warnings.filterwarnings("ignore")

NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:e-?\d+)?")
# 본문에만 등장해도 문제가 아닌 값 — 절 번호, 연도, 작업일, 배점 등
IGNORE = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "100", "2020", "11939",
          "1648", "39", "15", "40", "95", "99", "2.4", "2.5", "2.6", "2.7", "2.8",
          "2.9", "2.10", "2.11", "2.12", "3.5", "4.2", "4.3", "4.5", "4.7", "6.5"}


# 검사기 자신의 산출물. 25 와 같은 집합을 쓴다.
SELF_MADE = AUDIT_ARTIFACTS   # common 에 단일 정의


def parse_blocks() -> list[tuple[int, list[str], list[str]]]:
    """(표번호, 출처파일 목록, 표 안의 수치 셀 목록).

    캡션에 csv 가 둘 이상 병기된 표가 있다(예: 표 34). 전부 읽어 합집합과 비교한다.
    하나만 읽으면 다른 파일에서 온 값이 전부 불일치로 잡힌다.
    """
    lines = REPORT.read_text(encoding="utf-8").split("\n")
    out = []
    for i, l in enumerate(lines):
        if not l.startswith("〔표 "):
            continue
        m = re.match(r"^〔표 (\d+)〕", l)
        srcs = [s for s in re.findall(r"`([\w\.]+\.csv)`", l) if s not in SELF_MADE]
        if not m or not srcs:
            continue
        num, src = int(m.group(1)), srcs
        cells = []
        for l2 in lines[i + 1:i + 60]:
            if l2.startswith("〔") or re.match(r"^#{1,4} ", l2):
                break
            if not l2.strip().startswith("|"):
                continue
            if set(l2.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            for c in l2.split("|"):
                c = c.strip().replace("**", "")
                for v in NUM.findall(c):
                    cells.append(v)
        out.append((num, src, cells))
    return out


def csv_numbers(src: str) -> set[str]:
    p = TAB / src
    if not p.exists():
        return set()
    try:
        d = pd.read_csv(p, encoding="utf-8-sig")
    except Exception:
        return set()
    vals = set()
    for v in d.to_numpy().ravel():
        s = str(v)
        for x in NUM.findall(s):
            vals.add(x)
    for c in d.columns:
        for x in NUM.findall(str(c)):
            vals.add(x)
    return vals


def matches(cell: str, pool: set[str]) -> bool:
    """본문 셀 값이 csv 값 집합에 (반올림 허용) 존재하는가."""
    raw = cell.replace(",", "")
    if cell in pool or raw in pool:
        return True
    try:
        x = float(raw)
    except ValueError:
        return True          # 수치로 못 읽으면 판정하지 않는다
    dec = len(raw.split(".")[1]) if "." in raw else 0
    for s in pool:
        try:
            y = float(s.replace(",", ""))
        except ValueError:
            continue
        if round(y, dec) == x or (dec == 0 and abs(y - x) < 0.5):
            return True
        # 백분율 표기 차이 (0.0183 <-> 1.83)
        if dec > 0 and (round(y * 100, dec) == x or round(y / 100, dec) == x):
            return True
        # 만원 환산 (8,352,000원 -> 835만원)
        if abs(y) >= 10000 and round(y / 10000) == round(x):
            return True
        # 지수 표기 반올림 (1.34e-244 -> 1.3e-244)
        if y != 0 and x != 0 and abs(y) < 1e-10 and abs(x) < 1e-10:
            import math
            if abs(math.log10(abs(y)) - math.log10(abs(x))) < 0.05:
                return True
    # 본문 표의 '평균' 행처럼 csv 값에서 계산한 값은 셀 단위로 일치하지 않는다.
    return False


def truth() -> dict:
    """집계 수치의 정답을 매니페스트와 파일시스템에서 역산한다."""
    import glob
    import json
    rep = REPORT.read_text(encoding="utf-8")
    seen = []
    for n in re.findall(r"`((?:t|f)\d+[a-z]?_[\w\.]+\.(?:csv|png))`", rep):
        if n not in seen:
            seen.append(n)
    checked = [n for n in seen if n not in AUDIT_ARTIFACTS]
    ext = [n for n in checked if n in EXTERNAL_FILES]
    mp = ROOT / "outputs" / "run_manifest.json"
    man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    return {
        "검사 대상 산출물": len(checked),
        "그중 부록 A": len(ext),
        "그중 본문": len(checked) - len(ext),
        "매니페스트 cited_total": man.get("cited_total"),
        "표 캡션": len(re.findall(r"^〔표 (\d+)〕", rep, re.M)),
        "그림 캡션": len(re.findall(r"^〔그림 (\d+)〕", rep, re.M)),
        "표 파일": len(glob.glob(str(ROOT / "outputs/tables/*.csv"))),
        "그림 파일": len(glob.glob(str(ROOT / "outputs/figures/*.png"))),
        "src 스크립트": len(glob.glob(str(ROOT / "src/*.py"))),
        "run_all 등록": sum(1 for l in (ROOT / "run_all.sh").read_text(encoding="utf-8").split("\n")
                            if l.startswith("run ") and l[4:5].isdigit()),
    }


def t69_truth(T: dict) -> pd.DataFrame:
    """집계 수치의 정답을 표로 남긴다.

    문서의 각 줄에서 '이 숫자가 어떤 집계인가' 를 정규식으로 추정하는 방식을 시도했으나
    자유 서술에서는 신뢰할 수 없었다(오탐 8건). **추측으로 경고를 만드는 것은
    경고를 무시하게 만들므로 더 나쁘다.** 따라서 추정은 버리고 두 가지만 남긴다.

      (1) 매니페스트의 cited_total 과 보고서에서 역산한 검사 대상 수의 **정확한 일치** 검사
          — 추측이 없다. 보고서가 산출물 인용을 추가/삭제하면 즉시 어긋나므로
            문서 수치가 낡았는지를 이 한 줄이 잡는다. 실제로 자기참조 버그를 잡았다.
      (2) 정답 집계표 — 문서를 고칠 때 **여기서 값을 읽어 쓴다.** 문서에서 베끼지 않는다.
    """
    out = pd.DataFrame([{"집계 항목": k, "정답": v} for k, v in T.items()])
    save_table(out, "t69_doc_claim_audit", index=False)
    return out


def main() -> int:
    print("=" * 70)
    print("  27. 보고서 수치 대조 — 본문 표 vs 출처 csv")
    print("=" * 70)

    blocks = parse_blocks()
    rows, total_unmatched = [], 0
    for num, srcs, cells in blocks:
        pool = set()
        for s in srcs:
            pool |= csv_numbers(s)
        label = " + ".join(srcs)
        if not pool:
            rows.append({"표": num, "출처": label, "본문 수치": len(cells),
                         "출처에 없음": "—", "상태": "출처 파일 없음/읽기 실패"})
            continue
        bad = [c for c in cells if c not in IGNORE and not matches(c, pool)]
        total_unmatched += len(bad)
        rows.append({"표": num, "출처": label, "본문 수치": len(cells),
                     "출처에 없음": len(bad),
                     "상태": "일치" if not bad else ", ".join(sorted(set(bad))[:8])})
    df = pd.DataFrame(rows)
    save_table(df, "t68_report_number_audit", index=False)

    ok = df[df["상태"] == "일치"]
    print(f"\n  출처 csv 가 병기된 본문 표: {len(df)} 개")
    print(f"  전부 일치: {len(ok)} 개 / 확인 필요: {len(df) - len(ok)} 개")
    print(f"  출처에서 찾지 못한 수치 셀: {total_unmatched} 개")

    need = df[df["상태"] != "일치"]
    if len(need):
        print("\n  ── 사람이 확인할 목록 ──")
        for _, r in need.iterrows():
            print(f"    표 {r['표']:>2}  {r['출처']:<38} {r['상태']}")
        print("\n  주의: 본문 표는 발췌·단위변환·복수 출처 합산이 있으므로")
        print("  불일치가 곧 오류는 아니다. 위 목록은 확인 범위를 좁힌 것이다.")

    print("\n── 문서 집계 주장 대조 (매니페스트·파일시스템에서 역산) ──")
    T = truth()
    for k, v in T.items():
        print(f"    {k:<22} {v}")
    if T["매니페스트 cited_total"] not in (None, T["검사 대상 산출물"]):
        print(f"\n  ✗ 매니페스트({T['매니페스트 cited_total']})와 보고서 인용 역산"
              f"({T['검사 대상 산출물']})이 다르다. 전체 재실행이 필요하다.")
        return 1
    t69_truth(T)
    print("\n  매니페스트와 보고서 역산이 일치한다. 문서의 집계 수치는 위 표에서 읽어 쓴다.")
    print("  (문서 각 줄의 숫자를 정규식으로 추정해 경고하는 방식은 오탐이 많아 제거했다.")
    print("   추측으로 만든 경고는 경고를 무시하게 만들어 더 나쁘다.)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
