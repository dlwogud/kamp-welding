"""공통 로딩 · 상수 · 그림 설정.

데이터 구조상의 사실 (EDA로 확인, 보고서 '데이터 이해' 절의 근거):
  - Raw data   : 11,939행. 용접점별 공정조건. 설비 1종(Spot-01) · 품목 1종(65235-25800).
  - result     : 23행. 8개 작업일 x 3개 불량유형의 '건수' 집계. 총 39건.
                 -> 용접점(행) 단위 불량 라벨은 존재하지 않는다. 라벨 유효표본 = 8일.
  - working time 은 날짜 단위(시:분 없음) -> 작업일보다 잘게 쪼갤 수 없다.
  - Thickness 1/2 는 전 행 0.7 고정(분산 0) -> 설명변수로 사용 불가.
  - 2020-03-27 은 Raw data 에만 있고 result 에 라벨이 없다 -> 예측 대상(테스트셋).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "Welding Data Set_01.xlsx"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
for _p in (FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)

# 공정변수 (두께는 분산 0 이므로 제외)
PROC = ["weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"]
PROC_KR = {
    "weld force(bar)": "가압력(bar)",
    "weld current(kA)": "전류(kA)",
    "weld Voltage(v)": "전압(V)",
    "weld time(ms)": "통전시간(ms)",
}

# 'data set' 시트에 명시된 수집(사양) 범위
SPEC = {
    "Thickness 1(mm)": (0.3, 2.3),
    "Thickness 2(mm)": (0.3, 2.3),
    "weld force(bar)": (1.0, 12.0),
    "weld current(kA)": (12.0, 18.0),
    "weld Voltage(v)": (1.5, 3.5),
    "weld time(ms)": (30.0, 120.0),
}

# 라벨 없는 예측 대상일
TEST_DATE = dt.date(2020, 3, 27)

# 레시피 경계. 변동일 가압력 분포에서 확인된 모드:
#   주모드 ~2.3 (2.16-2.50) / 전이 3-5 / 고압 5-9 (피크 7.90) / 극고압 9+ (전압 2.51로 강하)
RECIPE_EDGES = [0.0, 3.0, 5.0, 9.0, np.inf]
RECIPE_NAMES = ["A_주모드(~3bar)", "T_전이(3-5bar)", "B_고압(5-9bar)", "C_극고압(9bar+)"]


def load_raw() -> pd.DataFrame:
    """Raw data 시트. 파생열 d(작업일), recipe 추가."""
    df = pd.read_excel(RAW_XLSX, sheet_name="Raw data")
    df.columns = [c.strip() for c in df.columns]
    df["d"] = pd.to_datetime(df["working time"]).dt.date
    df["recipe"] = pd.cut(
        df["weld force(bar)"], bins=RECIPE_EDGES, labels=RECIPE_NAMES, right=True
    )
    return df


def load_dup_index() -> set[tuple]:
    """06 이 확정한 복제 구간의 (작업일, idx) 집합.

    06 을 아직 실행하지 않았으면 빈 집합을 돌려준다(호출부에서 경고).
    복제 구간은 사전 기준 없이 '네 작업일에서 연속으로 일치하는 최장 구간'으로
    탐색한 결과이며, 800점 x 4일 = 3,200점(전체의 26.8%)이다.
    """
    f = OUT / "duplicated_block_index.csv"
    if not f.exists():
        return set()
    d = pd.read_csv(f, encoding="utf-8-sig")
    d["d"] = pd.to_datetime(d["d"]).dt.date
    return set(zip(d["d"], d["idx"]))


def mark_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """`is_dup` 열을 붙인다. True 면 복제 구간에 속한 행."""
    key = load_dup_index()
    out = df.copy()
    if not key:
        out["is_dup"] = False
        print("  [주의] outputs/duplicated_block_index.csv 가 없다. "
              "06_data_integrity.py 를 먼저 실행할 것.")
    else:
        out["is_dup"] = [(d, i) in key for d, i in zip(out["d"], out["idx"])]
    return out


def load_genuine(stable_only: bool = False) -> pd.DataFrame:
    """복제 구간을 제외한 실제 생산 데이터.

    stable_only=True 면 안정일 5일만 반환한다(모형 적합용).
    2020-04-03 은 작업일 전체가 복제 구간이므로 결과에서 완전히 사라진다.
    """
    df = mark_duplicates(load_raw())
    df = df[~df["is_dup"]]
    if stable_only:
        df = df[df["d"].astype(str).isin(STABLE_DAYS)]
    return df


# 검사기(25·27)가 스스로 만드는 산출물. 진위·대조 검사 대상에서 제외한다.
# 검사기의 출력을 '검사가 통과했다' 는 증거로 쓸 수 없고, 보고서가 이들을
# 인용하는 순간 검사 대상 개수가 바뀌는 자기참조가 생긴다.
AUDIT_ARTIFACTS = {
    "t63_run_manifest.csv",
    "t68_report_number_audit.csv",
    "t69_doc_claim_audit.csv",
}


# 보고서가 인용하는 부록 A(외부 검증) 산출물의 정확한 파일명.
# 25(진위 검사)와 27(수치 대조)이 함께 쓴다. 두 곳에 복제하면 또 어긋나므로
# 여기 한 곳에만 둔다. 접두어로 판정하면 f32_cost_breakeven.png 같은 본문
# 산출물이 부록 A 로 잘못 분류되어 검사를 빠져나간다(실제로 그런 버그가 있었다).
EXTERNAL_FILES = {
    "t76_external_press.csv",
    "t77_external_conformal.csv",
    "t78_external_model_ranking.csv",
    "f36_external_press.png",
    "t80_mold_supervised_ceiling.csv",
    "t81_mold_llp.csv",
    "f37_external_mold.png",
}


# 가압력 이질 운전의 혼입이 없는 작업일 (01 에서 데이터로 판정)
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]


def load_result() -> pd.DataFrame:
    """result 시트. 불량유형명을 채워 넣는다(원본은 첫 3행에만 기재)."""
    df = pd.read_excel(RAW_XLSX, sheet_name="result")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Unnamed: 6": "defect_name"})
    df["d"] = pd.to_datetime(df["working time"]).dt.date
    # 불량유형 1/2/3 <-> 파임불량/용접부족/크랙발생 (원본 상단 3행 기재분을 전체로 전파)
    name_map = (
        df.dropna(subset=["defect_name"]).set_index("defect type")["defect_name"].to_dict()
    )
    df["defect_name"] = df["defect type"].map(name_map)
    return df


def daily_label() -> pd.Series:
    """작업일별 총 불량건수 (라벨 있는 8일)."""
    res = load_result()
    return res.groupby("d")["defect"].sum().rename("defects")


def daily_label_by_type() -> pd.DataFrame:
    """작업일 x 불량유형 불량건수."""
    res = load_result()
    return res.pivot_table(index="d", columns="defect_name", values="defect", aggfunc="sum")


def setup_mpl():
    """한글 폰트 포함 matplotlib 설정. 폰트가 없으면 영문 라벨로 떨어진다."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    matplotlib.use("Agg")
    for cand in ("AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Malgun Gothic"):
        if any(f.name == cand for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams.update(
        {
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return plt


def save_table(df: pd.DataFrame, name: str, **kw):
    """표를 csv(기계판독용) + md(보고서 붙여넣기용) 로 동시 저장."""
    path = TAB / f"{name}.csv"
    df.to_csv(path, encoding="utf-8-sig", **kw)
    try:
        (TAB / f"{name}.md").write_text(df.to_markdown(), encoding="utf-8")
    except Exception as e:
        print(f"Warning: failed to write Markdown table {name}: {e}")
    print(f"  saved -> outputs/tables/{name}.csv")
    return path
