"""01. 데이터 구조 진단.

이 스크립트의 목적은 '좋은 성능'이 아니라 '라벨 구조를 밝히는 것'이다.
산출물은 결과보고서의 데이터 이해 절과, 문제 재정의의 근거가 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    FIG,
    PROC,
    PROC_KR,
    RECIPE_NAMES,
    SPEC,
    TEST_DATE,
    daily_label,
    daily_label_by_type,
    load_raw,
    load_result,
    save_table,
    setup_mpl,
)

plt = setup_mpl()
VAR_DAYS: list = []  # 고압모드 혼입일. 아래에서 데이터로 판정한다.


def t1_structure(raw: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """표1. 두 시트의 단위 불일치 = 이 과제의 핵심 제약."""
    rows = [
        ("Raw data", "용접점", len(raw), raw["d"].nunique(), "공정조건 4종(두께 2종은 분산 0)"),
        ("result", "작업일 x 불량유형", len(res), res["d"].nunique(), "불량 '건수' 집계(0~3)"),
    ]
    t = pd.DataFrame(rows, columns=["시트", "관측단위", "행 수", "작업일 수", "내용"])
    save_table(t, "t1_sheet_structure", index=False)
    return t


def t2_label_reality(raw: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """표2. 라벨의 실제 해상도. '11,939행 이진분류 불가'의 수치적 근거."""
    dl = daily_label()
    t = pd.DataFrame(
        {
            "항목": [
                "Raw data 용접점 수",
                "불량 라벨 총 건수",
                "라벨이 부여된 관측단위",
                "라벨 유효 표본 수",
                "설비 종류",
                "품목 종류",
                "소재 두께 고유값",
                "working time 해상도",
                "라벨 없는 작업일",
            ],
            "값": [
                f"{len(raw):,}",
                f"{int(res['defect'].sum())} 건",
                "작업일 (용접점 아님)",
                f"{res['d'].nunique()} 일",
                f"{raw['Machine_Name'].nunique()} 종 ({raw['Machine_Name'].iloc[0]})",
                f"{raw['Item No'].nunique()} 종 ({raw['Item No'].iloc[0]})",
                f"{raw['Thickness 1(mm)'].nunique()} 개 (0.7mm 고정, 분산 0)",
                "날짜 (시:분 없음)",
                f"{TEST_DATE} ({(raw['d'] == TEST_DATE).sum():,} 점) -> 예측 대상",
            ],
            "분석상 의미": [
                "설명변수는 충분",
                "극소수",
                "용접점 단위 지도학습 불가",
                "복잡한 모델 적용 불가 / 검정력 부재",
                "설비 간 비교 불가",
                "품목 간 비교 불가",
                "두께를 설명변수로 쓸 수 없음",
                "작업일 내 시간분해 불가",
                "테스트데이터 예측결과로 제출",
            ],
        }
    )
    save_table(t, "t2_label_reality", index=False)
    return t


def t3_quality(raw: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """표3. 변수별 사양범위 대비 실제 운전범위 + 품질 이슈."""
    rows = []
    for c in SPEC:
        lo, hi = SPEC[c]
        v = raw[c]
        rows.append(
            {
                "변수": c,
                "사양범위": f"{lo:g} ~ {hi:g}",
                "실제범위": f"{v.min():g} ~ {v.max():g}",
                "사양폭 점유율(%)": round((v.max() - v.min()) / (hi - lo) * 100, 1),
                "평균": round(v.mean(), 4),
                "표준편차": round(v.std(), 4),
                "고유값 수": v.nunique(),
                "사양이탈 건수": int(((v < lo) | (v > hi)).sum()),
                "결측": int(v.isna().sum()),
            }
        )
    t = pd.DataFrame(rows)
    save_table(t, "t3_spec_vs_actual", index=False)

    dup = int(raw.drop(columns=["idx", "recipe"]).duplicated().sum())
    # result 시트 완전성: 작업일마다 불량유형 3종이 모두 기재되어 있어야 23행이 아니라 24행이다.
    cnt = res.groupby("d")["defect type"].nunique()
    miss = cnt[cnt < 3]
    miss_txt = (
        "없음"
        if miss.empty
        else "; ".join(
            f"{d} (유형 {sorted(set([1, 2, 3]) - set(res.loc[res['d'] == d, 'defect type']))} 미기재)"
            for d in miss.index
        )
    )
    q = pd.DataFrame(
        {
            "품질 이슈": [
                "사양범위 이탈",
                "결측치(Raw data)",
                "완전중복행(idx 제외)",
                "통전시간 고유값",
                "두께 분산",
                "result 시트 누락",
            ],
            "관측": [
                "0 건 (전 변수)",
                "0 건",
                f"{dup:,} 건 ({dup / len(raw) * 100:.1f}%)",
                f"{raw['weld time(ms)'].nunique()} 개 "
                f"({', '.join(f'{v:g}' for v in sorted(raw['weld time(ms)'].unique()))})",
                "0 (0.7mm 고정)",
                miss_txt,
            ],
            "해석": [
                "사양 기준으로는 전 구간 정상 -> 사양이탈 룰은 불량 판별에 무력",
                "보간 불필요",
                "센서 분해능 한계. 중복은 제거하지 않음(실제 생산 이력)",
                "사실상 이산변수. 연속 가정 모델에 부적합",
                "지문의 '소재 두께 변화'는 본 데이터에 존재하지 않음",
                "해당일 불량건수가 과소집계일 수 있음 -> 민감도 분석에서 별도 확인",
            ],
        }
    )
    save_table(q, "t3b_quality_issues", index=False)
    return t


def t4_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """표4. 작업일 요약. 고압모드 혼입일 판정의 근거."""
    global VAR_DAYS
    g = raw.groupby("d")
    t = pd.DataFrame(
        {
            "용접점수": g.size(),
            "가압력_평균": g["weld force(bar)"].mean(),
            "가압력_표준편차": g["weld force(bar)"].std(),
            "가압력_p95": g["weld force(bar)"].quantile(0.95),
            "가압력_최대": g["weld force(bar)"].max(),
            "고압모드_비율(%)": g["weld force(bar)"].apply(lambda s: (s > 3).mean() * 100),
            "전류_평균": g["weld current(kA)"].mean(),
            "전류_표준편차": g["weld current(kA)"].std(),
            "전압_평균": g["weld Voltage(v)"].mean(),
            "전압_표준편차": g["weld Voltage(v)"].std(),
            "통전시간_표준편차": g["weld time(ms)"].std(),
        }
    )
    t = t.join(daily_label())
    t["불량률(건/1000점)"] = t["defects"] / t["용접점수"] * 1000
    # 혼입일 판정: 가압력 3bar 초과 용접점이 1% 이상인 날
    t["운전모드"] = np.where(t["고압모드_비율(%)"] >= 1.0, "혼입일", "안정일")
    VAR_DAYS = t.index[t["운전모드"] == "혼입일"].tolist()
    t = t.round(3)
    save_table(t, "t4_daily_summary")
    return t


def t5_defect_type() -> pd.DataFrame:
    """표5. 불량유형 x 작업일 교차표."""
    t = daily_label_by_type()
    t.loc["합계"] = t.sum()
    t["합계"] = t.sum(axis=1)
    save_table(t, "t5_defect_by_type")
    return t


def f1_daily_force(raw: pd.DataFrame, daily: pd.DataFrame):
    """그림1. 작업일별 가압력 분포 — 혼입일/안정일 이원화를 한 장으로."""
    days = sorted(raw["d"].unique())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    data = [raw.loc[raw["d"] == dd, "weld force(bar)"].values for dd in days]
    colors = ["#c0392b" if daily.loc[dd, "운전모드"] == "혼입일" else "#2e86c1" for dd in days]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, flierprops=dict(markersize=2, alpha=0.3))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for med in bp["medians"]:
        med.set_color("black")
    ax.axhline(3.0, ls="--", lw=1, color="gray")
    ax.text(0.15, 3.15, "레시피 경계 3bar", fontsize=8, color="gray")
    lbl = []
    for dd in days:
        mode = "혼입" if daily.loc[dd, "운전모드"] == "혼입일" else "안정"
        tag = "\n(예측대상)" if dd == TEST_DATE else ""
        lbl.append(f"{dd.strftime('%m-%d')}\n{mode}{tag}")
    ax.set_xticklabels(lbl, fontsize=8)
    for tick, dd in zip(ax.get_xticklabels(), days):
        if dd == TEST_DATE:
            tick.set_color("#c0392b")
            tick.set_fontweight("bold")
    ax.set_ylabel("가압력 (bar)")
    ax.set_title("작업일별 가압력 분포 — 단일 품목·단일 두께인데 운전모드가 이원화되어 있다", fontsize=11)
    fig.savefig(FIG / "f1_daily_force_box.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f1_daily_force_box.png")


def f2_recipe_profile(raw: pd.DataFrame):
    """그림2. 레시피별 공정변수 프로파일 — 9bar 이상에서 전압이 강하한다."""
    prof = raw.groupby("recipe", observed=True)[PROC].agg(["mean", "std", "size"])
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    present = [r for r in RECIPE_NAMES if r in prof.index]
    x = np.arange(len(present))
    for ax, c in zip(axes, PROC):
        m = prof.loc[present, (c, "mean")].values
        s = prof.loc[present, (c, "std")].fillna(0).values
        ax.errorbar(x, m, yerr=s, fmt="o-", capsize=4, color="#1f4e79", ms=5)
        ax.set_xticks(x)
        ax.set_xticklabels([p.split("_")[0] for p in present], fontsize=9)
        ax.set_title(PROC_KR[c], fontsize=10)
        ax.set_xlabel("레시피")
    fig.suptitle(
        "레시피별 공정변수 (평균 ± 표준편차) — 가압력 상승에 전류 상승이 동반되고, C(9bar+)에서 전압이 강하한다",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(FIG / "f2_recipe_profile.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f2_recipe_profile.png")

    out = prof.loc[present].round(4)
    out.columns = ["_".join(c) for c in out.columns]
    save_table(out, "t6_recipe_profile")


def f3_label_resolution(raw: pd.DataFrame, daily: pd.DataFrame):
    """그림3. 설명변수 11,939 vs 라벨 8 — 과제 제약을 시각적으로 못박는다."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    days = sorted(raw["d"].unique())
    a1.bar([d.strftime("%m-%d") for d in days], [(raw["d"] == d).sum() for d in days], color="#2e86c1")
    a1.set_title(f"Raw data: 용접점 {len(raw):,}개 (설명변수)", fontsize=10)
    a1.set_ylabel("용접점 수")
    a1.tick_params(axis="x", labelsize=8, rotation=45)

    lab = daily.dropna(subset=["defects"])
    a2.bar([d.strftime("%m-%d") for d in lab.index], lab["defects"].values, color="#c0392b")
    a2.set_title(f"result: 불량 {int(lab['defects'].sum())}건 / 작업일 {len(lab)}개 (라벨)", fontsize=10)
    a2.set_ylabel("불량 건수")
    a2.tick_params(axis="x", labelsize=8, rotation=45)
    a2.set_ylim(0, 9)
    fig.suptitle("라벨 해상도 불일치: 용접점 단위 지도학습은 성립하지 않는다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f3_label_resolution.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f3_label_resolution.png")


def main():
    raw, res = load_raw(), load_result()
    print("\n[01] 데이터 구조 진단")
    print("\n-- 표1 시트 구조")
    print(t1_structure(raw, res).to_string(index=False))
    print("\n-- 표2 라벨의 실제 해상도")
    print(t2_label_reality(raw, res).to_string(index=False))
    print("\n-- 표3 사양범위 대비 실제 운전범위")
    print(t3_quality(raw, res).to_string(index=False))
    print("\n-- 표4 작업일 요약")
    daily = t4_daily(raw)
    print(daily.to_string())
    print(f"\n   혼입일 판정: {[str(d) for d in VAR_DAYS]}")
    print("\n-- 표5 불량유형별")
    print(t5_defect_type().to_string())
    print("\n-- 그림")
    f1_daily_force(raw, daily)
    f2_recipe_profile(raw)
    f3_label_resolution(raw, daily)
    print("\n[01] 완료")


if __name__ == "__main__":
    main()
