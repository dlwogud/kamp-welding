"""05. 안정 공정조건 범위 제안 + 재검사 기준.

중요한 전제를 먼저 밝힌다.
  03 에서 공정조건 집계변수는 일별 불량건수를 설명하지 못했다(LOO 기준선 미달, 순열 p>0.15).
  따라서 '불량률을 최소화하는 공정조건'을 본 데이터로 역산하는 것은 불가능하다.
  그렇게 도출한 범위는 표본 8일의 잡음을 과적합한 것에 불과하다.

대신 본 절은 입증 가능한 것만 제안한다.
  (1) 공정 일관성 기준의 관리범위 : 안정일 5일의 자연변동으로부터 I-MR 관리한계를 세운다.
      '불량이 적은 조건'이 아니라 '이 설비가 통제상태일 때의 조건'이다. 이는 데이터로 입증된다.
  (2) 이질 레시피 혼입 관리 : 안정일 0% vs 혼입일 16~36% 라는 명확한 이원화를 관리지표로 만든다.
  (3) 재검사 기준 : 위험스코어 임계값별 재검사 물량을 제시해 현장이 운영부하로 선택하게 한다.
      불량 라벨이 없으므로 '검출률'은 제시할 수 없다. 제시하지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    FIG,
    OUT,
    PROC,
    PROC_KR,
    SPEC,
    TEST_DATE,
    load_raw,
    save_table,
    setup_mpl,
)

plt = setup_mpl()
MAIN_RECIPE = "A_주모드(~3bar)"
# 안정일 = 01 에서 판정한, 고압모드 혼입이 없는 작업일
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]


def control_limits(raw: pd.DataFrame) -> pd.DataFrame:
    """(1) 안정일 기준 I-MR 관리한계.

    개별값-이동범위 관리도의 표준식을 쓴다.
      sigma_hat = MRbar / d2,  d2 = 1.128 (n=2)
      관리한계 = xbar +- 3 * sigma_hat
    정규성에 덜 의존하는 분위수 기준(p0.1~p99.9)도 함께 제시해 둘 중 보수적인 쪽을 택하게 한다.
    """
    stable = raw[raw["d"].astype(str).isin(STABLE_DAYS)].sort_values(["d", "idx"])
    rows = []
    for c in PROC:
        v = stable[c].to_numpy()
        mr = np.abs(np.diff(v))
        sigma = mr.mean() / 1.128
        xbar = v.mean()
        lcl, ucl = xbar - 3 * sigma, xbar + 3 * sigma
        q_lo, q_hi = np.quantile(v, 0.001), np.quantile(v, 0.999)
        spec_lo, spec_hi = SPEC[c]
        # 공정능력: 사양 대비 산포가 얼마나 여유 있는가
        cp = (spec_hi - spec_lo) / (6 * sigma) if sigma > 0 else np.inf
        rows.append(
            {
                "변수": PROC_KR[c],
                "사양범위": f"{spec_lo:g} ~ {spec_hi:g}",
                "안정일 평균": round(xbar, 4),
                "sigma(MR기반)": round(sigma, 5),
                "관리한계 3sigma": f"{lcl:.3f} ~ {ucl:.3f}",
                "분위수 0.1~99.9%": f"{q_lo:.3f} ~ {q_hi:.3f}",
                "권고 관리범위": f"{max(lcl, q_lo):.3f} ~ {min(ucl, q_hi):.3f}",
                "Cp(사양기준)": round(cp, 1) if np.isfinite(cp) else "—",
            }
        )
    t = pd.DataFrame(rows)
    save_table(t, "t15_control_limits", index=False)
    return t


def mix_monitoring(raw: pd.DataFrame) -> pd.DataFrame:
    """(2) 이질 레시피 혼입 관리지표."""
    g = raw.groupby("d")
    t = pd.DataFrame(
        {
            "용접점수": g.size(),
            "비주레시피_비율(%)": g["recipe"].apply(lambda s: (s != MAIN_RECIPE).mean() * 100),
            "가압력>3bar(%)": g["weld force(bar)"].apply(lambda s: (s > 3).mean() * 100),
            "가압력>9bar 건수": g["weld force(bar)"].apply(lambda s: int((s > 9).sum())),
            "전압<2.6V 건수": g["weld Voltage(v)"].apply(lambda s: int((s < 2.6).sum())),
        }
    ).round(3)
    t["구분"] = np.where(t.index.astype(str).isin(STABLE_DAYS), "안정일", "혼입일")
    t.loc[TEST_DATE, "구분"] = "혼입일(예측대상)"
    save_table(t, "t16_mix_monitoring")
    return t


def reinspection(raw: pd.DataFrame) -> pd.DataFrame:
    """(3) 위험스코어 임계값별 재검사 물량.

    검출률은 산출하지 않는다 — 용접점 단위 정답이 없기 때문이다.
    현장이 선택할 수 있도록 '부하'만 정확히 제시한다.
    """
    sc = pd.read_csv(OUT / "weld_risk_scores.csv", encoding="utf-8-sig")
    sc["d"] = pd.to_datetime(sc["d"]).dt.date
    n_days = sc["d"].nunique()
    rows = []
    for thr in [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]:
        f = sc["risk"] > thr
        per_day = sc[f].groupby("d").size().reindex(sc["d"].unique(), fill_value=0)
        rows.append(
            {
                "임계값": thr,
                "전체 플래그 수": int(f.sum()),
                "전체 대비(%)": round(f.mean() * 100, 2),
                "일평균 재검사(점)": round(f.sum() / n_days, 1),
                "일최대 재검사(점)": int(per_day.max()),
                "혼입일 집중도(%)": round(
                    sc.loc[f, "recipe"].ne(MAIN_RECIPE).mean() * 100, 1
                ),
            }
        )
    t = pd.DataFrame(rows)
    save_table(t, "t17_reinspection_load", index=False)
    return t


def defect_hypotheses() -> pd.DataFrame:
    """불량유형별 공정 메커니즘 가설. 본 데이터로 검증되지 않았음을 명시한다."""
    t = pd.DataFrame(
        [
            {
                "불량유형": "파임불량 (14건)",
                "공정 메커니즘 가설": "과대 가압력에 의한 전극 압입 과다",
                "데이터상 관련 관측": "가압력 9bar 초과 56점 존재 (주모드의 4배)",
                "검증 상태": "미검증 — 용접점 단위 불량 라벨 없음",
            },
            {
                "불량유형": "용접부족 (13건)",
                "공정 메커니즘 가설": "입열 부족. 전압 강하 또는 통전시간 부족",
                "데이터상 관련 관측": "9bar 초과 구간에서 전압 2.702 -> 2.513 으로 강하",
                "검증 상태": "미검증 — 동일",
            },
            {
                "불량유형": "크랙발생 (12건)",
                "공정 메커니즘 가설": "과대 입열 후 급랭, 또는 가압 불안정에 의한 응력",
                "데이터상 관련 관측": "혼입일 전압 표준편차가 안정일의 8~12배",
                "검증 상태": "미검증 — 동일. 03-31 은 크랙 기록 자체가 누락",
            },
        ]
    )
    save_table(t, "t18_defect_hypotheses", index=False)
    return t


def f9_window(raw: pd.DataFrame, cl: pd.DataFrame):
    """그림9. 사양범위 vs 실제운전 vs 권고관리범위 3단 비교."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    stable = raw[raw["d"].astype(str).isin(STABLE_DAYS)]
    for ax, c in zip(axes, PROC):
        spec_lo, spec_hi = SPEC[c]
        all_lo, all_hi = raw[c].min(), raw[c].max()
        row = cl[cl["변수"] == PROC_KR[c]].iloc[0]
        rec_lo, rec_hi = [float(x) for x in row["권고 관리범위"].split("~")]

        bars = [
            ("사양범위", spec_lo, spec_hi, "#bdc3c7"),
            ("실제운전\n(전체)", all_lo, all_hi, "#e67e22"),
            ("권고 관리범위", rec_lo, rec_hi, "#27ae60"),
        ]
        for i, (name, lo, hi, col) in enumerate(bars):
            ax.barh(i, hi - lo, left=lo, height=0.55, color=col, alpha=0.85)
            ax.text((lo + hi) / 2, i, f"{lo:g}~{hi:g}", ha="center", va="center",
                    fontsize=7.5, color="black")
        ax.set_yticks(range(3))
        ax.set_yticklabels([b[0] for b in bars], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(PROC_KR[c], fontsize=10)
        ax.grid(axis="y", alpha=0)
    fig.suptitle(
        "사양범위는 넓고 실제 운전은 그보다 훨씬 좁다 — 권고 관리범위는 안정일 자연변동에서 도출",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "f9_stable_window.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f9_stable_window.png")


def f10_mix_chart(mix: pd.DataFrame):
    """그림10. 혼입률 관리도 — 현장이 매일 볼 수 있는 단일 지표."""
    fig, ax = plt.subplots(figsize=(9, 4))
    days = list(mix.index)
    vals = mix["비주레시피_비율(%)"].values
    colors = ["#2e86c1" if v < 1 else "#c0392b" for v in vals]
    ax.bar([d.strftime("%m-%d") for d in days], vals, color=colors, alpha=0.85)
    ax.axhline(1.0, ls="--", color="#e67e22", lw=1.5)
    ax.text(len(days) - 0.4, 1.8, "권고 관리한계 1%", fontsize=8.5, color="#e67e22", ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylabel("비주 레시피 혼입률 (%)")
    ax.set_title("이질 레시피 혼입률 — 안정일 0% vs 혼입일 16~36%, 이원화가 뚜렷하다", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    for tick, d in zip(ax.get_xticklabels(), days):
        if d == TEST_DATE:
            tick.set_color("#c0392b")
            tick.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(FIG / "f10_mix_control_chart.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f10_mix_control_chart.png")


def main():
    print("\n[05] 안정 공정조건 범위 + 재검사 기준")
    raw = load_raw()

    print("\n-- (1) 안정일 기준 관리한계")
    cl = control_limits(raw)
    print(cl.to_string(index=False))

    print("\n-- (2) 이질 레시피 혼입 모니터링")
    mix = mix_monitoring(raw)
    print(mix.to_string())

    print("\n-- (3) 위험스코어 임계값별 재검사 부하")
    ri = reinspection(raw)
    print(ri.to_string(index=False))

    print("\n-- 불량유형별 메커니즘 가설 (미검증)")
    print(defect_hypotheses().to_string(index=False))

    f9_window(raw, cl)
    f10_mix_chart(mix)
    print("\n[05] 완료")


if __name__ == "__main__":
    main()
