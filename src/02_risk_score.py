"""02. 용접점 단위 불량위험 스코어 (비지도) + 작업일 라벨로 외부검증.

문제 재정의의 핵심.
  용접점 단위 불량 라벨이 없으므로 지도학습은 불가능하다. 대신
    (1) 공정조건만으로 용접점별 '정상운전 이탈도'를 비지도로 산출하고,
    (2) 이를 작업일 단위로 집계해 실제 불량건수와 대조하여 외부검증한다.
  (1)은 최종 품질검사 이전 시점의 정보만 사용하므로 지문의 '검사 이전 예측' 요건을 만족한다.
  (2)는 표본이 8일뿐이라 검정력이 없다. 유의성 여부를 그대로 보고하는 것이 이 절의 목적이다.

스코어 정의 (2성분 합성):
  S1 이탈도   : 주 레시피(A, ~2.3bar) 중심으로부터의 로버스트 거리. 레시피 혼입을 잡는다.
  S2 국소이상 : 소속 레시피 내부에서의 로버스트 마할라노비스 거리. 레시피 내 산발 이상을 잡는다.
  risk = 두 성분의 분위수 평균 -> [0,1]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.covariance import MinCovDet

from common import (
    FIG,
    OUT,
    PROC,
    PROC_KR,
    TEST_DATE,
    daily_label,
    load_raw,
    save_table,
    setup_mpl,
)

plt = setup_mpl()
RNG = 42
MAIN_RECIPE = "A_주모드(~3bar)"


def robust_mahalanobis(X: np.ndarray, seed: int = RNG) -> np.ndarray:
    """MCD 기반 로버스트 마할라노비스 거리의 제곱.

    분산이 0에 가까운 열이 있으면 공분산이 특이해지므로, 해당 열은 호출 전에 제거한다.
    """
    mcd = MinCovDet(random_state=seed, support_fraction=0.75).fit(X)
    return mcd.mahalanobis(X)


def usable_cols(df: pd.DataFrame, cols: list[str], tol: float = 1e-10) -> list[str]:
    """분산이 사실상 0인 열을 제외한다(특이 공분산 방지)."""
    return [c for c in cols if df[c].std() > tol]


def build_scores(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # --- S1: 주 레시피 중심으로부터의 이탈도 -------------------------------
    # 주 레시피만으로 로버스트 중심/산포를 적합하고, 전체 용접점을 그 기준으로 채점한다.
    main = df[df["recipe"] == MAIN_RECIPE]
    cols1 = usable_cols(main, PROC)
    mcd = MinCovDet(random_state=RNG, support_fraction=0.75).fit(main[cols1].values)
    d2_main = mcd.mahalanobis(df[cols1].values)
    df["S1_이탈도_d2"] = d2_main
    # 카이제곱 자유도 = 사용 변수 수. p값이 작을수록 주 운전조건에서 멀다.
    df["S1_p"] = chi2.sf(d2_main, df=len(cols1))

    # --- S2: 소속 레시피 내부 국소이상 -------------------------------------
    df["S2_국소_d2"] = np.nan
    for rc, g in df.groupby("recipe", observed=True):
        cols2 = usable_cols(g, PROC)
        # 표본이 적거나 사용할 변수가 없으면 국소이상은 정의하지 않는다.
        if len(g) < 50 or len(cols2) < 2:
            df.loc[g.index, "S2_국소_d2"] = 0.0
            continue
        df.loc[g.index, "S2_국소_d2"] = robust_mahalanobis(g[cols2].values)

    # --- 합성 -------------------------------------------------------------
    r1 = df["S1_이탈도_d2"].rank(pct=True)
    r2 = df["S2_국소_d2"].rank(pct=True)
    df["risk"] = (r1 + r2) / 2
    df["S1_rank"], df["S2_rank"] = r1, r2
    return df


def validate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """작업일로 집계해 실제 불량건수와 대조 (외부검증). n=8 이라는 점을 그대로 보고한다."""
    g = df.groupby("d")
    agg = pd.DataFrame(
        {
            "용접점수": g.size(),
            "risk_평균": g["risk"].mean(),
            "risk_p95": g["risk"].quantile(0.95),
            "고위험점_비율(%)": g["risk"].apply(lambda s: (s > 0.9).mean() * 100),
            "S1이탈_유의비율(%)": g["S1_p"].apply(lambda s: (s < 0.01).mean() * 100),
            "비주레시피_비율(%)": g["recipe"].apply(lambda s: (s != MAIN_RECIPE).mean() * 100),
        }
    ).join(daily_label())
    agg["불량률(건/1000점)"] = agg["defects"] / agg["용접점수"] * 1000

    lab = agg.dropna(subset=["defects"])
    rows = []
    for c in ["risk_평균", "risk_p95", "고위험점_비율(%)", "S1이탈_유의비율(%)", "비주레시피_비율(%)", "용접점수"]:
        for target, tname in [("defects", "불량건수"), ("불량률(건/1000점)", "불량률")]:
            rho, p = spearmanr(lab[c], lab[target])
            rows.append(
                {
                    "집계지표": c,
                    "대상": tname,
                    "Spearman rho": round(rho, 3),
                    "p값": round(p, 3),
                    "n": len(lab),
                    "유의(α=0.05)": "예" if p < 0.05 else "아니오",
                }
            )
    corr = pd.DataFrame(rows).sort_values("p값")
    save_table(agg.round(3), "t7_daily_risk_agg")
    save_table(corr, "t8_external_validation", index=False)
    return agg, corr


def f4_risk_dist(df: pd.DataFrame):
    """그림4. 레시피별 위험스코어 분포."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    order = [r for r in df["recipe"].cat.categories if (df["recipe"] == r).any()]
    a1.boxplot([df.loc[df["recipe"] == r, "risk"].values for r in order], widths=0.6,
               flierprops=dict(markersize=2, alpha=0.25))
    a1.set_xticklabels([r.split("_")[0] + "\n" + r.split("_")[1] for r in order], fontsize=8)
    a1.set_ylabel("위험스코어 (0~1)")
    a1.set_title("레시피별 위험스코어", fontsize=10)

    sc = a2.scatter(df["weld force(bar)"], df["weld Voltage(v)"], c=df["risk"],
                    s=6, cmap="RdYlBu_r", alpha=0.6)
    a2.set_xlabel(PROC_KR["weld force(bar)"])
    a2.set_ylabel(PROC_KR["weld Voltage(v)"])
    a2.set_title("가압력 × 전압 평면의 위험스코어", fontsize=10)
    fig.colorbar(sc, ax=a2, label="risk")
    fig.suptitle("위험스코어는 주 레시피 이탈과 레시피 내 산발 이상을 함께 반영한다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f4_risk_distribution.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f4_risk_distribution.png")


def f5_validation(agg: pd.DataFrame):
    """그림5. 외부검증 산점도 — 신호가 약하다는 사실을 숨기지 않고 보여준다."""
    lab = agg.dropna(subset=["defects"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    specs = [
        ("비주레시피_비율(%)", "불량률(건/1000점)"),
        ("risk_평균", "불량률(건/1000점)"),
        ("고위험점_비율(%)", "defects"),
    ]
    for ax, (x, y) in zip(axes, specs):
        ax.scatter(lab[x], lab[y], s=70, color="#1f4e79", zorder=3)
        for dd, r in lab.iterrows():
            ax.annotate(dd.strftime("%m-%d"), (r[x], r[y]), fontsize=7.5,
                        xytext=(4, 4), textcoords="offset points")
        rho, p = spearmanr(lab[x], lab[y])
        ax.set_xlabel(x)
        ax.set_ylabel("불량건수" if y == "defects" else y)
        ax.set_title(f"ρ={rho:+.3f}, p={p:.3f} (n={len(lab)})", fontsize=10)
    fig.suptitle(
        "외부검증: 공정조건 집계지표와 실제 불량의 관계는 통계적으로 유의하지 않다 (표본 8일)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "f5_external_validation.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f5_external_validation.png")


def main():
    print("\n[02] 용접점 위험스코어 + 외부검증")
    raw = load_raw()
    df = build_scores(raw)

    print("\n-- 레시피별 용접점 분포")
    rc = df.groupby("recipe", observed=True).agg(
        용접점수=("idx", "size"),
        비율_pct=("idx", lambda s: round(len(s) / len(df) * 100, 2)),
        risk_평균=("risk", "mean"),
        가압력_평균=("weld force(bar)", "mean"),
        전압_평균=("weld Voltage(v)", "mean"),
    ).round(4)
    print(rc.to_string())
    save_table(rc, "t6b_recipe_counts")

    print("\n-- 작업일 집계 + 라벨")
    agg, corr = validate_daily(df)
    print(agg.round(3).to_string())
    print("\n-- 외부검증 상관 (유의한 것부터)")
    print(corr.to_string(index=False))

    f4_risk_dist(df)
    f5_validation(agg)

    # 용접점별 스코어 저장 (제출용 산출물의 기초)
    keep = ["idx", "Machine_Name", "Item No", "working time", "d", *PROC,
            "recipe", "S1_이탈도_d2", "S1_p", "S2_국소_d2", "risk"]
    df[keep].to_csv(OUT / "weld_risk_scores.csv", index=False, encoding="utf-8-sig")
    print(f"\n  saved -> outputs/weld_risk_scores.csv ({len(df):,} 행)")

    n_test = (df["d"] == TEST_DATE).sum()
    print(f"  (예측 대상 {TEST_DATE}: {n_test:,} 점 포함)")
    print("\n[02] 완료")


if __name__ == "__main__":
    main()
