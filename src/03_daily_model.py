"""03. 작업일 단위 불량건수 모델 + 2020-03-27 예측.

02에서 가장 강한 신호가 '생산량이 많은 날일수록 불량률이 낮다'(rho=-0.69)였다.
이는 불량건수가 생산량에 비례하지 않는다는 뜻이고, 그 자체로 검증해야 할 가설이다.

  H1 (비례) : 불량건수 ~ Poisson(lambda = r * 용접점수)   -- 전수검사라면 성립해야 한다
  H0 (상수) : 불량건수 ~ Poisson(lambda = c)              -- 생산량과 무관하게 일정

H0 가 이기면 result 시트의 defect 는 '전수검사 불량건수'가 아니라
'작업일마다 일정 규모로 수행된 표본검사의 적발건수'에 가깝다는 해석이 선다.
이 판정은 불량률 정의와 모든 후속 해석의 전제를 바꾸므로 먼저 수행한다.

이어서 공정조건 집계변수의 설명력을 LOO-CV 로 평가하고, 라벨이 없는 03-27 을 예측한다.
표본이 8일이므로 모든 결과는 '추정' 이 아니라 '검정력 부족' 을 함께 보고한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, poisson, spearmanr

from common import FIG, OUT, TEST_DATE, daily_label, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

# 03-27 예측에 쓸 후보 설명변수 (모두 공정조건 집계 = 검사 이전 확보 가능)
CANDS = ["비주레시피_비율", "risk_평균", "고위험점_비율", "가압력_표준편차", "전류_표준편차", "전압_표준편차"]


def daily_features() -> pd.DataFrame:
    """02 의 위험스코어와 공정조건을 작업일로 집계."""
    raw = load_raw()
    sc = pd.read_csv(OUT / "weld_risk_scores.csv", encoding="utf-8-sig")
    sc["d"] = pd.to_datetime(sc["d"]).dt.date
    g = raw.groupby("d")
    gs = sc.groupby("d")
    t = pd.DataFrame(
        {
            "용접점수": g.size(),
            "가압력_평균": g["weld force(bar)"].mean(),
            "가압력_표준편차": g["weld force(bar)"].std(),
            "전류_표준편차": g["weld current(kA)"].std(),
            "전압_표준편차": g["weld Voltage(v)"].std(),
            "risk_평균": gs["risk"].mean(),
            "고위험점_비율": gs["risk"].apply(lambda s: (s > 0.9).mean()),
            "비주레시피_비율": gs["recipe"].apply(lambda s: (s != "A_주모드(~3bar)").mean()),
        }
    ).join(daily_label())
    return t


def h_test_scaling(t: pd.DataFrame) -> pd.DataFrame:
    """H1(비례) vs H0(상수) 판정. 불량건수가 생산량을 따라가는가?"""
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].values
    n = lab["용접점수"].values

    # H1: 상수 발생률 r. lambda_i = r * n_i
    r = y.sum() / n.sum()
    ll1 = poisson.logpmf(y, r * n).sum()
    # H0: 상수 건수 c. lambda_i = c
    c = y.mean()
    ll0 = poisson.logpmf(y, c).sum()

    # 두 모형은 모수 수가 같고(1개) 중첩되지 않으므로 AIC 로 비교한다.
    aic1, aic0 = 2 * 1 - 2 * ll1, 2 * 1 - 2 * ll0

    # 적합도 검정: 각 모형의 편차(deviance)가 카이제곱(df=n-1)을 따르는지
    def dev(mu):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        return 2 * (term - (y - mu)).sum()

    d1, d0 = dev(r * n), dev(c)
    df_ = len(y) - 1
    rows = [
        {
            "모형": "H1 비례 (lambda = r x 용접점수)",
            "추정": f"r = {r*1000:.3f} 건/1000점",
            "logLik": round(ll1, 2),
            "AIC": round(aic1, 2),
            "Deviance": round(d1, 2),
            "적합도 p": round(chi2.sf(d1, df_), 4),
        },
        {
            "모형": "H0 상수 (lambda = c)",
            "추정": f"c = {c:.3f} 건/일",
            "logLik": round(ll0, 2),
            "AIC": round(aic0, 2),
            "Deviance": round(d0, 2),
            "적합도 p": round(chi2.sf(d0, df_), 4),
        },
    ]
    out = pd.DataFrame(rows)
    save_table(out, "t9_scaling_hypothesis", index=False)
    print(out.to_string(index=False))
    better = "H0(상수)" if aic0 < aic1 else "H1(비례)"
    print(f"\n   => AIC 기준 선호 모형: {better}  (dAIC = {abs(aic0-aic1):.2f})")
    if aic0 < aic1:
        print("      불량건수가 생산량에 비례하지 않는다. result 의 defect 는 전수검사 결과로")
        print("      보기 어렵고, 일정 규모의 표본검사 적발건수로 해석하는 편이 타당하다.")
        print("      -> 이후 분석의 종속변수는 '불량률' 이 아니라 '일별 불량건수' 로 둔다.")
    return out


def loo_eval(t: pd.DataFrame) -> pd.DataFrame:
    """LOO-CV. 설명변수 1개짜리 Poisson GLM 들을 절편전용 모형과 비교."""
    lab = t.dropna(subset=["defects"]).copy()
    y = lab["defects"].values
    nn = len(lab)

    def loo_mae(Xcol: np.ndarray | None) -> float:
        errs = []
        for i in range(nn):
            m = np.ones(nn, bool)
            m[i] = False
            if Xcol is None:
                pred = y[m].mean()
            else:
                X = sm.add_constant(Xcol[m].reshape(-1, 1), has_constant="add")
                try:
                    fit = sm.GLM(y[m], X, family=sm.families.Poisson()).fit()
                    pred = float(fit.predict([[1.0, Xcol[i]]])[0])
                except Exception:
                    pred = y[m].mean()
            errs.append(abs(y[i] - pred))
        return float(np.mean(errs))

    base = loo_mae(None)
    rows = [{"모형": "절편 전용 (기준선)", "LOO-MAE": round(base, 3), "기준선 대비": "—", "rho": "—", "p": "—"}]
    for c in CANDS:
        mae = loo_mae(lab[c].values)
        rho, p = spearmanr(lab[c], y)
        rows.append(
            {
                "모형": f"+ {c}",
                "LOO-MAE": round(mae, 3),
                "기준선 대비": f"{(mae-base)/base*100:+.1f}%",
                "rho": round(rho, 3),
                "p": round(p, 3),
            }
        )
    out = pd.DataFrame(rows).sort_values("LOO-MAE")
    save_table(out, "t10_loo_cv", index=False)
    print(out.to_string(index=False))
    best = out.iloc[0]["모형"]
    if best == "절편 전용 (기준선)":
        print("\n   => 어떤 공정조건 집계변수도 절편 전용 모형을 이기지 못한다.")
        print("      공정조건만으로는 일별 불량건수를 예측할 수 없다는 것이 본 분석의 결론이다.")
    return out


def perm_test(t: pd.DataFrame, n_perm: int = 20000) -> pd.DataFrame:
    """순열검정. 관측된 상관이 우연 범위인지 n=8 에서 직접 확인한다."""
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].values
    rows = []
    for c in CANDS:
        x = lab[c].values
        obs = spearmanr(x, y).statistic
        null = np.array([spearmanr(x, RNG.permutation(y)).statistic for _ in range(n_perm)])
        p = float((np.abs(null) >= abs(obs)).mean())
        rows.append(
            {
                "변수": c,
                "관측 rho": round(obs, 3),
                "순열 p(양측)": round(p, 4),
                "귀무분포 95% 범위": f"[{np.quantile(null,0.025):+.2f}, {np.quantile(null,0.975):+.2f}]",
                "판정": "유의" if p < 0.05 else "우연과 구분 불가",
            }
        )
    out = pd.DataFrame(rows).sort_values("순열 p(양측)")
    save_table(out, "t11_permutation_test", index=False)
    print(out.to_string(index=False))
    return out


def predict_test_day(t: pd.DataFrame) -> pd.DataFrame:
    """03-27 불량건수 예측. 설명변수에 설명력이 없으므로 기준선 모형으로 예측하고,
    포아송 예측구간을 함께 제시한다. 점추정만 내놓는 것보다 정직하고 현장에서도 쓸모 있다."""
    lab = t.dropna(subset=["defects"])
    tst = t.loc[[TEST_DATE]]
    c = lab["defects"].mean()
    lo, hi = poisson.ppf(0.025, c), poisson.ppf(0.975, c)

    # 참고용: 비례 모형으로도 예측해 둔다(H1 기각되더라도 비교값으로 보고).
    r = lab["defects"].sum() / lab["용접점수"].sum()
    prop = r * tst["용접점수"].iloc[0]

    out = pd.DataFrame(
        [
            {
                "작업일": str(TEST_DATE),
                "용접점수": int(tst["용접점수"].iloc[0]),
                "비주레시피_비율(%)": round(tst["비주레시피_비율"].iloc[0] * 100, 2),
                "예측_불량건수(기준선)": round(c, 2),
                "95% 예측구간": f"[{int(lo)}, {int(hi)}]",
                "참고_비례모형 예측": round(prop, 2),
                "근거": "공정조건 변수의 LOO 설명력이 기준선 미만이므로 기준선 모형 채택",
            }
        ]
    )
    save_table(out, "t12_prediction_0327", index=False)
    print(out.to_string(index=False))

    sub = pd.DataFrame(
        {
            "working_time": [str(TEST_DATE)],
            "Machine_Name": ["Spot-01"],
            "Item_No": ["65235-25800"],
            "n_welds": [int(tst["용접점수"].iloc[0])],
            "pred_defect_count": [round(c, 2)],
            "pred_lower_95": [int(lo)],
            "pred_upper_95": [int(hi)],
        }
    )
    sub.to_csv(OUT / "prediction_20200327.csv", index=False, encoding="utf-8-sig")
    print("\n  saved -> outputs/prediction_20200327.csv")
    return out


def f6_scaling(t: pd.DataFrame):
    """그림6. 불량건수 vs 생산량 — 비례하지 않는다는 핵심 증거."""
    lab = t.dropna(subset=["defects"])
    r = lab["defects"].sum() / lab["용접점수"].sum()
    c = lab["defects"].mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.1))
    a1.scatter(lab["용접점수"], lab["defects"], s=80, color="#1f4e79", zorder=3)
    for dd, row in lab.iterrows():
        a1.annotate(dd.strftime("%m-%d"), (row["용접점수"], row["defects"]),
                    fontsize=7.5, xytext=(5, 4), textcoords="offset points")
    xs = np.linspace(600, 2100, 50)
    a1.plot(xs, r * xs, "--", color="#c0392b", label=f"H1 비례 ({r*1000:.2f}건/1000점)")
    a1.axhline(c, ls="-", color="#27ae60", label=f"H0 상수 ({c:.2f}건/일)")
    a1.set_xlabel("용접점수 (생산량)")
    a1.set_ylabel("불량건수")
    a1.set_title("불량건수는 생산량을 따라가지 않는다", fontsize=10)
    a1.legend(fontsize=8)
    a1.set_ylim(0, 10)

    a2.scatter(lab["용접점수"], lab["defects"] / lab["용접점수"] * 1000, s=80, color="#c0392b", zorder=3)
    for dd, row in lab.iterrows():
        a2.annotate(dd.strftime("%m-%d"),
                    (row["용접점수"], row["defects"] / row["용접점수"] * 1000),
                    fontsize=7.5, xytext=(5, 4), textcoords="offset points")
    rho, p = spearmanr(lab["용접점수"], lab["defects"] / lab["용접점수"])
    a2.set_xlabel("용접점수 (생산량)")
    a2.set_ylabel("불량률 (건/1000점)")
    a2.set_title(f"그 결과 불량률은 생산량에 반비례한다  ρ={rho:+.2f}, p={p:.3f}", fontsize=10)
    fig.suptitle("종속변수 정의의 전제: defect 는 전수검사 건수가 아니라 표본검사 적발건수에 가깝다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f6_scaling_check.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f6_scaling_check.png")


def main():
    print("\n[03] 작업일 단위 모델")
    t = daily_features()
    save_table(t.round(4), "t7b_daily_features")
    print(t.round(3).to_string())

    print("\n-- H1(비례) vs H0(상수) 판정")
    h_test_scaling(t)

    print("\n-- LOO-CV (설명변수 1개 Poisson GLM vs 절편 전용)")
    loo_eval(t)

    print("\n-- 순열검정 (n=8 에서의 우연 범위)")
    perm_test(t)

    print(f"\n-- {TEST_DATE} 예측")
    predict_test_day(t)

    f6_scaling(t)
    print("\n[03] 완료")


if __name__ == "__main__":
    main()
