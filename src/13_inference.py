"""13. 과제 A 의 추론 고도화 — 분포가정 없는 구간, 베이지안, 동등성 검정.

03 에서 2020-03-27 예측구간을 포아송 분포로 산출했다. 그러나 표본이 8일뿐인데
포아송 가정이 맞는지 검증할 방법이 사실상 없다. 가정에 전적으로 의존하는 구간이다.

본 절은 세 가지를 보강한다. 모두 소표본에 대해 확립된 표준 방법이다.

(1) Jackknife+ 예측구간 (Barber, Candes, Ramdas, Tibshirani 2021, Ann. Statist.)
    분포가정 없이 유한표본 피복확률을 보장한다(1-2alpha 이상).
    중요한 한계가 드러난다. 분포가정 없는 구간에서 신뢰수준 1-alpha 의 유한구간이
    존재하려면 ceil((1-alpha)(n+1)) <= n 이어야 한다. n=8 에서는
    ceil(0.95*9)=9 > 8 이므로 95% 유한구간이 원리상 존재하지 않는다.
    산출 가능한 최대 구간은 명목 88.9% Jackknife+ 구간, 일반 유한표본 보장 하한 77.8%다.
    -> 유의할 점은 '가정을 빼면 구간이 넓어진다'가 아니다. 실제로 명목 88.9% Jackknife+
       구간 [0.86, 8.00] 은 포아송 95% 구간 [1, 10] 보다 오히려 좁다.
       분포가정이 사주는 것은 좁은 구간이 아니라 '95% 라고 말할 자격'이다.
       n=8 에서 95% 는 가정 없이는 주장 자체가 불가능하다.

(2) 베이지안 Gamma-Poisson (켤레사전분포, 해석해)
    약한 정보 사전분포를 두고 사후예측분포(음이항)를 구한다. MCMC 불필요.
    사전분포 민감도를 함께 제시해 결론이 사전분포에 끌려가지 않음을 보인다.

(3) 동등성 검정 TOST (Schuirmann 1987; Lakens 2017)
    '효과를 검출하지 못했다'와 '효과가 없다'는 다르다. 3.5절의 검정력 분석은 전자를
    보였고, 본 절은 후자를 검정한다. 귀무가설을 뒤집어
      H0: |rho| >= delta   vs   H1: |rho| < delta
    로 두고, 기각 가능한 최소 delta 를 찾는다. 그 delta 가 크면 '효과 없음'조차
    주장할 수 없다는 뜻이며, 이것이 본 데이터의 정확한 상태다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.stats import nbinom, norm, poisson, spearmanr

from common import FIG, OUT, TEST_DATE, daily_label, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
ALPHA = 0.05
MAIN_RECIPE = "A_주모드(~3bar)"


def daily() -> pd.DataFrame:
    raw = load_raw()
    g = raw.groupby("d")
    t = pd.DataFrame({
        "용접점수": g.size(),
        "전류_표준편차": g["weld current(kA)"].std(),
        "가압력_표준편차": g["weld force(bar)"].std(),
        "비주레시피_비율": g["recipe"].apply(lambda s: (s != MAIN_RECIPE).mean()),
    }).join(daily_label())
    return t


# ══════════════════════════════════════════════════════════════════
# (1) Jackknife+ 예측구간
# ══════════════════════════════════════════════════════════════════
def jackknife_plus(y: np.ndarray, alpha: float) -> tuple[float, float, bool]:
    """절편 전용 모형의 Jackknife+ 예측구간.

    각 i 에 대해 i 를 뺀 적합값 mu_-i 와 잔차 R_i = |y_i - mu_-i| 를 구한 뒤
      lower = (mu_-i - R_i) 들의 alpha 분위수(하위)
      upper = (mu_-i + R_i) 들의 1-alpha 분위수(상위)
    를 취한다. 유한구간 존재 조건은 ceil((1-alpha)(n+1)) <= n.
    """
    n = len(y)
    need = int(np.ceil((1 - alpha) * (n + 1)))
    finite = need <= n
    lo_c, hi_c = [], []
    for i in range(n):
        m = np.ones(n, bool); m[i] = False
        mu = y[m].mean()
        r = abs(y[i] - mu)
        lo_c.append(mu - r); hi_c.append(mu + r)
    lo_c, hi_c = np.sort(lo_c), np.sort(hi_c)
    if not finite:
        return -np.inf, np.inf, False
    k = need - 1
    return float(lo_c[n - 1 - k]), float(hi_c[k]), True


def t38_intervals(t: pd.DataFrame) -> pd.DataFrame:
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    n = len(y)
    c = y.mean()
    rows = []

    # 포아송 (03 에서 쓴 방법)
    rows.append({
        "방법": "포아송 (분포가정)", "신뢰수준": "95%",
        "구간": f"[{int(poisson.ppf(0.025, c))}, {int(poisson.ppf(0.975, c))}]",
        "유한구간": "예", "가정": "불량건수가 포아송을 따른다",
    })

    # Jackknife+ 여러 신뢰수준.
    # 유한구간 존재 조건 ceil((1-a)(n+1)) <= n 은 a >= 1/(n+1) 과 동치이므로
    # 달성 가능한 최대 신뢰수준은 정확히 n/(n+1) 이다. 부동소수점 오차를 피해
    # a = 1/(n+1) 을 그대로 쓴다(0.111 로 반올림하면 조건을 아슬아슬하게 벗어난다).
    a_max = 1.0 / (n + 1)
    for a, lab_ in [(0.05, "95%"),
                    (a_max, f"명목 {(1 - a_max) * 100:.1f}% Jackknife+ 구간, 일반 유한표본 보장 하한 {(1 - 2 * a_max) * 100:.1f}%"),
                    (0.20, "80%")]:
        lo, hi, fin = jackknife_plus(y, a)
        rows.append({
            "방법": "Jackknife+ (분포가정 없음)", "신뢰수준": lab_,
            "구간": "(-inf, inf) — 존재하지 않음" if not fin else f"[{max(lo,0):.2f}, {hi:.2f}]",
            "유한구간": "아니오" if not fin else "예",
            "가정": "교환가능성만",
        })

    out = pd.DataFrame(rows)
    save_table(out, "t38_prediction_intervals", index=False)
    return out


# ══════════════════════════════════════════════════════════════════
# (2) 베이지안 Gamma-Poisson
# ══════════════════════════════════════════════════════════════════
def t39_bayes(t: pd.DataFrame) -> pd.DataFrame:
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    n, s = len(y), y.sum()

    rows = []
    priors = [("Jeffreys (a=0.5, b=0)", 0.5, 1e-9),
              ("약정보 (a=1, b=0.2)", 1.0, 0.2),
              ("중간정보 (a=5, b=1)", 5.0, 1.0),
              ("강정보, 의도적 편향 (a=20, b=10)", 20.0, 10.0)]
    for name, a0, b0 in priors:
        a1, b1 = a0 + s, b0 + n           # Gamma 사후
        mean = a1 / b1
        lo, hi = gamma_dist.ppf([0.025, 0.975], a1, scale=1 / b1)
        # 사후예측 = 음이항(r=a1, p=b1/(b1+1))
        p = b1 / (b1 + 1)
        pl, ph = nbinom.ppf([0.025, 0.975], a1, p)
        rows.append({
            "사전분포": name,
            "사후 평균 발생률": round(mean, 3),
            "사후 95% 구간": f"[{lo:.2f}, {hi:.2f}]",
            "사후예측 95% 구간": f"[{int(pl)}, {int(ph)}]",
            "사후예측 평균": round(a1 / b1, 2),
        })
    out = pd.DataFrame(rows)
    save_table(out, "t39_bayes_prior_sensitivity", index=False)
    return out


# ══════════════════════════════════════════════════════════════════
# (3) 동등성 검정 TOST
# ══════════════════════════════════════════════════════════════════
def tost_rho(r: float, n: int, delta: float) -> float:
    """상관에 대한 TOST. Fisher z 변환 기반 두 단측검정의 큰 쪽 p 값.

    H0: |rho| >= delta  를 기각하면 '|rho| < delta 로 동등'하다고 말할 수 있다.
    """
    if n <= 3:
        return 1.0
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    zd = np.arctanh(delta)
    p_upper = norm.cdf((z - zd) / se)    # H0: rho >= delta 기각 (아래쪽)
    p_lower = 1 - norm.cdf((z + zd) / se)  # H0: rho <= -delta 기각 (위쪽)
    return float(max(p_upper, p_lower))


def t40_equivalence(t: pd.DataFrame) -> pd.DataFrame:
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    n = len(y)
    rows = []
    for var in ["전류_표준편차", "가압력_표준편차", "비주레시피_비율", "용접점수"]:
        r, p_two = spearmanr(lab[var], y)
        # 기각 가능한 최소 delta 탐색
        d_min = None
        for d in np.arange(0.05, 1.0, 0.01):
            if tost_rho(r, n, d) < ALPHA:
                d_min = d
                break
        rows.append({
            "변수": var,
            "관측 rho": round(r, 3),
            "유의성 검정 p": round(p_two, 3),
            "TOST p (delta=0.5)": round(tost_rho(r, n, 0.5), 3),
            "기각 가능 최소 delta": "없음" if d_min is None else round(float(d_min), 2),
            "결론": "효과 유무 모두 판정 불가" if d_min is None or d_min >= 0.9
                    else f"|rho| < {d_min:.2f} 로만 동등 주장 가능",
        })
    out = pd.DataFrame(rows)
    save_table(out, "t40_equivalence_tost", index=False)
    return out


# ══════════════════════════════════════════════════════════════════
def f24(t: pd.DataFrame, bayes: pd.DataFrame):
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    c = y.mean()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))

    # 좌: 구간 비교
    items = []
    items.append(("포아송 95%\n(분포가정)", poisson.ppf(0.025, c), poisson.ppf(0.975, c), "#2e86c1"))
    lo, hi, fin = jackknife_plus(y, 1.0 / (len(y) + 1))
    items.append(("명목 88.9% Jackknife+ 구간\n일반 유한표본 보장 하한 77.8%", max(lo, 0), hi, "#c0392b"))
    lo8, hi8, _ = jackknife_plus(y, 0.20)
    items.append(("Jackknife+ 80%\n(가정 없음)", max(lo8, 0), hi8, "#e67e22"))
    a1h = bayes[bayes["사전분포"].str.startswith("약정보")]["사후예측 95% 구간"].iloc[0]
    bl, bh = [float(x) for x in a1h.strip("[]").split(", ")]
    items.append(("베이지안 95%\n(약정보 사전)", bl, bh, "#27ae60"))

    for i, (name, lo_, hi_, col) in enumerate(items):
        a1.plot([lo_, hi_], [i, i], lw=7, color=col, solid_capstyle="butt", alpha=0.85)
        a1.plot([c], [i], "o", color="black", ms=7, zorder=3)
        a1.text(hi_ + 0.25, i, f"[{lo_:.1f}, {hi_:.1f}]", va="center", fontsize=9)
    a1.set_yticks(range(len(items)))
    a1.set_yticklabels([x[0] for x in items], fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel("2020-03-27 불량건수 예측")
    a1.set_xlim(-0.5, 16)
    a1.set_title("구간 폭이 아니라 '95%' 라는 주장 자체가 가정에 기댄다", fontsize=10.5)

    # 우: 달성 가능한 신뢰수준
    ns = np.arange(4, 41)
    maxconf = ns / (ns + 1) * 100
    a2.plot(ns, maxconf, "-", color="#1f4e79", lw=2.2)
    a2.axhline(95, ls="--", color="#c0392b", lw=1.4)
    a2.text(39, 95.6, "95%", fontsize=9, color="#c0392b", ha="right")
    a2.axvline(8, ls=":", color="black", lw=1.4)
    a2.plot([8], [8 / 9 * 100], "o", color="#c0392b", ms=9, zorder=3)
    a2.annotate(f"n=8 → 최대 {8/9*100:.1f}%", (8, 8 / 9 * 100), fontsize=9.5,
                xytext=(14, -14), textcoords="offset points", color="#c0392b")
    a2.axvline(19, ls=":", color="#27ae60", lw=1.4)
    a2.annotate("95% 유한구간에 필요한 n=19", (19, 95), fontsize=9,
                xytext=(10, -34), textcoords="offset points", color="#27ae60")
    a2.set_xlabel("라벨이 있는 작업일 수 (n)")
    a2.set_ylabel("분포가정 없이 달성 가능한 최대 신뢰수준 (%)")
    a2.set_ylim(75, 100)
    a2.set_title("n=8 에서는 분포가정 없는 95% 유한구간이 존재하지 않는다", fontsize=10.5)

    fig.suptitle("n=8 에서 분포가정 없는 95% 예측구간은 원리상 존재하지 않는다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f24_inference.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f24_inference.png")


def main():
    print("\n[13] 과제 A 추론 고도화")
    t = daily()

    print("\n══ (1) 예측구간 — 분포가정 유무 비교 ══")
    iv = t38_intervals(t)
    print(iv.to_string(index=False))
    print("\n   n=8 에서 ceil(0.95*9)=9 > 8 이므로 분포가정 없는 95% 유한구간은 존재하지 않는다.")
    print("   명목 88.9% Jackknife+ 구간, 일반 유한표본 보장 하한 77.8%.")
    print("   분포가정 없는 명목 95% 유한구간을 쓰려면 n >= 19 가 필요하다.")

    print("\n══ (2) 베이지안 Gamma-Poisson — 사전분포 민감도 ══")
    bayes = t39_bayes(t)
    print(bayes.to_string(index=False))
    pr = bayes["사후예측 95% 구간"].tolist()
    print(f"\n   사전분포를 Jeffreys 부터 의도적으로 편향된 강정보까지 바꿔도")
    print(f"   사후예측 구간은 {pr[0]} ~ {pr[-1]} 범위에 머문다. 결론이 사전분포에 끌려가지 않는다.")

    print("\n══ (3) 동등성 검정 TOST — '효과 없음'을 주장할 수 있는가 ══")
    eq = t40_equivalence(t)
    print(eq.to_string(index=False))
    print("\n   유의성 검정은 '효과 있음'을, 동등성 검정은 '효과 없음'을 주장하는 절차다.")
    print("   본 데이터는 양쪽 모두 기각하지 못한다. 전형적인 검정력 부족 상태이며,")
    print("   따라서 '공정조건은 불량과 무관하다'는 주장도 데이터로 뒷받침되지 않는다.")

    f24(t, bayes)

    # 최종 제출 예측치 갱신 (구간을 가정 없는 쪽으로 보수화)
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    c = float(y.mean())
    lo89, hi89, _ = jackknife_plus(y, 1.0 / (len(y) + 1))
    tst = t.loc[[TEST_DATE]]
    sub = pd.DataFrame({
        "working_time": [str(TEST_DATE)],
        "Machine_Name": ["Spot-01"],
        "Item_No": ["65235-25800"],
        "n_welds": [int(tst["용접점수"].iloc[0])],
        "pred_defect_count": [round(c, 2)],
        "pred_lower_95_poisson": [int(poisson.ppf(0.025, c))],
        "pred_upper_95_poisson": [int(poisson.ppf(0.975, c))],
        "pred_lower_889_jackknife_plus": [round(max(lo89, 0), 2)],
        "pred_upper_889_jackknife_plus": [round(hi89, 2)],
    })
    sub.to_csv(OUT / "prediction_20200327.csv", index=False, encoding="utf-8-sig")
    print("\n  saved -> outputs/prediction_20200327.csv (분포가정 있는/없는 구간 병기)")
    print("\n[13] 완료")


if __name__ == "__main__":
    main()
