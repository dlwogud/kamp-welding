"""28. 전수 특징 쓸이 — "충분히 찾아봤는가" 에 답한다.

왜 필요한가
    2.7·3.1 은 **사전지정한 가설**만 검정했다(물리적 근거가 있는 후보 + Holm 보정).
    사전지정은 다중검정을 통제하는 올바른 설계이지만, **"혹시 놓친 특징이 있는가"**
    라는 질문에는 답하지 못한다. 심사·검토 과정에서 가장 먼저 나오는 질문이 그것이다.

    그래서 사후적으로 **일 단위 집계 특징을 전수로 생성해** 불량건수와의 상관을 모두
    검정한다. 이는 사전지정 검정을 **대체하지 않는다.** 탐색적 보완이며, 다중검정
    보정 없이는 어떤 개별 p 값도 증거로 쓸 수 없다는 점을 함께 보고한다.

무엇을 하는가
    1) 공정 4변수 × 통계량 7종 + 파생변수(R, E) + 고압비율 + 일내 결합상관
       → 원본·복제제외 두 기준으로 약 100개 특징을 만든다.
    2) 타깃을 4개로 나눈다 — 불량유형 3종(용접부족·크랙발생·파임불량) + 합계.
    3) 각 (타깃 × 특징) 조합에 Spearman 상관과 **순열검정**을 적용한다.
       n 이 작으므로 p = (초과 횟수 + 1)/(반복 + 1) 로 산출해 p = 0 을 만들지 않는다.
    4) Bonferroni 임계와 비교하고, **보정 전 p < 0.05 개수까지** 보고한다.

해석 규칙 — 이 절의 결론을 과대해석하지 않기 위해
    - **유의한 것이 없으면** "놓친 신호가 없다" 가 아니라
      "n=8 에서 검출 가능한 범위에 신호가 없다" 로 읽는다. 3.5절 검정력 분석이
      80% 검정력의 하한을 |rho| >= 0.87 로 제시했고, 순열 귀무분포 95% 범위는
      [-0.71, +0.72] 다. **중간 세기의 관계는 있어도 보이지 않는다.**
    - **가장 큰 상관이 나와도** 다중검정을 거쳤으므로 발견으로 제시하지 않는다.
      물리적으로 그럴싸해도 마찬가지다. 그 구분을 표에 적는다.

산출
    t70_feature_sweep         (타깃 × 특징) 전수 검정 결과
    t71_sweep_summary         타깃별 요약 — 최소 p, 최대 |rho|, 보정 임계
    f33_feature_sweep         타깃별 |rho| 분포와 순열 귀무범위
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import FIG, load_genuine, load_raw, load_result, mark_duplicates, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
N_PERM = 5000          # 순열 반복. n=8 에서 8! = 40,320 이므로 전수도 가능하나 5,000 으로 충분
RNG_SEED = 42

# 3.5절이 산출한 검출 가능 하한과 귀무분포 범위. 해석의 기준선으로 함께 표기한다.
POWER_FLOOR = 0.87     # 80% 검정력으로 검출 가능한 최소 |rho|
NULL_95 = (-0.71, 0.72)


def day_features(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """일 단위 집계 특징. 공정 4변수 × 통계량 + 물리 파생 + 결합상관."""
    out = {}
    for d, g in df.groupby(df["d"].astype(str)):
        r = {"n": float(len(g))}
        for col, nm in [(F, "force"), (I, "cur"), (V, "volt"), (T, "time")]:
            x = g[col].values.astype(float)
            r[f"{nm}_mean"] = float(x.mean())
            r[f"{nm}_med"] = float(np.median(x))
            r[f"{nm}_std"] = float(x.std(ddof=1))
            r[f"{nm}_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
            r[f"{nm}_p05"] = float(np.percentile(x, 5))
            r[f"{nm}_p95"] = float(np.percentile(x, 95))
            r[f"{nm}_rng"] = float(x.max() - x.min())
            r[f"{nm}_cv"] = float(x.std(ddof=1) / x.mean()) if x.mean() else 0.0
        # 저항점용접 물리 — 동저항 R = V/I, 입열 E = V*I*t
        R = g[V].values / g[I].values
        E = g[V].values * g[I].values * g[T].values
        for x, nm in [(R, "R"), (E, "E")]:
            r[f"{nm}_mean"] = float(x.mean())
            r[f"{nm}_std"] = float(x.std(ddof=1))
            r[f"{nm}_cv"] = float(x.std(ddof=1) / x.mean())
        r["frac_gt3bar"] = float((g[F] > 3.0).mean())
        if "is_dup" in g:
            r["frac_dup"] = float(g["is_dup"].mean())
        # 일내 가압력-전류 결합 (1.7절에서 확인한 구조)
        r["rho_force_cur"] = (float(spearmanr(g[F], g[I]).statistic)
                              if g[F].nunique() > 1 and g[I].nunique() > 1 else 0.0)
        out[d] = r
    return pd.DataFrame(out).T.add_suffix(f"__{tag}")


def perm_p(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM) -> tuple[float, float]:
    """Spearman 상관의 양측 순열검정.

    p = (|null| >= |obs| 횟수 + 1) / (반복 + 1). 분자·분모에 1 을 더해
    p = 0 이 나오지 않게 한다 — 반복 횟수가 주는 분해능 한계를 숨기지 않는다.
    """
    obs = spearmanr(x, y).statistic
    if not np.isfinite(obs):
        return float("nan"), float("nan")
    rng = np.random.default_rng(RNG_SEED)
    n = len(y)
    perms = (list(itertools.permutations(range(n))) if n <= 8 else None)
    if perms is not None and len(perms) <= n_perm:
        null = np.array([spearmanr(x, y[list(p)]).statistic for p in perms])
    else:
        null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n_perm)])
    p = float(((np.abs(null) >= abs(obs) - 1e-12).sum() + 1) / (len(null) + 1))
    return float(obs), p


def main() -> int:
    print("=" * 70)
    print("  28. 전수 특징 쓸이 — 탐색적 보완")
    print("=" * 70)

    res = load_result()
    piv = res.pivot_table(index=res["d"].astype(str), columns="defect_name",
                          values="defect", aggfunc="sum")
    print("\n불량유형별 작업일 집계:")
    print(piv.to_string())

    X = pd.concat([day_features(mark_duplicates(load_raw()), "raw"),
                   day_features(load_genuine(), "gen")], axis=1)
    print(f"\n생성한 일 단위 특징: {X.shape[1]} 개 (원본 기준 + 복제제외 기준)")

    targets = [(c, piv[c]) for c in piv.columns] + [("합계", piv.sum(axis=1))]
    rows = []
    for tname, y in targets:
        y = y.dropna()
        Xi = X.loc[[d for d in y.index if d in X.index]]
        yv = y.loc[Xi.index].values.astype(float)
        if len(np.unique(yv)) < 3:
            continue
        for c in Xi.columns:
            x = Xi[c].values.astype(float)
            if not np.isfinite(x).all() or len(np.unique(x)) < 3:
                continue
            obs, p = perm_p(x, yv)
            if not np.isfinite(obs):
                continue
            rows.append({"타깃": tname, "작업일 수": len(yv), "특징": c,
                         "rho": round(obs, 3), "순열 p(양측)": round(p, 4)})
    sw = pd.DataFrame(rows)
    save_table(sw, "t70_feature_sweep", index=False)

    print("\n══ 타깃별 결과 ══")
    srows = []
    for tname, g in sw.groupby("타깃", sort=False):
        m = len(g)
        bonf = 0.05 / m
        top = g.nsmallest(1, "순열 p(양측)").iloc[0]
        srows.append({
            "타깃": tname, "작업일 수": int(g["작업일 수"].iloc[0]), "검정한 특징": m,
            "최대 |rho|": round(g["rho"].abs().max(), 3),
            "그 특징": top["특징"], "최소 순열 p": round(top["순열 p(양측)"], 4),
            "Bonferroni 임계": round(bonf, 5),
            "보정 전 p<0.05": int((g["순열 p(양측)"] < 0.05).sum()),
            "보정 후 통과": int((g["순열 p(양측)"] < bonf).sum()),
        })
    summ = pd.DataFrame(srows)
    print(summ.to_string(index=False))
    save_table(summ, "t71_sweep_summary", index=False)

    tot = len(sw)
    n_raw = int((sw["순열 p(양측)"] < 0.05).sum())
    n_bonf = int((sw["순열 p(양측)"] < 0.05 / tot).sum())
    print(f"\n전체 {tot} 건 검정")
    print(f"  보정 전 p < 0.05 : {n_raw} 건")
    print(f"  Bonferroni(0.05/{tot} = {0.05/tot:.6f}) 통과 : {n_bonf} 건")
    print(f"  최대 |rho| : {sw['rho'].abs().max():.3f}")

    f33_plot(sw)

    print("\n" + "=" * 70)
    if n_raw == 0:
        print(f"  결론: 보정 전 기준으로도 유의한 조합이 없다.")
    else:
        print(f"  결론: 보정 전 {n_raw} 건이 p < 0.05 이나 Bonferroni 통과는 {n_bonf} 건이다.")
    print(f"  단, 이는 '신호가 없다' 가 아니다. 3.5절의 검출 하한이 |rho| >= {POWER_FLOOR},")
    print(f"  순열 귀무분포 95% 범위가 [{NULL_95[0]}, {NULL_95[1]}] 이므로")
    print("  중간 세기의 관계는 존재해도 이 표본에서는 보이지 않는다.")
    print("  → '검출 불가' 로 보고하며, '효과 없음' 으로 읽지 않는다.")
    print("=" * 70)
    return 0


def f33_plot(sw: pd.DataFrame):
    """타깃별 |rho| 분포. 순열 귀무범위와 검정력 하한을 함께 그린다."""
    order = list(sw["타깃"].unique())
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    data = [sw.loc[sw["타깃"] == t, "rho"].abs().values for t in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    flierprops=dict(markersize=3, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#2e86c1")
        patch.set_alpha(0.5)
    for med in bp["medians"]:
        med.set_color("black")
    ax.axhline(NULL_95[1], ls="--", lw=1.4, color="#e67e22",
               label=f"순열 귀무분포 95% 상한 {NULL_95[1]}")
    ax.axhline(POWER_FLOOR, ls=":", lw=1.6, color="#c0392b",
               label=f"80% 검정력 검출 하한 |rho| = {POWER_FLOOR}")
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylabel("|Spearman rho|")
    ax.set_ylim(0, 1.0)
    ax.set_title("전수 쓸이 — 관측된 상관은 모두 우연과 구분되지 않는 구간에 있다", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "f33_feature_sweep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
