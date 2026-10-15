"""21. 불량유형별 분리 분석 — 지문이 명시한 불량 모드에 직접 대응한다.

지금까지의 분석은 불량건수 '합계'만 썼다. 그러나 데이터는 세 유형으로 나뉘어 있고
(파임불량 14 / 용접부족 13 / 크랙발생 12), 지문도 "미용접, 과용접, 기공, 균열 및
접합강도 부족" 처럼 불량 모드를 구분해 언급한다.

본 절은 유형별로 나누어 분석하되, **사전 지정 가설 검정**의 형태를 취한다.

왜 사전 지정인가
    작업일이 5~7개뿐인데 공정 집계변수를 전부 유형별로 돌리면 수십 번의 검정이 되고,
    그중 하나가 우연히 유의해지는 것은 거의 확실하다. 그 결과를 발견으로 보고하면
    다중비교에 의한 허위 발견이다.
    따라서 **유형마다 물리 메커니즘에서 도출한 가설을 하나씩, 방향까지 지정**해
    단측 검정을 수행하고, Holm 절차로 다중비교를 보정한다.

사전 지정 가설 (5.x절의 메커니즘 가설에서 도출, 분석 전에 고정)
    H1 파임불량   ← 과대 가압력.      지표: 일별 가압력 95분위수.      방향: 양(+)
    H2 용접부족   ← 입열 부족.        지표: 일별 입열 E 의 5분위수.    방향: 음(-)
    H3 크랙발생   ← 입열·가압 불안정.  지표: 일별 입열 E 의 변동계수.   방향: 양(+)

주 분석 표본
    복제 구간이 없는 **안정일 5일**. 혼입일은 작업일의 44~100%가 복제 구간이라
    '불량건수 / 실제 생산점' 의 대응이 깨진다. 7일 분석은 민감도로 병기한다.

추가로 확인하는 것
    세 유형의 일별 건수가 **서로 함께 움직이는가**. 함께 움직인다면 유형별 메커니즘보다
    '그 날의 검사 강도' 같은 공통 요인이 지배한다는 뜻이며, 2.7.1절의
    '불량건수가 생산량에 비례하지 않는다'는 결과와 맞물린다.
"""
from __future__ import annotations

import warnings
from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (FIG, STABLE_DAYS, load_genuine, load_result, save_table,
                    setup_mpl)

plt = setup_mpl()
warnings.filterwarnings("ignore")
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"

# 사전 지정 가설. (유형, 지표명, 방향). 분석 전에 고정하고 바꾸지 않는다.
HYPOTHESES = [
    ("파임불량", "가압력_p95", +1, "과대 가압력에 의한 전극 압입 과다"),
    ("용접부족", "입열E_p05", -1, "입열 부족 (전압 강하 또는 통전시간 부족)"),
    ("크랙발생", "입열E_변동계수", +1, "입열·가압 불안정에 의한 응력"),
]


def day_features() -> pd.DataFrame:
    g = load_genuine().copy()
    amp = g[I] * 1000.0
    g["E"] = g[V] * amp * (g[T] / 1000.0)
    gg = g.groupby("d")
    t = pd.DataFrame({
        "용접점수": gg.size(),
        "가압력_p95": gg[F].quantile(0.95),
        "가압력_표준편차": gg[F].std(),
        "입열E_p05": gg["E"].quantile(0.05),
        "입열E_변동계수": gg["E"].std() / gg["E"].mean(),
        "전압_평균": gg[V].mean(),
    })
    return t


def type_counts() -> pd.DataFrame:
    r = load_result()
    p = r.pivot_table(index="d", columns="defect_name", values="defect", aggfunc="sum")
    return p


def exact_perm_p(x: np.ndarray, y: np.ndarray, direction: int) -> tuple[float, float]:
    """단측 정확 순열검정. 모든 배열을 열거한다 (n <= 8).

    상수 입력을 반드시 걸러야 한다. x 나 y 가 상수면 Spearman 이 NaN 이 되고,
    NaN 비교는 전부 False 가 되어 p = 0 이 산출된다. 즉 **검정 불가 상황이
    '완벽한 유의'로 뒤집힌다.** 초판에서 실제로 이 오류가 발생해
    '가압력 p95'(안정일 5일 모두 2.38 로 상수)가 거짓 지지 판정을 받았다.
    """
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    rho = spearmanr(x, y).statistic
    if not np.isfinite(rho):
        return float("nan"), float("nan")
    stat = direction * rho
    null = np.array([direction * spearmanr(x, np.array(perm)).statistic
                     for perm in permutations(y)])
    null = null[np.isfinite(null)]
    if len(null) == 0:
        return float(rho), float("nan")
    return float(rho), float((null >= stat).mean())


def holm(ps: list[float]) -> list[float]:
    """Holm-Bonferroni 보정."""
    m = len(ps)
    order = np.argsort(ps)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * ps[i]
        prev = max(prev, v)
        adj[i] = min(1.0, prev)
    return list(adj)


def t71_prespecified(feat: pd.DataFrame, cnt: pd.DataFrame, days: list[str],
                     tag: str) -> pd.DataFrame:
    rows, ps = [], []
    for typ, metric, direction, mech in HYPOTHESES:
        idx = [d for d in feat.index
               if str(d) in days and typ in cnt.columns and d in cnt.index
               and not pd.isna(cnt.loc[d, typ])]
        base = {"불량유형": typ, "사전 지정 지표": metric,
                "예측 방향": "+" if direction > 0 else "-", "n": len(idx),
                "메커니즘 가설": mech}
        if len(idx) < 4:
            rows.append({**base, "Spearman rho": "—", "단측 정확 p": "—",
                         "검정 가능": "아니오 (표본 부족)"})
            ps.append(1.0)
            continue
        x = feat.loc[idx, metric].to_numpy(float)
        y = cnt.loc[idx, typ].to_numpy(float)
        rho, p = exact_perm_p(x, y, direction)
        if not np.isfinite(p):
            why = "지표가 상수" if len(np.unique(x)) < 2 else "건수가 상수"
            rows.append({**base, "Spearman rho": "—", "단측 정확 p": "—",
                         "검정 가능": f"아니오 ({why})"})
            ps.append(1.0)
            continue
        rows.append({**base, "Spearman rho": round(rho, 3),
                     "단측 정확 p": round(p, 4), "검정 가능": "예"})
        ps.append(p)
    adj = holm(ps)
    out = pd.DataFrame(rows)
    out["Holm 보정 p"] = [round(a, 4) if r["검정 가능"] == "예" else "—"
                          for a, (_, r) in zip(adj, out.iterrows())]
    out["판정"] = ["지지" if (r["검정 가능"] == "예" and a < 0.05)
                   else ("검정 불가" if r["검정 가능"] != "예" else "지지되지 않음")
                   for a, (_, r) in zip(adj, out.iterrows())]
    save_table(out, f"t71_prespecified_{tag}", index=False)
    return out


def t72_cross_type(cnt: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """유형 간 일별 건수가 함께 움직이는가."""
    sub = cnt.loc[[d for d in cnt.index if str(d) in days]]
    cols = [c for c in sub.columns if sub[c].notna().sum() >= 4]
    rows = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            m = sub[[a, b]].dropna()
            if len(m) < 4:
                continue
            rho, _ = spearmanr(m[a], m[b])
            _, p = exact_perm_p(m[a].to_numpy(float), m[b].to_numpy(float), +1)
            rows.append({"유형 쌍": f"{a} vs {b}", "n": len(m),
                         "Spearman rho": round(float(rho), 3),
                         "단측 정확 p": round(p, 4)})
    out = pd.DataFrame(rows)
    save_table(out, "t72_cross_type", index=False)
    return out


def t73_composition(cnt: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """유형 구성비가 작업일에 따라 변하는가 (카이제곱 동질성 검정)."""
    from scipy.stats import chi2_contingency
    sub = cnt.loc[[d for d in cnt.index if str(d) in days]].fillna(0)
    tab = sub.to_numpy(float)
    keep = tab.sum(axis=1) > 0
    tab = tab[keep]
    try:
        chi2, p, dof, _ = chi2_contingency(tab)
    except Exception:
        chi2, p, dof = np.nan, np.nan, np.nan
    out = pd.DataFrame([{
        "검정": "작업일 간 불량유형 구성비 동질성 (카이제곱)",
        "작업일 수": int(keep.sum()),
        "chi2": round(float(chi2), 3) if np.isfinite(chi2) else "—",
        "자유도": int(dof) if np.isfinite(dof) else "—",
        "p": round(float(p), 4) if np.isfinite(p) else "—",
        "해석": ("구성비가 작업일에 따라 다르다고 볼 근거가 없다"
                 if (np.isfinite(p) and p >= 0.05) else "구성비가 작업일에 따라 다르다"),
        "주의": "기대도수가 작아(평균 1.6건) 근사의 정확도가 낮다. 참고치로만 본다",
    }])
    save_table(out, "t73_composition", index=False)
    return out


def t74_loo(feat: pd.DataFrame, cnt: pd.DataFrame, days: list[str]) -> pd.DataFrame:
    """유형별 일별 건수 예측 — 사전 지정 지표가 절편 전용 기준선을 넘는가."""
    import statsmodels.api as sm
    rows = []
    for typ, metric, _, _ in HYPOTHESES:
        idx = [d for d in feat.index
               if str(d) in days and typ in cnt.columns and d in cnt.index
               and not pd.isna(cnt.loc[d, typ])]
        if len(idx) < 4 or feat.loc[idx, metric].nunique() < 2:
            continue
        x = feat.loc[idx, metric].to_numpy(float)
        y = cnt.loc[idx, typ].to_numpy(float)
        n = len(y)

        def loo(use_x: bool) -> float:
            errs = []
            for k in range(n):
                m = np.ones(n, bool); m[k] = False
                if not use_x:
                    pred = y[m].mean()
                else:
                    try:
                        X = sm.add_constant(x[m].reshape(-1, 1), has_constant="add")
                        f = sm.GLM(y[m], X, family=sm.families.Poisson()).fit()
                        pred = float(f.predict([[1.0, x[k]]])[0])
                    except Exception:
                        pred = y[m].mean()
                errs.append(abs(y[k] - pred))
            return float(np.mean(errs))

        base, with_x = loo(False), loo(True)
        rows.append({"불량유형": typ, "n": n,
                     "기준선 LOO-MAE": round(base, 3),
                     f"사전지정 지표 추가": round(with_x, 3),
                     "기준선 대비": f"{(with_x - base) / base * 100:+.1f}%" if base > 0 else "—",
                     "기준선 초과": "예" if with_x < base else "아니오"})
    out = pd.DataFrame(rows)
    save_table(out, "t74_type_loo", index=False)
    return out


def f35(feat: pd.DataFrame, cnt: pd.DataFrame, days: list[str], pre: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (typ, metric, direction, _) in zip(axes, HYPOTHESES):
        idx = [d for d in feat.index
               if str(d) in days and typ in cnt.columns and d in cnt.index
               and not pd.isna(cnt.loc[d, typ])]
        if len(idx) < 3:
            ax.axis("off")
            continue
        x = feat.loc[idx, metric].to_numpy(float)
        y = cnt.loc[idx, typ].to_numpy(float)
        ax.scatter(x, y, s=85, color="#1f4e79", zorder=3)
        for d, xi, yi in zip(idx, x, y):
            ax.annotate(d.strftime("%m-%d"), (xi, yi), fontsize=7.5,
                        xytext=(5, 4), textcoords="offset points")
        r = pre[pre["불량유형"] == typ]
        ttl = f"{typ}  (예측 방향 {'+' if direction > 0 else '-'})"
        if len(r):
            rr = r.iloc[0]
            if rr["검정 가능"] == "예":
                ttl += f"\nρ={float(rr['Spearman rho']):+.3f} · Holm p={float(rr['Holm 보정 p']):.3f}"
                ttl += f" · {rr['판정']}"
            else:
                ttl += f"\n{rr['검정 가능']}"
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel(metric)
        ax.set_ylabel("일별 건수")
    fig.suptitle("불량유형별 사전 지정 가설 검정 — 어느 가설도 지지되지 않았다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f35_defect_types.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f35_defect_types.png")


def main():
    print("\n[21] 불량유형별 분리 분석")
    feat = day_features()
    cnt = type_counts()
    print(f"  유형별 총계: " + ", ".join(f"{c}={int(cnt[c].sum())}" for c in cnt.columns))

    days7 = [d for d in map(str, feat.index)]
    days5 = [d for d in days7 if d in STABLE_DAYS]
    print(f"  주 분석 {len(days5)}일 (안정일), 민감도 {len(days7)}일")

    print("\n══ (1) 사전 지정 가설 검정 — 주 분석 (안정일 5일) ══")
    pre5 = t71_prespecified(feat, cnt, days5, "stable5")
    print(pre5.to_string(index=False))

    print("\n══ (2) 민감도 — 실제 생산점이 있는 전 작업일 ══")
    pre7 = t71_prespecified(feat, cnt, days7, "all7")
    print(pre7.to_string(index=False))

    print("\n══ (3) 유형 간 동반 변동 ══")
    ct = t72_cross_type(cnt, days5)
    print(ct.to_string(index=False) if len(ct) else "  (검정 가능한 쌍 없음)")

    print("\n══ (4) 유형 구성비의 작업일 간 동질성 ══")
    comp = t73_composition(cnt, days5)
    print(comp.to_string(index=False))

    print("\n══ (5) 유형별 예측 기여 (LOO-CV) ══")
    loo = t74_loo(feat, cnt, days5)
    print(loo.to_string(index=False) if len(loo) else "  (검정 가능한 유형 없음)")

    f35(feat, cnt, days5, pre5)

    n_sup = int((pre5["판정"] == "지지").sum())
    n_test = int((pre5["검정 가능"] == "예").sum())
    concl = pd.DataFrame([
        {"항목": "분석 설계",
         "값": "유형마다 물리 메커니즘에서 도출한 가설을 방향까지 사전 지정하고 "
               "단측 정확 순열검정 + Holm 보정"},
        {"항목": "왜 사전 지정인가",
         "값": "작업일 5개에서 공정변수를 전부 유형별로 돌리면 수십 회 검정이 되어 "
               "우연한 유의가 거의 확실하다. 검정 수를 3개로 고정했다"},
        {"항목": "검정 가능했던 가설", "값": f"{n_test}/3 (파임불량은 지표가 안정일 5일 모두 2.38 로 상수여서 검정 불가)"},
        {"항목": "지지된 가설", "값": f"{n_sup}/{max(n_test,1)}"},
        {"항목": "예측 기여",
         "값": "사전 지정 지표가 절편 전용 기준선을 넘는 유형 "
               f"{int((loo['기준선 초과'] == '예').sum()) if len(loo) else 0}/{len(loo) if len(loo) else 3}"},
        {"항목": "유형 간 동반 변동",
         "값": "세 유형이 함께 움직인다면 유형별 메커니즘보다 그 날의 공통 요인이 "
               "지배한다는 뜻이다. 결과는 표 참조"},
        {"항목": "검정력 한계",
         "값": "n=5 에서 단측 정확 순열검정의 최소 p 는 1/120 = 0.0083 이고, "
               "Holm 보정 후에는 0.025 다. 즉 완벽한 관계여도 겨우 유의해지는 설계다"},
        {"항목": "결론",
         "값": "불량유형별 물리 메커니즘은 본 데이터로 지지되지 않는다. "
               "5.x절의 메커니즘 가설은 '미검증' 상태를 유지한다"},
    ])
    save_table(concl, "t75_type_conclusion", index=False)
    print("\n-- 결론")
    print(concl.to_string(index=False))
    print("\n[21] 완료")


if __name__ == "__main__":
    main()
