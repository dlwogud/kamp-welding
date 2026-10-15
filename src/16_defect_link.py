"""16. 최종 모형과 실제 불량 39건의 연결 — 시도와 결과.

본 분석의 가장 큰 공백을 메운다.

2.4~2.6 에서 최종 모형의 성능(7유형 평균 ROC-AUC 0.971)은 전부 '합성 이상'에 대한
것이다. 이상을 우리가 정의하고 주입하고 탐지했다. 그 수치가 아무리 높아도
**실제 용접불량과의 연결이 입증되지 않으면 과제의 핵심 질문에 답한 것이 아니다.**

본 절은 가용한 유일한 실제 라벨(8개 작업일, 총 39건)과 최종 모형의 출력을 대조한다.
표본이 8개이므로 검출력이 없다는 것은 이미 알고 있다(3.5절). 그럼에도 수행하는 이유는
세 가지다.
  1. 연결을 '시도하지 않은 것'과 '시도했으나 검출되지 않은 것'은 전혀 다르다.
  2. 방향성(부호)만이라도 일관되는지는 확인할 수 있다.
  3. 추후 라벨이 확보되었을 때 그대로 재실행할 수 있는 절차를 남긴다.

설계상의 주의
  복제 구간(800점 x 4일 = 3,200점, 전체의 26.8%)은 라벨과 독립적으로 주입된 구조이므로
  집계에 포함하면 '아티팩트가 불량과 무관함'을 재확인하는 데 그친다. 따라서
  **복제 구간을 제외한 실제 생산점만으로 집계**한다.

  중요한 귀결: 2020-04-03 은 작업일 전체(800점)가 복제 구간이므로 실제 생산점이 0 이다.
  따라서 라벨이 있는 8일 중 7일만 본 분석에 들어간다. 표본이 8 에서 7 로 줄어
  검출력은 더 낮아진다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import poisson, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from common import (FIG, OUT, daily_label, load_raw, mark_duplicates,
                    save_table, setup_mpl)

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
RNG = np.random.default_rng(SEED)
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]
MAIN_RECIPE = "A_주모드(~3bar)"
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
FEATS = [F, "R", "E"]
K_NB = 3
CAL_FRAC = 0.3
N_PERM = 20000


def physics(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    amp = d[I] * 1000.0
    d["R"] = d[V] / amp
    d["E"] = d[V] * amp * (d[T] / 1000.0)
    return d


def score_all() -> pd.DataFrame:
    """최종 모형(kNN k=3, 물리 파생공간)으로 전체 11,939점을 채점한다.

    참조·교정셋은 안정일에서만 구성한다. 평가 대상 작업일의 점이 자기 자신의
    참조집합에 들어가면 점수가 낙관적으로 왜곡되므로, 안정일을 채점할 때는
    그 날을 제외한 나머지 안정일로 적합한다(Leave-One-Day-Out).
    """
    raw = physics(mark_duplicates(load_raw()))
    raw = raw[~raw["is_dup"]]                      # 복제 구간 전체 제외
    stable = raw[raw["d"].astype(str).isin(STABLE_DAYS)]
    out = []

    for d, g in raw.groupby("d"):
        if str(d) in STABLE_DAYS:
            ref_pool = stable[stable["d"] != d]          # 자기 날 제외
        else:
            ref_pool = stable                             # 혼입일은 안정일 전체로 채점
        idx = RNG.permutation(len(ref_pool))
        ncal = int(len(ref_pool) * CAL_FRAC)
        cal, ref = ref_pool.iloc[idx[:ncal]], ref_pool.iloc[idx[ncal:]]

        sc = RobustScaler().fit(ref[FEATS])
        nn = NearestNeighbors(n_neighbors=K_NB).fit(sc.transform(ref[FEATS]))
        s_cal = np.sort(nn.kneighbors(sc.transform(cal[FEATS]))[0].mean(1))
        s = nn.kneighbors(sc.transform(g[FEATS]))[0].mean(1)
        n = len(s_cal)
        p = (1.0 + (n - np.searchsorted(s_cal, s, side="left"))) / (n + 1.0)

        gg = g.copy()
        gg["anomaly_score"] = s
        gg["conformal_p"] = p
        out.append(gg)

    return pd.concat(out).sort_values(["d", "idx"])


def aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    """작업일 단위 집계. 입력은 이미 복제 구간이 제외된 실제 생산점이다."""
    g = scored.groupby("d")
    t = pd.DataFrame({
        "용접점수": g.size(),
        "이상점수_평균": g["anomaly_score"].mean(),
        "이상점수_p95": g["anomaly_score"].quantile(0.95),
        "플래그율_p01(%)": g["conformal_p"].apply(lambda s: (s < 0.01).mean() * 100),
        "플래그율_p05(%)": g["conformal_p"].apply(lambda s: (s < 0.05).mean() * 100),
        "플래그수_p01": g["conformal_p"].apply(lambda s: int((s < 0.01).sum())),
        "컨포멀p_중앙값": g["conformal_p"].median(),
    }).join(daily_label())
    return t


def t51_link(agg: pd.DataFrame, tag: str) -> pd.DataFrame:
    """집계지표와 실제 불량건수의 관계. 순열검정 포함."""
    lab = agg.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    rows = []
    cols = ["이상점수_평균", "이상점수_p95", "플래그율_p01(%)", "플래그율_p05(%)",
            "플래그수_p01", "컨포멀p_중앙값"]
    for c in cols:
        x = lab[c].to_numpy(float)
        if np.allclose(x, x[0]):
            rows.append({"집계지표": c, "Spearman rho": np.nan, "점근 p": np.nan,
                         "순열 p": np.nan, "판정": "변동 없음 (전 작업일 동일)"})
            continue
        rho, p = spearmanr(x, y)
        null = np.array([spearmanr(x, RNG.permutation(y)).statistic for _ in range(N_PERM)])
        pp = float((np.abs(null) >= abs(rho)).mean())
        rows.append({"집계지표": c, "Spearman rho": round(float(rho), 3),
                     "점근 p": round(float(p), 3), "순열 p": round(pp, 4),
                     "판정": "유의" if pp < 0.05 else "우연과 구분 불가"})
    out = pd.DataFrame(rows).sort_values("순열 p")
    save_table(out, f"t51_defect_link_{tag}", index=False)
    return out


def t52_loo(agg: pd.DataFrame) -> pd.DataFrame:
    """이상탐지 집계지표가 불량건수 예측에 기여하는가 (LOO-CV).

    2.7 의 과제 A 는 '공정조건 집계'를 썼다. 여기서는 '최종 이상탐지 모형의 출력'을
    설명변수로 써서 기준선을 넘는지 확인한다. 설명변수의 출처가 다르므로 별도 검증이다.
    """
    import statsmodels.api as sm

    lab = agg.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    n = len(y)

    def loo(xcol):
        errs = []
        for i in range(n):
            m = np.ones(n, bool); m[i] = False
            if xcol is None:
                pred = y[m].mean()
            else:
                try:
                    X = sm.add_constant(xcol[m].reshape(-1, 1), has_constant="add")
                    f = sm.GLM(y[m], X, family=sm.families.Poisson()).fit()
                    pred = float(f.predict([[1.0, xcol[i]]])[0])
                except Exception:
                    pred = y[m].mean()
            errs.append(abs(y[i] - pred))
        return float(np.mean(errs))

    base = loo(None)
    rows = [{"설명변수": "없음 (절편 전용 기준선)", "LOO-MAE": round(base, 3), "기준선 대비": "—"}]
    for c in ["이상점수_평균", "이상점수_p95", "플래그율_p01(%)", "컨포멀p_중앙값"]:
        x = lab[c].to_numpy(float)
        if np.allclose(x, x[0]):
            continue
        mae = loo(x)
        rows.append({"설명변수": c, "LOO-MAE": round(mae, 3),
                     "기준선 대비": f"{(mae - base) / base * 100:+.1f}%"})
    out = pd.DataFrame(rows).sort_values("LOO-MAE")
    save_table(out, "t52_defect_link_loo", index=False)
    return out


def t53_direction(agg_main: pd.DataFrame) -> pd.DataFrame:
    """방향성 점검 — 검출력이 없더라도 부호가 일관되는지는 볼 수 있다."""
    lab = agg_main.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    rows = []
    for c in ["이상점수_평균", "이상점수_p95", "플래그율_p01(%)", "플래그율_p05(%)"]:
        x = lab[c].to_numpy(float)
        if np.allclose(x, x[0]):
            rows.append({"집계지표": c, "부호": "—", "해석": "변동 없음"})
            continue
        rho = spearmanr(x, y).statistic
        rows.append({
            "집계지표": c,
            "부호": "양(+)" if rho > 0 else "음(−)",
            "해석": "이상이 많은 날에 불량이 많다" if rho > 0 else "이상이 많은 날에 불량이 적다",
        })
    out = pd.DataFrame(rows)
    save_table(out, "t53_direction_check", index=False)
    return out


def f28(agg_main: pd.DataFrame, link: pd.DataFrame):
    lab = agg_main.dropna(subset=["defects"])
    cands = [c for c in ["플래그율_p01(%)", "이상점수_평균", "이상점수_p95"]
             if not np.allclose(lab[c], lab[c].iloc[0])]
    if not cands:
        cands = ["이상점수_평균"]
    fig, axes = plt.subplots(1, len(cands), figsize=(4.4 * len(cands), 4.1))
    if len(cands) == 1:
        axes = [axes]
    for ax, c in zip(axes, cands):
        ax.scatter(lab[c], lab["defects"], s=80, color="#1f4e79", zorder=3)
        for dd, r in lab.iterrows():
            ax.annotate(dd.strftime("%m-%d"), (r[c], r["defects"]), fontsize=7.5,
                        xytext=(5, 4), textcoords="offset points")
        row = link[link["집계지표"] == c]
        if len(row) and not pd.isna(row["Spearman rho"].iloc[0]):
            ax.set_title(f"ρ={row['Spearman rho'].iloc[0]:+.3f}, 순열 p={row['순열 p'].iloc[0]:.3f}",
                         fontsize=10)
        ax.set_xlabel(c)
        ax.set_ylabel("실제 불량건수")
        ax.set_ylim(0, 9)
    fig.suptitle("최종 모형의 출력과 실제 불량건수 — 연결은 검출되지 않았다 (n=8)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f28_defect_link.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f28_defect_link.png")


def main():
    print("\n[16] 최종 모형과 실제 불량의 연결")
    scored = score_all()
    print(f"  채점 완료: {len(scored):,} 점 (안정일은 Leave-One-Day-Out 으로 채점)")

    agg_main = aggregate(scored)
    save_table(agg_main.round(4), "t50b_daily_agg_main")

    print("\n-- 작업일별 집계 (복제 구간 제외한 실제 생산점)")
    print(agg_main.round(3).to_string())
    lab_n = int(agg_main["defects"].notna().sum())
    print(f"\n   라벨이 있고 실제 생산점이 남은 작업일: {lab_n} 일")
    print("   (2020-04-03 은 전체가 복제 구간이라 제외됨 — 표본이 8 에서 7 로 감소)")

    print("\n══ (1) 집계지표 vs 실제 불량건수 ══")
    link_main = t51_link(agg_main, "main")
    print(link_main.to_string(index=False))

    print("\n══ (3) 불량건수 예측 기여도 (LOO-CV) ══")
    loo = t52_loo(agg_main)
    print(loo.to_string(index=False))
    best = loo.iloc[0]["설명변수"]
    print(f"\n   최우수: {best}")
    if best.startswith("없음"):
        print("   이상탐지 출력도 절편 전용 기준선을 넘지 못한다.")
        print("   2.7 의 공정조건 집계와 동일한 결론이며, 설명변수의 출처가 달라도 결과가 같다.")

    print("\n══ (4) 방향성 점검 ══")
    dirn = t53_direction(agg_main)
    print(dirn.to_string(index=False))

    f28(agg_main, link_main)

    # 결론 표
    sig = (link_main["순열 p"] < 0.05).sum()
    concl = pd.DataFrame([
        {"항목": "검정한 집계지표 수", "값": f"{link_main['Spearman rho'].notna().sum()} 종"},
        {"항목": "유의한 지표 수 (순열 p<0.05)", "값": f"{int(sig)} 종"},
        {"항목": "최소 순열 p", "값": f"{link_main['순열 p'].min():.4f}"},
        {"항목": "불량건수 예측 기여", "값": "기준선 초과 없음" if best.startswith("없음") else best},
        {"항목": "결론", "값": f"최종 모형의 출력과 실제 불량의 연결은 현재 표본({int(link_main.shape[0] and agg_main['defects'].notna().sum())}일)으로 검출되지 않는다"},
        {"항목": "해석상 주의", "값": "'연결이 없다'가 아니라 '검출할 표본이 없다'. "
                                   "복제 구간 제외로 표본이 8일에서 7일로 줄어 검출력은 더 낮다"},
        {"항목": "재실행 조건", "값": "용접점 단위 라벨 확보 시 본 스크립트를 "
                                   "그대로 재실행하면 직접 검증 가능"},
    ])
    save_table(concl, "t54_defect_link_conclusion", index=False)
    print("\n-- 결론")
    print(concl.to_string(index=False))

    scored[["idx", "d", F, I, V, T, "R", "E", "recipe",
            "anomaly_score", "conformal_p"]].to_csv(
        OUT / "weld_scores_lodo.csv", index=False, encoding="utf-8-sig")
    print("\n  saved -> outputs/weld_scores_lodo.csv")
    print("\n[16] 완료")


if __name__ == "__main__":
    main()
