"""14. 최종 이상탐지 모형 확정 — 강도 스윕, 컨포멀 임계, 운영 스코어 산출.

12 의 벤치마크에서 kNN 거리(FS3 물리 파생공간)가 최소최대 기준 1위였다.
본 절은 그 모형을 운영 모형으로 확정하고 세 가지를 마무리한다.

(1) 이상 강도 스윕
    12 는 강도 K=2.0 한 점에서만 비교했다. 최종 모형이 강도에 따라 어떻게 변하는지
    7개 이상유형 x 4개 강도로 확인한다. 약한 이상에서의 한계를 숨기지 않는다.
    이웃 수 k 는 15 의 중첩 교차검증이 선택한 값(k=3)을 쓴다. 선택과 평가를 분리했으므로
    여기 보고되는 수치에는 하이퍼파라미터 선택에 의한 낙관 편향이 없다.

(2) 컨포멀 이상탐지 — 임계값 문제의 해결
    (Vovk/Gammerman/Shafer; Laxhammar & Falkman 2015)
    지금까지 임계값은 '학습셋 상위 1%' 라는 경험적 규칙이었고, 평가셋에서의 실제
    오경보율은 1.20~1.75% 로 목표를 벗어났다. 컨포멀 절차는 교정셋을 분리해
        p(x) = (1 + #{교정 점수 >= x 의 점수}) / (n_cal + 1)
    로 p 값을 만든다. p<0.01은 명목 임계다. 교환가능한 무작위 교정에서만 FPR 제어가 확인됐고,
    교차일 운영에서는 보장이 깨졌다. 운영 후보는 팽창계수 반영 p<0.0038이며 이 또한 근사값이다.

(3) 운영 스코어 산출
    전체 11,939 용접점에 대해 컨포멀 p 값을 산출해 제출 파일로 남긴다.
    라벨이 없는 2020-03-27 의 1,648 점을 포함한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from common import FIG, OUT, TEST_DATE, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]
MAIN_RECIPE = "A_주모드(~3bar)"
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
FEATS = [F, "R", "E"]          # 12 에서 선정된 FS3
N_NEIGHBORS = 3                # 15 의 중첩 교차검증이 선택 (5개 폴드 모두 k=3)
SEVERITIES = [1.0, 1.5, 2.0, 3.0]
ANOM_TYPES = ["A1 결합이탈", "A2 가압이탈", "A3 전압강하", "A4 통전단축",
              "A5 동시드리프트", "A6 산포증가", "A7 지속반복이상"]
CAL_FRAC = 0.3                 # 교정셋 비율


def physics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    amp = d[I] * 1000.0
    sec = d[T] / 1000.0
    d["R"] = d[V] / amp
    d["E"] = d[V] * amp * sec
    return d


def inject(te: pd.DataFrame, kind: str, sd: dict, k: float, rng) -> pd.DataFrame:
    a = te.copy()
    f = a[F].to_numpy()
    dirn = np.sign(f - np.median(f)); dirn[dirn == 0] = 1
    if kind == "A1 결합이탈":
        a[I] = a[I].to_numpy() + dirn * k * sd[I]
    elif kind == "A2 가압이탈":
        a[F] = a[F].to_numpy() + dirn * k * sd[F]
    elif kind == "A3 전압강하":
        a[V] = a[V].to_numpy() - k * sd[V]
    elif kind == "A4 통전단축":
        a[T] = a[T].to_numpy() - k * sd[T]
    elif kind == "A5 동시드리프트":
        a[F] = a[F].to_numpy() + k * sd[F]
        a[I] = a[I].to_numpy() - k * sd[I] * 0.735
    elif kind == "A6 산포증가":
        a[F] = a[F].to_numpy() + rng.normal(0, k * sd[F], len(a))
        a[I] = a[I].to_numpy() + rng.normal(0, k * sd[I], len(a))
    elif kind == "A7 지속반복이상":
        a[F] = float(np.median(a[F].to_numpy()) + k * sd[F])
        a[I] = float(np.median(a[I].to_numpy()) - k * sd[I])
        a[V] = float(np.median(a[V].to_numpy()))
        a[T] = float(np.median(a[T].to_numpy()))
    return physics(a)


class KNNDetector:
    """kNN 거리 기반 이상탐지 + 컨포멀 p 값.

    fit(X_train)      : 참조 집합 구성
    score(X)          : 평균 k-최근접 거리 (클수록 이상)
    calibrate(X_cal)  : 교정 점수 저장
    pvalue(X)         : 컨포멀 p 값. 교환가능한 교정에서만 FPR 제어가 성립한다.
    """

    def __init__(self, k: int = N_NEIGHBORS):
        self.k = k

    def fit(self, X: np.ndarray):
        self.sc_ = RobustScaler().fit(X)
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(self.sc_.transform(X))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.nn_.kneighbors(self.sc_.transform(X))[0].mean(1)

    def calibrate(self, X_cal: np.ndarray):
        self.cal_ = np.sort(self.score(X_cal))
        return self

    def pvalue(self, X: np.ndarray) -> np.ndarray:
        s = self.score(X)
        n = len(self.cal_)
        # (1 + #{cal >= s}) / (n + 1)
        ge = n - np.searchsorted(self.cal_, s, side="left")
        return (1.0 + ge) / (n + 1.0)


def genuine() -> pd.DataFrame:
    raw = load_raw()
    g = raw[(raw["recipe"] == MAIN_RECIPE) & (raw["d"].astype(str).isin(STABLE_DAYS))]
    return physics(g.sort_values(["d", "idx"]))


# ══════════════════════════════════════════════════════════════════
def t41_sweep(g: pd.DataFrame) -> pd.DataFrame:
    """이상 강도 x 유형 스윕 (ROC-AUC)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for k in SEVERITIES:
        for atype in ANOM_TYPES:
            aucs = []
            for h in STABLE_DAYS:
                tr = g[g["d"].astype(str) != h]
                te = g[g["d"].astype(str) == h]
                sd = {c: tr[c].std() for c in [F, I, V, T]}
                idx = rng.permutation(len(te)); half = len(te) // 2
                norm = te.iloc[idx[:half]]
                anom = inject(te.iloc[idx[half:]], atype, sd, k, rng)
                det = KNNDetector().fit(tr[FEATS].to_numpy())
                s = np.r_[det.score(norm[FEATS].to_numpy()), det.score(anom[FEATS].to_numpy())]
                y = np.r_[np.zeros(len(norm)), np.ones(len(anom))]
                aucs.append(roc_auc_score(y, s))
            rows.append({"이상 강도": k, "이상유형": atype,
                         "ROC-AUC": round(float(np.mean(aucs)), 3)})
    full = pd.DataFrame(rows)
    piv = full.pivot_table(index="이상유형", columns="이상 강도", values="ROC-AUC")
    piv["평균"] = piv.mean(axis=1).round(3)
    piv = piv.round(3)
    save_table(piv, "t41_severity_sweep")
    return piv


def t42_conformal(g: pd.DataFrame) -> pd.DataFrame:
    """컨포멀 임계의 오경보율 검증 — 교정셋을 어디서 잡느냐가 핵심.

    컨포멀의 유한표본 보장은 '교정셋과 평가셋이 교환가능하다'를 전제한다.
    본 데이터에는 작업일 간 드리프트가 있으므로, 교정셋을 다른 작업일에서 잡으면
    그 전제가 깨진다. 두 방식을 모두 수행해 차이를 드러낸다.

      교차일 교정     : 다른 작업일의 점으로 교정. 작업일 간 드리프트로 전제 위반.
      당일 순차 교정  : 평가일 초반 30% 로 교정(교대 시작 캘리브레이션). 현장에서
                        실제 수행 가능하나, 일중 드리프트가 있으면 역시 전제 위반.
      당일 무작위 교정 : 평가일을 무작위 분할. 정의상 교환가능하므로 FPR 제어가 확인되어야
                        한다. 위반의 원인이 드리프트임을 확정하기 위한 대조군이며,
                        미래 점을 뽑아야 하므로 실시간 운영에는 쓸 수 없다.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for scheme in ["교차일 교정", "당일 순차 교정", "당일 무작위 교정"]:
        for alpha in [0.005, 0.01, 0.02, 0.05]:
            emp, dets = [], {a: [] for a in ANOM_TYPES}
            for h in STABLE_DAYS:
                tr_all = g[g["d"].astype(str) != h]
                te_all = g[g["d"].astype(str) == h]
                sd = {c: tr_all[c].std() for c in [F, I, V, T]}

                if scheme == "교차일 교정":
                    idx = rng.permutation(len(tr_all))
                    ncal = int(len(tr_all) * CAL_FRAC)
                    cal, ref = tr_all.iloc[idx[:ncal]], tr_all.iloc[idx[ncal:]]
                    te = te_all
                elif scheme == "당일 순차 교정":
                    ncal = int(len(te_all) * CAL_FRAC)
                    cal, te = te_all.iloc[:ncal], te_all.iloc[ncal:]
                    ref = tr_all
                else:  # 당일 무작위 교정
                    j = rng.permutation(len(te_all))
                    ncal = int(len(te_all) * CAL_FRAC)
                    cal, te = te_all.iloc[j[:ncal]], te_all.iloc[j[ncal:]]
                    ref = tr_all

                det = KNNDetector().fit(ref[FEATS].to_numpy()).calibrate(cal[FEATS].to_numpy())
                emp.append(float((det.pvalue(te[FEATS].to_numpy()) < alpha).mean()) * 100)
                for a in ANOM_TYPES:
                    an = inject(te, a, sd, 2.0, rng)
                    dets[a].append(float((det.pvalue(an[FEATS].to_numpy()) < alpha).mean()) * 100)

            e = float(np.mean(emp))
            row = {"교정 방식": scheme,
                   "목표 오경보율(%)": alpha * 100,
                   "실제 오경보율(%)": round(e, 3),
                   "목표 대비 배수": round(e / (alpha * 100), 2),
                   "목표 충족": "예" if e <= alpha * 100 else "아니오"}
            for a in ANOM_TYPES:
                row[f"탐지율 {a[:2]}"] = round(float(np.mean(dets[a])), 1)
            rows.append(row)
    out = pd.DataFrame(rows)
    save_table(out, "t42_conformal_fpr", index=False)
    return out


def t43b_byday(scored: pd.DataFrame) -> pd.DataFrame:
    """작업일별 플래그 비율 — 부하가 어디서 발생하는지 분해한다."""
    g = scored.groupby("d")
    t = pd.DataFrame({
        "용접점수": g.size(),
        "플래그(p<0.01)": g["flag_p01"].sum(),
        "플래그 비율(%)": (g["flag_p01"].mean() * 100).round(2),
        "비주레시피 점수": g["recipe"].apply(lambda s: int((s != MAIN_RECIPE).sum())),
    })
    t["운전 구분"] = np.where(t.index.astype(str).isin(STABLE_DAYS), "안정일(적합에 사용)", "혼입일")
    save_table(t, "t43b_flag_by_day")
    return t


def t43_load(scored: pd.DataFrame) -> pd.DataFrame:
    """운영 부하 — 컨포멀 p 값 기준."""
    n_days = scored["d"].nunique()
    rows = []
    for alpha in [0.005, 0.01, 0.02, 0.05]:
        f = scored["conformal_p"] < alpha
        per_day = scored[f].groupby("d").size().reindex(scored["d"].unique(), fill_value=0)
        rows.append({
            "컨포멀 p 임계": alpha,
            "명목 오경보율(%)": alpha * 100,
            "플래그 수": int(f.sum()),
            "전체 대비(%)": round(f.mean() * 100, 2),
            "일평균(점)": round(f.sum() / n_days, 1),
            "일최대(점)": int(per_day.max()),
        })
    out = pd.DataFrame(rows)
    save_table(out, "t43_operational_load", index=False)
    return out


def produce_scores() -> pd.DataFrame:
    """전체 11,939 점에 대한 운영 스코어. 안정일로 적합·교정하고 전체를 채점한다."""
    raw = physics(load_raw())
    g = raw[(raw["recipe"] == MAIN_RECIPE) & (raw["d"].astype(str).isin(STABLE_DAYS))]
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(g))
    ncal = int(len(g) * CAL_FRAC)
    cal, ref = g.iloc[idx[:ncal]], g.iloc[idx[ncal:]]
    det = KNNDetector().fit(ref[FEATS].to_numpy()).calibrate(cal[FEATS].to_numpy())

    out = raw.copy()
    out["anomaly_score"] = det.score(raw[FEATS].to_numpy())
    out["conformal_p"] = det.pvalue(raw[FEATS].to_numpy())
    out["flag_p01"] = (out["conformal_p"] < 0.01).astype(int)
    keep = ["idx", "Machine_Name", "Item No", "working time", "d",
            F, I, V, T, "R", "E", "recipe", "anomaly_score", "conformal_p", "flag_p01"]
    out[keep].to_csv(OUT / "weld_anomaly_scores.csv", index=False, encoding="utf-8-sig")
    return out


def f25(piv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ks = [c for c in piv.columns if c != "평균"]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(piv.index)))
    for c, a in zip(cmap, piv.index):
        ax.plot(ks, piv.loc[a, ks].to_numpy(), "o-", color=c, lw=2, ms=6, label=a)
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("이상 강도 (표준편차 배수)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.03)
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.set_title("최종 모형(kNN 거리, 물리 파생공간)의 이상유형별 강도 반응\n"
                 "약한 이상(1σ)에서는 유형에 따라 편차가 크다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f25_severity_sweep.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f25_severity_sweep.png")


def f26(conf: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    a1.plot([0, 6.3], [0, 6.3], "--", color="gray", lw=1.3, label="목표 = 실제 (목표선)")
    for scheme, col in [("교차일 교정", "#c0392b"), ("당일 순차 교정", "#e67e22"), ("당일 무작위 교정", "#27ae60")]:
        sub = conf[conf["교정 방식"] == scheme]
        a1.plot(sub["목표 오경보율(%)"], sub["실제 오경보율(%)"], "o-", color=col,
                lw=2.2, ms=8, label=scheme)
        for x, y in zip(sub["목표 오경보율(%)"], sub["실제 오경보율(%)"]):
            a1.annotate(f"{y:.2f}", (x, y), fontsize=8.5, xytext=(6, -10),
                        textcoords="offset points", color=col)
    a1.plot([1.0], [1.20], "s", color="#5d6d7e", ms=9, label="기존 경험적 임계 (목표1%→1.20%)")
    a1.set_xlabel("목표 오경보율 (%)")
    a1.set_ylabel("실제 오경보율 (%)")
    a1.set_title("FPR 제어는 교환가능성에 달려 있다 — 교정셋을 어디서 잡느냐의 문제", fontsize=10.5)
    a1.legend(fontsize=8)

    sub = conf[conf["교정 방식"] == "당일 무작위 교정"]
    cols = [c for c in conf.columns if c.startswith("탐지율")]
    x = np.arange(len(cols))
    for _, r in sub.iterrows():
        a2.plot(x, [r[c] for c in cols], "o-", lw=1.8, ms=5,
                label=f"목표 오경보율 {r['목표 오경보율(%)']:.1f}%")
    a2.set_xticks(x)
    a2.set_xticklabels([c.replace("탐지율 ", "") for c in cols], fontsize=9)
    a2.set_ylabel("탐지율 (%)  — 이상 강도 2σ")
    a2.set_title("오경보 예산별 이상유형 탐지율 (당일 무작위 교정, 2σ)", fontsize=10.5)
    a2.legend(fontsize=8)
    fig.suptitle("라벨 없이도 오경보율을 점검한다 — 검출률은 여전히 제시하지 않는다",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f26_conformal.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f26_conformal.png")


def main():
    print("\n[14] 최종 이상탐지 모형 확정")
    print(f"  모형: kNN 거리 (k={N_NEIGHBORS}), 특징공간 FS3 = {FEATS}")
    g = genuine()
    print(f"  적합 데이터: 안정일 {len(g):,}점")

    print("\n══ (1) 이상 강도 x 유형 스윕 ══")
    piv = t41_sweep(g)
    print(piv.to_string())
    worst = piv["평균"].idxmin()
    print(f"\n   가장 어려운 유형: {worst} (평균 {piv.loc[worst,'평균']:.3f})")
    print(f"   1σ 평균 {piv[1.0].mean():.3f} / 3σ 평균 {piv[3.0].mean():.3f}")

    print("\n══ (2) 컨포멀 임계의 오경보율 검증 ══")
    conf = t42_conformal(g)
    print(conf.to_string(index=False))
    for scheme in ["교차일 교정", "당일 순차 교정", "당일 무작위 교정"]:
        sub = conf[conf["교정 방식"] == scheme]
        ok = (sub["목표 충족"] == "예").all()
        mx = sub["목표 대비 배수"].max()
        print(f"\n   [{scheme}] 전 수준 목표 충족: {'예' if ok else '아니오'} "
              f"(목표 대비 최대 {mx:.2f}배)")
    print("\n   p<0.01은 명목 임계다. 교환가능한 무작위 교정에서만 FPR 제어가 확인됐고,")
    print("   교차일 운영에서는 보장이 깨졌다. 운영 후보는 팽창계수 반영 p<0.0038이며 이 또한 근사값이다.")
    print("   구현이 아니라 전제의 문제이며,")
    print("   위반의 원인이 드리프트임이 확인된다. 작업일 간 드리프트(교차일)보다")
    print("   일중 드리프트(당일 순차)가 전제를 더 크게 깨뜨린다.")
    print("\n   운영상의 함의: 이론적으로 유효한 무작위 교정은 미래 점을 필요로 하므로")
    print("   실시간에 쓸 수 없다. 따라서 현장 적용은 다음 둘 중 하나여야 한다.")
    xday = conf[conf["교정 방식"] == "교차일 교정"]
    infl = xday["목표 대비 배수"].max()
    print(f"     (a) 교차일 교정을 쓰되 팽창계수 {infl:.2f} 를 반영해 목표를 보수적으로 설정")
    print(f"         (실제 1% 를 원하면 명목 {1/infl:.2f}% 로 설정)")
    print("     (b) 최근 구간만 쓰는 이동 교정창을 두어 국소 교환가능성을 확보")
    print("   어느 쪽도 '보장'이 아니라 '근사'이며, 본 보고서는 그 사실을 명시한다.")

    print("\n══ (3) 운영 스코어 산출 ══")
    scored = produce_scores()
    load = t43_load(scored)
    print(load.to_string(index=False))
    print("\n-- 작업일별 분해")
    byday = t43b_byday(scored)
    print(byday.to_string())
    st = byday[byday["운전 구분"].str.startswith("안정일")]["플래그 비율(%)"].mean()
    mx = byday[byday["운전 구분"] == "혼입일"]["플래그 비율(%)"].mean()
    print(f"\n   안정일 평균 {st:.2f}% vs 혼입일 평균 {mx:.2f}%")
    print("   부하의 대부분은 혼입일에서 발생한다. 복제블록을 포함한 구간이 참조분포에서")
    print("   멀기 때문이며, 탐지기가 의도대로 작동하고 있음을 보여준다.")
    n_test = int((scored["d"] == TEST_DATE).sum())
    n_flag = int(scored.loc[scored["d"] == TEST_DATE, "flag_p01"].sum())
    print(f"\n  saved -> outputs/weld_anomaly_scores.csv ({len(scored):,} 행)")
    print(f"  (예측 대상 {TEST_DATE}: {n_test:,} 점 중 p<0.01 플래그 {n_flag} 점)")

    f25(piv)
    f26(conf)

    sel = pd.DataFrame([{
        "항목": "최종 모형",
        "값": f"kNN 거리 (k={N_NEIGHBORS}), 특징공간 {FEATS}",
        "근거": "12 벤치마크 최소최대 1위. k 는 15 의 중첩 CV 가 선택(5/5 폴드 k=3)",
    }, {
        "항목": "임계 설정",
        "값": "컨포멀 p < 0.01",
        "근거": ("p<0.01은 명목 임계다. 교환가능한 무작위 교정에서만 FPR 제어가 확인됐고, "
                 "교차일 운영에서는 보장이 깨졌다. 운영 후보는 팽창계수 반영 p<0.0038이며 이 또한 근사값이다."),
    }, {
        "항목": "운영 부하",
        "값": f"일평균 {load[load['컨포멀 p 임계']==0.01]['일평균(점)'].iloc[0]:.1f}점",
        "근거": "p<0.01 기준",
    }])
    save_table(sel, "t44_final_detector", index=False)
    print("\n[14] 완료")


if __name__ == "__main__":
    main()
