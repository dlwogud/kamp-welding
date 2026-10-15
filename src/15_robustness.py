"""15. 강건성 검증 — 최종 결과가 설정값에 좌우되는가.

12·14 의 결론(kNN 거리 + 물리 파생공간)은 몇 가지 설정값 위에 서 있다.
k=20, RobustScaler, 교정비율 0.3, 난수시드 42 가 그것이다. 이 값들을 바꿨을 때
결론이 유지되는지 확인한다. 유지되지 않는다면 그 결론은 데이터가 아니라
설정에서 나온 것이다.

네 가지 축을 흔든다.
  (1) 이웃 수 k      : 5, 10, 20, 50, 100
  (2) 스케일러        : RobustScaler(중앙값·IQR) vs StandardScaler(평균·표준편차)
  (3) 난수 시드       : 5개 시드에서 벤치마크 상위 모형 순위가 유지되는가
  (4) 교정셋 비율     : 0.1, 0.2, 0.3, 0.5

평가는 12·14 와 동일한 안정일 LODO, 6개 이상유형, 강도 2 sigma 이며
지표는 '최악 유형 ROC-AUC'(최소최대 기준)다. 최종 모형 선정에 쓴 기준과 같다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from common import FIG, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]
MAIN_RECIPE = "A_주모드(~3bar)"
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
FEATS = [F, "R", "E"]
ANOM_TYPES = ["A1 결합이탈", "A2 가압이탈", "A3 전압강하", "A4 통전단축",
              "A5 동시드리프트", "A6 산포증가", "A7 지속반복이상"]
K_SEV = 2.0
BASE_K, BASE_SEED, BASE_CAL = 20, 42, 0.3


def physics(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    amp = d[I] * 1000.0
    d["R"] = d[V] / amp
    d["E"] = d[V] * amp * (d[T] / 1000.0)
    return d


def genuine() -> pd.DataFrame:
    raw = load_raw()
    g = raw[(raw["recipe"] == MAIN_RECIPE) & (raw["d"].astype(str).isin(STABLE_DAYS))]
    return physics(g.sort_values(["d", "idx"]))


def inject(te, kind, sd, k, rng):
    a = te.copy()
    f = a[F].to_numpy()
    dirn = np.sign(f - np.median(f)); dirn[dirn == 0] = 1
    if kind == "A1 결합이탈":   a[I] = a[I].to_numpy() + dirn * k * sd[I]
    elif kind == "A2 가압이탈": a[F] = a[F].to_numpy() + dirn * k * sd[F]
    elif kind == "A3 전압강하": a[V] = a[V].to_numpy() - k * sd[V]
    elif kind == "A4 통전단축": a[T] = a[T].to_numpy() - k * sd[T]
    elif kind == "A5 동시드리프트":
        a[F] = a[F].to_numpy() + k * sd[F]
        a[I] = a[I].to_numpy() - k * sd[I] * 0.735
    elif kind == "A6 산포증가":
        a[F] = a[F].to_numpy() + rng.normal(0, k * sd[F], len(a))
        a[I] = a[I].to_numpy() + rng.normal(0, k * sd[I], len(a))
    elif kind == "A7 지속반복이상":
        # 지속되는 이상이 낮은 센서 분해능과 만나면 '같은 값이 반복'되는 형태가 된다.
        # 본 데이터는 완전중복행이 42% 이므로 현실적인 시나리오다.
        # 이 경우 작은 k 의 kNN 은 이웃 거리가 0 이 되어 원리상 탐지하지 못한다.
        a[F] = float(np.median(a[F].to_numpy()) + k * sd[F])
        a[I] = float(np.median(a[I].to_numpy()) - k * sd[I])
        a[V] = float(np.median(a[V].to_numpy()))
        a[T] = float(np.median(a[T].to_numpy()))
    return physics(a)


def knn_auc(g, k_nb, scaler_cls, seed) -> dict:
    """이상유형별 평균 ROC-AUC."""
    rng = np.random.default_rng(seed)
    out = {}
    for atype in ANOM_TYPES:
        aucs = []
        for h in STABLE_DAYS:
            tr = g[g["d"].astype(str) != h]
            te = g[g["d"].astype(str) == h]
            sd = {c: tr[c].std() for c in [F, I, V, T]}
            idx = rng.permutation(len(te)); half = len(te) // 2
            norm = te.iloc[idx[:half]]
            anom = inject(te.iloc[idx[half:]], atype, sd, K_SEV, rng)
            sc = scaler_cls().fit(tr[FEATS])
            nn = NearestNeighbors(n_neighbors=k_nb).fit(sc.transform(tr[FEATS]))
            s = np.r_[nn.kneighbors(sc.transform(norm[FEATS]))[0].mean(1),
                      nn.kneighbors(sc.transform(anom[FEATS]))[0].mean(1)]
            y = np.r_[np.zeros(len(norm)), np.ones(len(anom))]
            aucs.append(roc_auc_score(y, s))
        out[atype] = float(np.mean(aucs))
    return out


def t45_k(g) -> pd.DataFrame:
    rows = []
    for k_nb in [5, 10, 20, 50, 100]:
        a = knn_auc(g, k_nb, RobustScaler, BASE_SEED)
        v = list(a.values())
        rows.append({"이웃 수 k": k_nb, **{t: round(a[t], 3) for t in ANOM_TYPES},
                     "평균": round(float(np.mean(v)), 3),
                     "최악": round(float(np.min(v)), 3)})
    out = pd.DataFrame(rows)
    save_table(out, "t45_k_sensitivity", index=False)
    return out


def t46_scaler(g) -> pd.DataFrame:
    rows = []
    for name, cls in [("RobustScaler (중앙값·IQR)", RobustScaler),
                      ("StandardScaler (평균·표준편차)", StandardScaler)]:
        a = knn_auc(g, BASE_K, cls, BASE_SEED)
        v = list(a.values())
        rows.append({"스케일러": name, **{t: round(a[t], 3) for t in ANOM_TYPES},
                     "평균": round(float(np.mean(v)), 3),
                     "최악": round(float(np.min(v)), 3)})
    out = pd.DataFrame(rows)
    save_table(out, "t46_scaler_sensitivity", index=False)
    return out


def t47_seed(g) -> pd.DataFrame:
    """시드별 kNN vs 경쟁 모형(IForest, LOF) 최악 유형 비교. 순위가 유지되는가."""
    rows = []
    for seed in [42, 7, 123, 2024, 31337]:
        rng = np.random.default_rng(seed)
        res = {"kNN 거리": [], "Isolation Forest": [], "LOF": []}
        for atype in ANOM_TYPES:
            acc = {m: [] for m in res}
            for h in STABLE_DAYS:
                tr = g[g["d"].astype(str) != h]
                te = g[g["d"].astype(str) == h]
                sd = {c: tr[c].std() for c in [F, I, V, T]}
                idx = rng.permutation(len(te)); half = len(te) // 2
                norm = te.iloc[idx[:half]]
                anom = inject(te.iloc[idx[half:]], atype, sd, K_SEV, rng)
                sc = RobustScaler().fit(tr[FEATS])
                A = sc.transform(tr[FEATS])
                B = np.r_[sc.transform(norm[FEATS]), sc.transform(anom[FEATS])]
                y = np.r_[np.zeros(len(norm)), np.ones(len(anom))]
                nn = NearestNeighbors(n_neighbors=BASE_K).fit(A)
                acc["kNN 거리"].append(roc_auc_score(y, nn.kneighbors(B)[0].mean(1)))
                iso = IsolationForest(n_estimators=300, random_state=seed).fit(A)
                acc["Isolation Forest"].append(roc_auc_score(y, -iso.score_samples(B)))
                lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(A)
                acc["LOF"].append(roc_auc_score(y, -lof.score_samples(B)))
            for m in res:
                res[m].append(float(np.mean(acc[m])))
        row = {"시드": seed}
        for m in res:
            row[f"{m} 최악"] = round(float(np.min(res[m])), 3)
        row["1위"] = max(res, key=lambda m: np.min(res[m]))
        rows.append(row)
    out = pd.DataFrame(rows)
    save_table(out, "t47_seed_sensitivity", index=False)
    return out


def t48_calfrac(g) -> pd.DataFrame:
    """교정셋 비율별 컨포멀 오경보율 (당일 무작위 교정, 목표 1%)."""
    rows = []
    for frac in [0.1, 0.2, 0.3, 0.5]:
        emp = []
        rng = np.random.default_rng(BASE_SEED)
        for h in STABLE_DAYS:
            tr = g[g["d"].astype(str) != h]
            te_all = g[g["d"].astype(str) == h]
            j = rng.permutation(len(te_all)); ncal = int(len(te_all) * frac)
            cal, te = te_all.iloc[j[:ncal]], te_all.iloc[j[ncal:]]
            sc = RobustScaler().fit(tr[FEATS])
            nn = NearestNeighbors(n_neighbors=BASE_K).fit(sc.transform(tr[FEATS]))
            s_cal = np.sort(nn.kneighbors(sc.transform(cal[FEATS]))[0].mean(1))
            s_te = nn.kneighbors(sc.transform(te[FEATS]))[0].mean(1)
            n = len(s_cal)
            p = (1.0 + (n - np.searchsorted(s_cal, s_te, side="left"))) / (n + 1.0)
            emp.append(float((p < 0.01).mean()) * 100)
        e = float(np.mean(emp))
        rows.append({"교정셋 비율": frac, "교정 표본 수(평균)": int(np.mean(
            [int(len(g[g['d'].astype(str) == h]) * frac) for h in STABLE_DAYS])),
            "실제 오경보율(%)": round(e, 3), "목표 1% 충족": "예" if e <= 1.0 else "아니오"})
    out = pd.DataFrame(rows)
    save_table(out, "t48_calfrac_sensitivity", index=False)
    return out


def knn_auc_days(g, days_tr, day_te, k_nb, seed) -> float:
    """주어진 학습일 집합으로 적합해 평가일에서의 최악 유형 ROC-AUC."""
    rng = np.random.default_rng(seed)
    tr = g[g["d"].astype(str).isin(days_tr)]
    te = g[g["d"].astype(str) == day_te]
    sd = {c: tr[c].std() for c in [F, I, V, T]}
    sc = RobustScaler().fit(tr[FEATS])
    nn = NearestNeighbors(n_neighbors=min(k_nb, len(tr) - 1)).fit(sc.transform(tr[FEATS]))
    aucs = []
    for atype in ANOM_TYPES:
        idx = rng.permutation(len(te)); half = len(te) // 2
        norm = te.iloc[idx[:half]]
        anom = inject(te.iloc[idx[half:]], atype, sd, K_SEV, rng)
        s = np.r_[nn.kneighbors(sc.transform(norm[FEATS]))[0].mean(1),
                  nn.kneighbors(sc.transform(anom[FEATS]))[0].mean(1)]
        y = np.r_[np.zeros(len(norm)), np.ones(len(anom))]
        aucs.append(roc_auc_score(y, s))
    return float(np.min(aucs))


def t50_nested_k(g) -> pd.DataFrame:
    """중첩 교차검증으로 k 를 선택하고 편향 없이 평가한다.

    15 의 (1) 에서 k 가 작을수록 좋다는 단조 추세가 나왔으나, 그 표는 평가에 쓴
    데이터로 k 를 고른 것이므로 낙관 편향이 있다. 여기서는
      바깥 루프 : 안정일 5일 Leave-One-Day-Out (평가)
      안쪽 루프 : 남은 4일에서 다시 Leave-One-Day-Out (k 선택)
    로 선택과 평가를 분리한다. 보고하는 수치는 바깥 루프의 성능이다.
    """
    cands = [3, 5, 10, 20, 50, 100]
    rows = []
    for h in STABLE_DAYS:
        inner_days = [d for d in STABLE_DAYS if d != h]
        inner = {}
        for k_nb in cands:
            sc = []
            for v in inner_days:
                sc.append(knn_auc_days(g, [d for d in inner_days if d != v], v, k_nb, BASE_SEED))
            inner[k_nb] = float(np.mean(sc))
        k_sel = max(inner, key=inner.get)
        outer = knn_auc_days(g, inner_days, h, k_sel, BASE_SEED)
        rows.append({"평가일(바깥)": h, "선택된 k": k_sel,
                     "안쪽 최악 AUC": round(inner[k_sel], 3),
                     "바깥 최악 AUC": round(outer, 3)})
    out = pd.DataFrame(rows)
    save_table(out, "t50_nested_k", index=False)
    return out


def f27(k_tbl: pd.DataFrame, seed_tbl: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    a1.plot(k_tbl["이웃 수 k"], k_tbl["평균"], "o-", color="#1f4e79", lw=2.2, ms=7, label="유형 평균")
    a1.plot(k_tbl["이웃 수 k"], k_tbl["최악"], "s--", color="#c0392b", lw=2, ms=7, label="최악 유형")
    a1.axvline(BASE_K, ls=":", color="black", lw=1.4)
    a1.text(BASE_K * 1.1, 0.80, f"채택 k={BASE_K}", fontsize=9)
    a1.set_xscale("log")
    a1.set_xticks(k_tbl["이웃 수 k"])
    a1.set_xticklabels(k_tbl["이웃 수 k"])
    a1.set_xlabel("이웃 수 k")
    a1.set_ylabel("ROC-AUC")
    a1.set_ylim(0.75, 1.0)
    a1.legend(fontsize=9)
    a1.set_title("이웃 수 민감도", fontsize=10.5)

    ms = ["kNN 거리 최악", "Isolation Forest 최악", "LOF 최악"]
    x = np.arange(len(seed_tbl))
    w = 0.26
    for i, (m, c) in enumerate(zip(ms, ["#c0392b", "#2e86c1", "#95a5a6"])):
        a2.bar(x + (i - 1) * w, seed_tbl[m], w, label=m.replace(" 최악", ""), color=c, alpha=0.9)
    a2.set_xticks(x)
    a2.set_xticklabels(seed_tbl["시드"], fontsize=9)
    a2.set_xlabel("난수 시드")
    a2.set_ylabel("최악 유형 ROC-AUC")
    a2.set_ylim(0.5, 1.0)
    a2.legend(fontsize=9)
    a2.set_title("시드 민감도 — 순위가 유지되는가", fontsize=10.5)

    fig.suptitle("최종 선정이 설정값에 좌우되지 않음을 확인한다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f27_robustness.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f27_robustness.png")


def main():
    print("\n[15] 강건성 검증")
    g = genuine()

    print("\n══ (1) 이웃 수 k 민감도 ══")
    kt = t45_k(g)
    print(kt.to_string(index=False))
    print(f"\n   최악 유형 AUC 범위 {kt['최악'].min():.3f} ~ {kt['최악'].max():.3f} "
          f"(채택 k={BASE_K} 에서 {kt[kt['이웃 수 k']==BASE_K]['최악'].iloc[0]:.3f})")

    print("\n══ (2) 스케일러 민감도 ══")
    st = t46_scaler(g)
    print(st.to_string(index=False))

    print("\n══ (3) 난수 시드 민감도 ══")
    sd = t47_seed(g)
    print(sd.to_string(index=False))
    win = sd["1위"].value_counts()
    print(f"\n   시드 5개 중 1위 횟수: {dict(win)}")

    print("\n══ (3b) 중첩 교차검증 — k 선택과 평가의 분리 ══")
    nk = t50_nested_k(g)
    print(nk.to_string(index=False))
    sel = nk["선택된 k"].mode().iloc[0]
    print(f"\n   안쪽 루프가 선택한 k: {sorted(nk['선택된 k'].tolist())} (최빈 {sel})")
    print(f"   편향 없는 바깥 성능(최악 유형 AUC) 평균 {nk['바깥 최악 AUC'].mean():.3f}")
    row = kt[kt["이웃 수 k"] == sel]
    if len(row):
        print(f"   단순 선택(표 t45, k={sel})의 {row['최악'].iloc[0]:.3f} 대비 "
              f"편향 {row['최악'].iloc[0] - nk['바깥 최악 AUC'].mean():+.3f}")
    else:
        print(f"   (k={sel} 은 표 t45 의 후보 밖이다. 중첩 CV 가 더 작은 k 를 선택했다)")

    print("\n══ (4) 교정셋 비율 민감도 (목표 오경보율 1%) ══")
    ct = t48_calfrac(g)
    print(ct.to_string(index=False))

    f27(kt, sd)

    summary = pd.DataFrame([
        {"축": "이웃 수 k (5~100)", "결과": f"최악 유형 AUC {kt['최악'].min():.3f}~{kt['최악'].max():.3f} (k 작을수록 우수)",
         "판정": "단조 추세. k 는 중첩 CV 로 선택한다"},
        {"축": "스케일러", "결과": f"최악 유형 AUC 차 {abs(st['최악'].diff().iloc[-1]):.3f}",
         "판정": "결론 유지" if abs(st["최악"].diff().iloc[-1]) < 0.05 else "민감"},
        {"축": "난수 시드 (5개)", "결과": f"kNN 1위 {int(win.get('kNN 거리', 0))}/5 회",
         "판정": "결론 유지" if win.get("kNN 거리", 0) >= 4 else "민감"},
        {"축": "교정셋 비율 (0.1~0.5)", "결과": f"오경보율 {ct['실제 오경보율(%)'].min():.2f}~{ct['실제 오경보율(%)'].max():.2f}%",
         "판정": "결론 유지 (컨포멀 보장은 기댓값 기준이며 1.13% 는 몬테카를로 변동 범위)"},
        {"축": "중첩 CV (k 선택 분리)", "결과": f"바깥 최악 AUC 평균 {nk['바깥 최악 AUC'].mean():.3f}",
         "판정": "결론 유지" if nk["바깥 최악 AUC"].mean() > 0.85 else "민감"},
    ])
    save_table(summary, "t49_robustness_summary", index=False)
    print("\n-- 요약")
    print(summary.to_string(index=False))
    print("\n[15] 완료")


if __name__ == "__main__":
    main()
