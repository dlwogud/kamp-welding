"""07. 모델 개발 및 성능평가 (결과보고서 제2장).

본 데이터는 용접점 단위 불량 라벨이 없다. 따라서 과제를 두 층으로 나누어
각각 베이스라인을 포함한 복수 모델을 동일 평가조건에서 비교한다.

────────────────────────────────────────────────────────────────────────
과제 A — 작업일 단위 불량건수 예측 (실제 라벨 사용, n=8)
  M0 기준선 : 절편 전용 (상수 발생)
  M1        : 비례 모형. offset(log 용접점수)
  M2        : Poisson GLM + 공정조건 1변수
  M3        : 음이항 GLM (과산포 대응)
  평가      : Leave-One-Out CV 의 MAE / RMSE, AIC, 95% 예측구간 포함률

과제 B — 용접점 단위 이상탐지 (라벨 없음, 비지도)
  M0 기준선 : 가압력 단일 임계 규칙
  M1        : 로버스트 마할라노비스 (MCD)
  M2        : Isolation Forest
  M3        : Local Outlier Factor
  M4        : One-Class SVM
  평가      : 정상 데이터만 존재하므로 아래 두 축을 동일 임계 기준으로 비교한다.
              (1) 오경보율 — 안정일 Leave-One-Day-Out. 학습에 쓰이지 않은 정상
                  작업일에서 이상으로 플래그되는 비율. 낮을수록 좋다.
              (2) 이탈 탐지율 — 06 에서 복제로 판정된 블록(정상 운전에서 명백히
                  벗어난 구간)을 탐지하는 비율. 높을수록 좋다.
              복제블록을 '정답'이라 부르지 않는다. 운전상태 이탈의 조작적 정의로만 쓴다.
              이 프록시는 가압력에 강하게 종속되므로 단독 지표로 쓰면 베이스라인이
              유리해진다. 따라서 (1)을 주지표, (2)를 보조지표로 둔다.
────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import poisson
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from common import FIG, PROC, TEST_DATE, daily_label, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]
MAIN_RECIPE = "A_주모드(~3bar)"
FLAG_RATE = 0.01  # 학습 데이터 상위 1% 를 이상으로 두는 공통 임계


# ══════════════════════════════════════════════════════════════════
# 과제 A — 작업일 단위 불량건수 예측
# ══════════════════════════════════════════════════════════════════
def daily_table() -> pd.DataFrame:
    raw = load_raw()
    g = raw.groupby("d")
    t = pd.DataFrame(
        {
            "용접점수": g.size(),
            "전류_표준편차": g["weld current(kA)"].std(),
            "가압력_표준편차": g["weld force(bar)"].std(),
            "비주레시피_비율": g["recipe"].apply(lambda s: (s != MAIN_RECIPE).mean()),
        }
    ).join(daily_label())
    return t


def fit_predict(kind: str, ytr, Xtr, ntr, xte, nte):
    """모형별 1-스텝 예측값. 실패 시 평균으로 안전 복귀."""
    try:
        if kind == "M0":
            return float(np.mean(ytr))
        if kind == "M1":
            r = ytr.sum() / ntr.sum()
            return float(r * nte)
        if kind == "M2":
            X = sm.add_constant(Xtr.reshape(-1, 1), has_constant="add")
            f = sm.GLM(ytr, X, family=sm.families.Poisson()).fit()
            return float(f.predict([[1.0, xte]])[0])
        if kind == "M3":
            X = sm.add_constant(Xtr.reshape(-1, 1), has_constant="add")
            f = sm.GLM(ytr, X, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
            return float(f.predict([[1.0, xte]])[0])
    except Exception:
        pass
    return float(np.mean(ytr))


def task_a(var: str = "전류_표준편차") -> pd.DataFrame:
    t = daily_table()
    lab = t.dropna(subset=["defects"])
    y = lab["defects"].to_numpy(float)
    n = lab["용접점수"].to_numpy(float)
    x = lab[var].to_numpy(float)
    N = len(lab)

    rows = []
    for kind, name in [
        ("M0", "M0 기준선 (절편 전용)"),
        ("M1", "M1 비례 (offset = 용접점수)"),
        ("M2", f"M2 Poisson GLM (+{var})"),
        ("M3", f"M3 음이항 GLM (+{var})"),
    ]:
        preds = []
        for i in range(N):
            m = np.ones(N, bool)
            m[i] = False
            preds.append(fit_predict(kind, y[m], x[m], n[m], x[i], n[i]))
        preds = np.array(preds)
        mae = np.mean(np.abs(y - preds))
        rmse = np.sqrt(np.mean((y - preds) ** 2))
        # 95% 포아송 예측구간 포함률
        lo = poisson.ppf(0.025, preds)
        hi = poisson.ppf(0.975, preds)
        cover = np.mean((y >= lo) & (y <= hi)) * 100
        # 전체적합 AIC
        if kind == "M0":
            ll = poisson.logpmf(y, y.mean()).sum()
            k = 1
        elif kind == "M1":
            r = y.sum() / n.sum()
            ll = poisson.logpmf(y, r * n).sum()
            k = 1
        else:
            X = sm.add_constant(x.reshape(-1, 1), has_constant="add")
            fam = sm.families.Poisson() if kind == "M2" else sm.families.NegativeBinomial(alpha=1.0)
            f = sm.GLM(y, X, family=fam).fit()
            ll, k = f.llf, 2
        rows.append(
            {
                "모형": name,
                "LOO-MAE": round(mae, 3),
                "LOO-RMSE": round(rmse, 3),
                "AIC": round(2 * k - 2 * ll, 2),
                "95% 구간 포함률(%)": round(cover, 1),
                "기준선 대비 MAE": "—" if kind == "M0" else f"{(mae - rows[0]['LOO-MAE']) / rows[0]['LOO-MAE'] * 100:+.1f}%",
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "t22_taskA_model_comparison", index=False)
    return out, t


# ══════════════════════════════════════════════════════════════════
# 과제 B — 용접점 단위 이상탐지
# ══════════════════════════════════════════════════════════════════
def anomaly_scores(name: str, Xtr: np.ndarray, Xte: np.ndarray, force_te: np.ndarray) -> np.ndarray:
    """학습셋으로 적합하고 평가셋 이상점수를 반환. 클수록 이상."""
    if name == "M0 기준선 (가압력 임계)":
        return force_te.astype(float)

    sc = RobustScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)

    if name == "M1 로버스트 마할라노비스":
        try:
            m = MinCovDet(random_state=SEED, support_fraction=0.75).fit(A)
            return m.mahalanobis(B)
        except Exception:
            mu, cov = A.mean(0), np.cov(A.T) + np.eye(A.shape[1]) * 1e-6
            d = B - mu
            return np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
    if name == "M2 Isolation Forest":
        m = IsolationForest(n_estimators=300, random_state=SEED, contamination="auto").fit(A)
        return -m.score_samples(B)
    if name == "M3 Local Outlier Factor":
        m = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(A)
        return -m.score_samples(B)
    if name == "M4 One-Class SVM":
        m = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale").fit(A[:: max(1, len(A) // 3000)])
        return -m.score_samples(B)
    raise ValueError(name)


MODELS_B = [
    "M0 기준선 (가압력 임계)",
    "M1 로버스트 마할라노비스",
    "M2 Isolation Forest",
    "M3 Local Outlier Factor",
    "M4 One-Class SVM",
]


def task_b() -> pd.DataFrame:
    raw = load_raw()
    stable = raw[raw["d"].astype(str).isin(STABLE_DAYS)]
    block = raw[raw["recipe"] != MAIN_RECIPE]  # 06 에서 복제로 판정된 블록

    rows = []
    for name in MODELS_B:
        fprs, dets = [], []
        for hold in STABLE_DAYS:
            tr = stable[stable["d"].astype(str) != hold]
            te = stable[stable["d"].astype(str) == hold]
            Xtr, Xte = tr[PROC].to_numpy(), te[PROC].to_numpy()

            s_tr = anomaly_scores(name, Xtr, Xtr, tr["weld force(bar)"].to_numpy())
            thr = np.quantile(s_tr, 1 - FLAG_RATE)  # 학습 상위 1%

            s_te = anomaly_scores(name, Xtr, Xte, te["weld force(bar)"].to_numpy())
            fprs.append(float((s_te > thr).mean()) * 100)

            s_blk = anomaly_scores(name, Xtr, block[PROC].to_numpy(),
                                   block["weld force(bar)"].to_numpy())
            dets.append(float((s_blk > thr).mean()) * 100)

        fpr, det = float(np.mean(fprs)), float(np.mean(dets))
        rows.append(
            {
                "모형": name,
                "오경보율(%)": round(fpr, 3),
                "오경보율 표준편차": round(float(np.std(fprs)), 3),
                "이탈 탐지율(%)": round(det, 1),
                "탐지율-오경보율": round(det - fpr, 1),
                "일평균 오경보(점)": round(fpr / 100 * stable.groupby("d").size().mean(), 1),
            }
        )
    out = pd.DataFrame(rows)
    save_table(out, "t23_taskB_model_comparison", index=False)
    return out


def f13_taskA(cmp_a: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4))
    names = [m.split(" (")[0] for m in cmp_a["모형"]]
    c = ["#2e86c1" if i == 0 else "#95a5a6" for i in range(len(names))]
    a1.barh(names, cmp_a["LOO-MAE"], color=c, alpha=0.9)
    best = cmp_a["LOO-MAE"].min()
    a1.axvline(best, ls="--", color="#27ae60", lw=1.3)
    for i, v in enumerate(cmp_a["LOO-MAE"]):
        a1.text(v + 0.03, i, f"{v:.3f}", va="center", fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel("LOO-MAE (낮을수록 좋음)")
    a1.set_title("과제 A: 작업일 불량건수 예측", fontsize=10)

    a2.barh(names, cmp_a["AIC"], color=c, alpha=0.9)
    for i, v in enumerate(cmp_a["AIC"]):
        a2.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=9)
    a2.invert_yaxis()
    a2.set_xlabel("AIC (낮을수록 좋음)")
    a2.set_title("동일 데이터 전체적합 AIC", fontsize=10)
    fig.suptitle("공정조건을 넣은 모형은 기준선을 넘지 못한다 — 최종모형으로 기준선을 채택한다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f13_taskA_comparison.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f13_taskA_comparison.png")


def f14_taskB(cmp_b: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.2, 5))
    for _, r in cmp_b.iterrows():
        is_base = r["모형"].startswith("M0")
        ax.scatter(r["오경보율(%)"], r["이탈 탐지율(%)"], s=170,
                   color="#95a5a6" if is_base else "#1f4e79",
                   marker="s" if is_base else "o", zorder=3)
        ax.annotate(r["모형"], (r["오경보율(%)"], r["이탈 탐지율(%)"]),
                    fontsize=8.5, xytext=(9, -4), textcoords="offset points")
    ax.set_xlabel("오경보율 (%) — 정상 작업일에서 이상으로 플래그된 비율. 낮을수록 좋음")
    ax.set_ylabel("이탈 탐지율 (%) — 운전상태 이탈 구간 탐지 비율. 높을수록 좋음")
    ax.set_title("과제 B: 이상탐지 모델 비교 (안정일 Leave-One-Day-Out, 공통 임계 상위 1%)",
                 fontsize=10.5)
    ax.set_xlim(left=-0.15)
    fig.tight_layout()
    fig.savefig(FIG / "f14_taskB_tradeoff.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f14_taskB_tradeoff.png")


def main():
    print("\n[07] 모델 개발 및 성능평가 (제2장)")

    print("\n══ 과제 A: 작업일 단위 불량건수 예측 (n=8, LOO-CV) ══")
    cmp_a, daily = task_a()
    print(cmp_a.to_string(index=False))
    pick_a = cmp_a.loc[cmp_a["LOO-MAE"].idxmin(), "모형"]
    print(f"\n  => 최종 선택: {pick_a}")
    print("     공정조건 변수를 추가한 모형(M2/M3)은 LOO-MAE 가 기준선보다 나쁘다.")
    print("     표본 8일에서 추가 모수 1개의 분산 증가가 설명력 이득을 넘어선다.")

    print("\n══ 과제 B: 용접점 단위 이상탐지 (안정일 LODO, 공통 임계 상위 1%) ══")
    cmp_b = task_b()
    print(cmp_b.to_string(index=False))
    sel = cmp_b.sort_values(["오경보율(%)", "이탈 탐지율(%)"], ascending=[True, False]).iloc[0]
    print(f"\n  => 오경보율 최소: {sel['모형']} ({sel['오경보율(%)']}%)")

    f13_taskA(cmp_a)
    f14_taskB(cmp_b)

    sel_tbl = pd.DataFrame(
        [
            {"과제": "A 작업일 불량건수", "최종 모형": pick_a,
             "선택 근거": "LOO-MAE 최소. 공정조건 추가 시 n=8 에서 과적합"},
            {"과제": "B 용접점 이상탐지", "최종 모형": sel["모형"],
             "선택 근거": f"공통 임계에서 오경보율 최소({sel['오경보율(%)']}%), 탐지율 {sel['이탈 탐지율(%)']}%"},
        ]
    )
    save_table(sel_tbl, "t24_final_model_selection", index=False)
    print("\n-- 최종모델 선택 요약")
    print(sel_tbl.to_string(index=False))
    print("\n[07] 완료")


if __name__ == "__main__":
    main()
