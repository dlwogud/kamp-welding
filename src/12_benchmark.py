"""12. 이상탐지 종합 벤치마크 — 다중 이상유형 x 다중 특징공간 x 다중 모형.

08·11 의 평가에는 구조적 약점이 있었다. 합성 이상이 '전류를 결합 반대방향으로 이동'
한 가지뿐이었고, 최종 선정된 모형(결합)은 바로 그 유형에 맞춰 설계되어 있었다.
한 가지 이상유형으로 모형을 고르면 그 유형에 특화된 모형이 뽑히는 것이 당연하다.

본 절은 평가를 세 축으로 확장한다.

(1) 이상유형 6종 — 현장에서 실제로 보고되는 고장모드를 각각 모사한다
    A1 결합이탈   전류만 결합 반대방향 이동      전극 마모, 전류 제어 이상
    A2 가압이탈   가압력만 이동                 가압 기구 고착·공압 저하
    A3 전압강하   전압만 하강                   2차회로 접촉불량, 케이블 열화
    A4 통전단축   통전시간만 단축               타이머 이상, 조기 차단
    A5 동시드리프트 가압력·전류가 결합을 유지한 채 함께 이동   설정값 변경, 보정 누락
    A6 산포증가   값은 유지하고 분산만 증가      기계적 유격, 접촉 불안정

    A5 는 특히 중요하다. 결합을 유지한 채 이동하므로 '결합 잔차'로는 원리상 탐지되지
    않는다. 본 분석의 최종모형이 가진 사각지대를 드러내기 위해 일부러 포함한다.

(2) 특징공간 3종
    FS1 가압력·전류            — 08·11 에서 쓴 공간
    FS2 원 4변수               — 전압·통전시간까지 포함
    FS3 물리 파생              — 가압력, 동저항 R=V/I, 입열 E=V*I*t
       RSW 품질감시의 고전적 접근은 동저항 모니터링이다. 다만 본 데이터는 전압이
       거의 일정(CV 0.155%)하므로 R 은 전류의 단조변환에 가깝다. 독립 정보가 적을
       것으로 예상되나, 통전시간이 들어간 E 는 다른 축을 더한다. 확인한다.

(3) 모형 9종 (문헌 표준 포함)
    베이스라인 : 단변량 임계(가압력), 단변량 임계(전류)
    통계적     : Hotelling T^2, 로버스트 마할라노비스(MCD), PCA 재구성오차
    기계학습   : Isolation Forest, LOF, One-Class SVM, kNN 거리
    구조기반   : 정상운전 모델 잔차(11), 결합(IForest+잔차)

평가는 08 과 동일한 안정일 Leave-One-Day-Out 이며, 지표는 임계값에 의존하지 않는
ROC-AUC 와 평균정밀도(AP)다. 모형 선정은 '한 유형 최고'가 아니라
'유형 전반에서 최악이 가장 덜 나쁜 것'(최소최대 기준)으로 한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from common import FIG, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]
MAIN_RECIPE = "A_주모드(~3bar)"
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"

K = 2.0  # 이상 강도 (각 변수의 학습셋 표준편차 배수). 모든 유형에 동일 적용.

FEATSPACES = {
    "FS1 가압력·전류": [F, I],
    "FS2 원 4변수": [F, I, V, T],
    "FS3 물리 파생": [F, "R", "E"],
}
ANOM_TYPES = ["A1 결합이탈", "A2 가압이탈", "A3 전압강하", "A4 통전단축",
              "A5 동시드리프트", "A6 산포증가"]
MODELS = ["M0 단변량(가압력)", "M1 단변량(전류)", "M2 Hotelling T2",
          "M3 마할라노비스(MCD)", "M4 PCA 재구성오차", "M5 Isolation Forest",
          "M6 LOF", "M7 One-Class SVM", "M8 kNN 거리", "M9 정상운전모델 잔차",
          "M10 결합(IForest+잔차)"]


def physics(df: pd.DataFrame) -> pd.DataFrame:
    """RSW 물리 파생량. 전류 kA->A, 통전시간 ms->s 로 환산."""
    d = df.copy()
    amp = d[I] * 1000.0
    sec = d[T] / 1000.0
    d["R"] = d[V] / amp          # 동저항 (ohm)
    d["E"] = d[V] * amp * sec    # 입열 에너지 (J)
    return d


def genuine() -> pd.DataFrame:
    raw = load_raw()
    g = raw[(raw["recipe"] == MAIN_RECIPE) & (raw["d"].astype(str).isin(STABLE_DAYS))].copy()
    return physics(g.sort_values(["d", "idx"]))


def inject(te: pd.DataFrame, kind: str, sd: dict, rng) -> pd.DataFrame:
    """이상유형별 합성 이상 주입. 주입 후 물리 파생량을 재계산한다."""
    a = te.copy()
    f = a[F].to_numpy()
    direction = np.sign(f - np.median(f))
    direction[direction == 0] = 1

    if kind == "A1 결합이탈":
        # 결합(가압력↑ -> 전류↓)과 반대로 전류를 이동. 가압력은 유지.
        a[I] = a[I].to_numpy() + direction * K * sd[I]
    elif kind == "A2 가압이탈":
        a[F] = a[F].to_numpy() + direction * K * sd[F]
    elif kind == "A3 전압강하":
        a[V] = a[V].to_numpy() - K * sd[V]
    elif kind == "A4 통전단축":
        a[T] = a[T].to_numpy() - K * sd[T]
    elif kind == "A5 동시드리프트":
        # 결합관계를 유지한 채 동시 이동. 잔차 기반 탐지의 사각지대.
        a[F] = a[F].to_numpy() + K * sd[F]
        a[I] = a[I].to_numpy() - K * sd[I] * 0.735  # 관측된 결합 강도만큼 반대로
    elif kind == "A6 산포증가":
        a[F] = a[F].to_numpy() + rng.normal(0, K * sd[F], len(a))
        a[I] = a[I].to_numpy() + rng.normal(0, K * sd[I], len(a))
    else:
        raise ValueError(kind)
    return physics(a)


# ══════════════════════════════════════════════════════════════════
def fit_score(model: str, tr: pd.DataFrame, ev: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """학습셋으로 적합하고 평가셋 이상점수 반환. 클수록 이상."""
    Xtr, Xev = tr[cols].to_numpy(float), ev[cols].to_numpy(float)

    if model == "M0 단변량(가압력)":
        return _uni(tr[F].to_numpy(), ev[F].to_numpy())
    if model == "M1 단변량(전류)":
        return _uni(tr[I].to_numpy(), ev[I].to_numpy())

    sc = RobustScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xev)

    if model == "M2 Hotelling T2":
        mu = A.mean(0)
        S = np.cov(A.T) + np.eye(A.shape[1]) * 1e-9
        d = B - mu
        return np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
    if model == "M3 마할라노비스(MCD)":
        try:
            return MinCovDet(random_state=SEED, support_fraction=0.8).fit(A).mahalanobis(B)
        except Exception:
            mu = A.mean(0); S = np.cov(A.T) + np.eye(A.shape[1]) * 1e-9
            d = B - mu
            return np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
    if model == "M4 PCA 재구성오차":
        k = max(1, A.shape[1] - 1)
        p = PCA(n_components=k, random_state=SEED).fit(A)
        return ((B - p.inverse_transform(p.transform(B))) ** 2).sum(1)
    if model == "M5 Isolation Forest":
        return -IsolationForest(n_estimators=300, random_state=SEED).fit(A).score_samples(B)
    if model == "M6 LOF":
        return -LocalOutlierFactor(n_neighbors=35, novelty=True).fit(A).score_samples(B)
    if model == "M7 One-Class SVM":
        step = max(1, len(A) // 2000)
        return -OneClassSVM(kernel="rbf", nu=0.02, gamma="scale").fit(A[::step]).score_samples(B)
    if model == "M8 kNN 거리":
        nn = NearestNeighbors(n_neighbors=20).fit(A)
        return nn.kneighbors(B)[0].mean(1)
    if model in ("M9 정상운전모델 잔차", "M10 결합(IForest+잔차)"):
        res = _residual(tr, ev, cols)
        if model == "M9 정상운전모델 잔차":
            return res
        iso = -IsolationForest(n_estimators=300, random_state=SEED).fit(A).score_samples(B)
        return (_pct(iso) + _pct(res)) / 2
    raise ValueError(model)


def _uni(tr: np.ndarray, ev: np.ndarray) -> np.ndarray:
    med = np.median(tr)
    mad = np.median(np.abs(tr - med)) or 1e-12
    return np.abs(ev - med) / mad


def _pct(a: np.ndarray) -> np.ndarray:
    return pd.Series(a).rank(pct=True).to_numpy()


def _residual(tr: pd.DataFrame, ev: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """정상운전 모델(전류 = f(나머지 변수))의 표준화 잔차 절대값.

    목표변수가 특징공간에 없으면(FS3 는 전류를 직접 쓰지 않음) 첫 비-가압력 변수를
    목표로 삼는다. 어떤 공간에서도 '한 변수를 나머지로 예측하고 그 이탈을 본다'는
    구조는 동일하게 성립한다.
    """
    tgt = I if I in cols else [c for c in cols if c != F][0]
    xs = [c for c in cols if c != tgt]
    m = HistGradientBoostingRegressor(random_state=SEED).fit(tr[xs], tr[tgt])
    s = (tr[tgt].to_numpy() - m.predict(tr[xs])).std() or 1e-12
    return np.abs((ev[tgt].to_numpy() - m.predict(ev[xs])) / s)


# ══════════════════════════════════════════════════════════════════
def run() -> pd.DataFrame:
    g = genuine()
    rng = np.random.default_rng(SEED)
    rows = []

    for fs_name, cols in FEATSPACES.items():
        for atype in ANOM_TYPES:
            per_model = {m: {"auc": [], "ap": []} for m in MODELS}
            for h in STABLE_DAYS:
                tr = g[g["d"].astype(str) != h]
                te = g[g["d"].astype(str) == h]
                sd = {c: tr[c].std() for c in [F, I, V, T]}

                n = len(te)
                idx = rng.permutation(n)
                half = n // 2
                norm = te.iloc[idx[:half]]
                anom = inject(te.iloc[idx[half:]], atype, sd, rng)
                ev = pd.concat([norm, anom])
                y = np.r_[np.zeros(len(norm)), np.ones(len(anom))]

                for m in MODELS:
                    try:
                        s = fit_score(m, tr, ev, cols)
                        if not np.all(np.isfinite(s)):
                            s = np.nan_to_num(s, nan=0.0, posinf=1e12, neginf=0.0)
                        per_model[m]["auc"].append(roc_auc_score(y, s))
                        per_model[m]["ap"].append(average_precision_score(y, s))
                    except Exception:
                        per_model[m]["auc"].append(np.nan)
                        per_model[m]["ap"].append(np.nan)

            for m in MODELS:
                rows.append({
                    "특징공간": fs_name, "이상유형": atype, "모형": m,
                    "ROC-AUC": round(float(np.nanmean(per_model[m]["auc"])), 3),
                    "AP": round(float(np.nanmean(per_model[m]["ap"])), 3),
                })
    full = pd.DataFrame(rows)
    save_table(full, "t34_benchmark_full", index=False)
    return full


def summarize(full: pd.DataFrame):
    # 특징공간별 평균 (어느 공간이 유리한가)
    fs = full.pivot_table(index="특징공간", values="ROC-AUC", aggfunc="mean").round(3)
    fs["최악 유형 평균"] = full.groupby(["특징공간", "이상유형"])["ROC-AUC"].mean() \
                                .groupby("특징공간").min().round(3)
    save_table(fs, "t35_by_featurespace")

    # 최적 특징공간에서 모형 x 이상유형
    best_fs = fs["ROC-AUC"].idxmax()
    sub = full[full["특징공간"] == best_fs]
    piv = sub.pivot_table(index="모형", columns="이상유형", values="ROC-AUC")
    piv = piv.reindex(MODELS)[ANOM_TYPES]
    piv["평균"] = piv.mean(axis=1).round(3)
    piv["최악"] = piv[ANOM_TYPES].min(axis=1).round(3)
    piv = piv.round(3)
    save_table(piv, "t36_model_by_anomtype")
    return fs, best_fs, piv


def f22_heatmap(piv: pd.DataFrame, best_fs: str):
    d = piv[ANOM_TYPES]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(d.to_numpy(), cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(ANOM_TYPES)))
    ax.set_xticklabels(ANOM_TYPES, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(d.index)))
    ax.set_yticklabels(d.index, fontsize=9)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                    color="black" if 0.5 < v < 0.95 else "white")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="ROC-AUC")
    ax.set_title(f"이상유형별 모형 성능 ({best_fs})\n"
                 "한 가지 유형으로 모형을 고르면 그 유형에 특화된 모형이 뽑힌다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f22_benchmark_heatmap.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f22_benchmark_heatmap.png")


def f23_minimax(piv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9.2, 5))
    d = piv.sort_values("최악")
    y = np.arange(len(d))
    ax.barh(y, d["평균"], 0.42, label="유형 평균", color="#95a5a6", alpha=0.9)
    ax.barh(y + 0.42, d["최악"], 0.42, label="최악 유형", color="#c0392b", alpha=0.9)
    ax.set_yticks(y + 0.21)
    ax.set_yticklabels(d.index, fontsize=9)
    ax.axvline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("평균이 높아도 특정 이상유형에서 무너지는 모형이 있다\n"
                 "최종 선정은 '최악 유형이 가장 덜 나쁜 것' 기준", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f23_minimax.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f23_minimax.png")


def main():
    print("\n[12] 이상탐지 종합 벤치마크")
    print(f"  특징공간 {len(FEATSPACES)}종 x 이상유형 {len(ANOM_TYPES)}종 x 모형 {len(MODELS)}종 "
          f"x LODO {len(STABLE_DAYS)}폴드")
    print("  (수 분 소요)")

    full = run()
    fs, best_fs, piv = summarize(full)

    print("\n-- 특징공간별 평균 ROC-AUC")
    print(fs.to_string())
    print(f"\n  => 최적 특징공간: {best_fs}")

    print(f"\n-- 모형 x 이상유형 ({best_fs})")
    print(piv.to_string())

    best = piv["최악"].idxmax()
    print(f"\n  => 최소최대 기준 최종 선정: {best}")
    print(f"     최악 유형 ROC-AUC {piv.loc[best,'최악']:.3f}, 평균 {piv.loc[best,'평균']:.3f}")
    top_mean = piv["평균"].idxmax()
    if top_mean != best:
        print(f"     (평균 1위는 {top_mean} {piv.loc[top_mean,'평균']:.3f} 이나 "
              f"최악 유형 {piv.loc[top_mean,'최악']:.3f} 로 특정 유형에서 취약)")

    f22_heatmap(piv, best_fs)
    f23_minimax(piv)

    sel = pd.DataFrame([{
        "항목": "최적 특징공간", "값": best_fs,
        "근거": f"전 이상유형 평균 ROC-AUC {fs.loc[best_fs,'ROC-AUC']:.3f}",
    }, {
        "항목": "최종 모형", "값": best,
        "근거": f"최소최대 기준. 최악 유형 {piv.loc[best,'최악']:.3f}, 평균 {piv.loc[best,'평균']:.3f}",
    }, {
        "항목": "평가 규모", "값": f"{len(full):,} 조합",
        "근거": f"{len(FEATSPACES)}공간 x {len(ANOM_TYPES)}유형 x {len(MODELS)}모형, 각 LODO 5폴드",
    }])
    save_table(sel, "t37_benchmark_selection", index=False)
    print("\n[12] 완료")


if __name__ == "__main__":
    main()
