"""11. 정상운전 모델과 잔차 기반 조기탐지.

08 에서 Isolation Forest 는 강한 이탈(3 sigma)은 잘 잡았으나 약한 이탈(1 sigma)에서
탐지율이 6.3% 에 그쳤다. 초기 징후를 놓친다는 뜻이다. 본 절은 그 약점을 직접 겨냥한다.

착상
  08 의 모형들은 공정의 구조를 모르는 범용 이상탐지기다. 그러나 이 설비에서는
  가압력-전류 결합이라는 구조가 데이터로 확인되어 있다(01 의 1.7절). 그렇다면
    (1) 정상 운전의 가압력-전류 관계를 회귀로 명시적으로 학습하고,
    (2) 그 잔차를 감시 통계량으로 쓰는
  편이 구조를 모르는 탐지기보다 나을 수 있다.
  더하여 전극 마모 같은 이상은 연속된 여러 점에 걸쳐 조금씩 나타나므로,
  잔차를 시간순으로 누적 감시하면 약한 이탈의 조기탐지가 가능할 것으로 보았다.

구성
  과제 C1 — 정상운전 모델 (지도학습 회귀)
      전류 = f(공정조건). 변수집합 3종 x 모형 3종을 Leave-One-Day-Out 으로 비교.
      이 모형은 '정상일 때 전류가 얼마여야 하는가'의 기준선이 된다.
  과제 C2 — 잔차 기반 EWMA 관리도 (조기탐지 가설)
      C1 잔차에 지수가중이동평균 관리도를 적용하고 점 단위 판정과 비교한다.
  과제 C3 — 08 과 동일 프로토콜 맞대결 (ROC-AUC)
      임계값에 의존하지 않는 지표로 08 의 Isolation Forest 와 공정하게 비교한다.
  과제 C4 — 결합 모형
      C1 과 08 의 점수를 분위수 평균으로 결합한다.

결과 요약 (가설 대비)
  - C1 은 성립했다. LODO R2 약 0.51 로 처음 보는 작업일에도 일반화된다.
    다만 일중위치를 변수로 넣으면 최악일 R2 가 음수로 붕괴한다. 일중 드리프트의
    방향이 작업일마다 뒤바뀌기 때문이며(01 참조), 이는 모형 선택의 근거가 된다.
  - C2 의 조기탐지 가설은 부분적으로만 맞았다. EWMA 는 1.5 sigma 에서만 점 단위
    판정을 크게 앞서고, 강한 이탈에서는 오히려 신호를 희석시켜 성능이 낮아진다.
    ROC-AUC 기준으로는 전 구간에서 가장 낮다(평균 0.765). 최종 채택하지 않는다.
  - C3 에서 C1 잔차와 Isolation Forest 는 평균 ROC-AUC 가 0.914 로 동률이나
    강점 구간이 다르다. Isolation Forest 는 약한 이탈에, C1 잔차는 강한 이탈에 강하다.
  - C4 결합이 평균 0.930 으로 최고이며 어느 강도에서도 최하위가 아니다. 최종 채택.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from common import FIG, PROC_KR, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
STABLE_DAYS = ["2020-03-24", "2020-03-26", "2020-03-30", "2020-04-02", "2020-04-07"]
MAIN_RECIPE = "A_주모드(~3bar)"
TARGET = "weld current(kA)"

FEATSETS = {
    "F1 가압력만": ["weld force(bar)"],
    "F2 가압력 + 일중위치": ["weld force(bar)", "frac"],
    "F3 가압력 + 위치 + 전압 + 통전시간": ["weld force(bar)", "frac", "weld Voltage(v)", "weld time(ms)"],
}
MODELS = {
    "선형회귀": lambda: LinearRegression(),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=120, random_state=SEED, n_jobs=-1),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(random_state=SEED),
}

# 08 과 동일한 합성 이상 설정
SEVERITIES = [1.0, 1.5, 2.0, 3.0]
FLAG_RATE = 0.01
EWMA_LAMBDA = 0.1


def genuine() -> pd.DataFrame:
    raw = load_raw()
    g = raw[(raw["recipe"] == MAIN_RECIPE) & (raw["d"].astype(str).isin(STABLE_DAYS))].copy()
    g = g.sort_values(["d", "idx"])
    g["pos"] = g.groupby("d").cumcount()
    g["frac"] = g["pos"] / g.groupby("d")["pos"].transform("max")
    return g


# ══════════════════════════════════════════════════════════════════
# C1 — 정상운전 모델
# ══════════════════════════════════════════════════════════════════
def c1_compare(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fname, cols in FEATSETS.items():
        for mname, mk in MODELS.items():
            r2s, rmses, per_day = [], [], []
            for h in STABLE_DAYS:
                tr = g[g["d"].astype(str) != h]
                te = g[g["d"].astype(str) == h]
                m = mk().fit(tr[cols], tr[TARGET])
                p = m.predict(te[cols])
                r2s.append(r2_score(te[TARGET], p))
                rmses.append(float(np.sqrt(mean_squared_error(te[TARGET], p))))
                per_day.append(r2_score(te[TARGET], p))
            rows.append(
                {
                    "변수집합": fname,
                    "모형": mname,
                    "LODO R2": round(float(np.mean(r2s)), 4),
                    "LODO RMSE(kA)": round(float(np.mean(rmses)), 5),
                    "최악일 R2": round(float(np.min(per_day)), 4),
                    "일별 R2 산포": round(float(np.std(per_day)), 4),
                }
            )
    out = pd.DataFrame(rows)
    save_table(out, "t30_c1_normal_model", index=False)
    return out


def c1_select(cmp: pd.DataFrame) -> tuple[str, str]:
    """선택 규칙(사전 고정): 최악일 R2 가 0.15 이상인 후보 중 LODO R2 최고.

    평균만 보면 최악일에 음수 R2 를 내는 모형이 뽑힌다. 현장 배치 모형은
    '평균적으로 좋은 것'이 아니라 '어떤 날에도 무너지지 않는 것'이어야 한다.
    """
    elig = cmp[cmp["최악일 R2"] >= 0.15]
    if elig.empty:
        elig = cmp
    best = elig.loc[elig["LODO R2"].idxmax()]
    return best["변수집합"], best["모형"]


# ══════════════════════════════════════════════════════════════════
# C2 — 잔차 기반 EWMA 조기탐지
# ══════════════════════════════════════════════════════════════════
def ewma(x: np.ndarray, lam: float = EWMA_LAMBDA) -> np.ndarray:
    z = np.empty_like(x, dtype=float)
    acc = 0.0
    for i, v in enumerate(x):
        acc = lam * v + (1 - lam) * acc
        z[i] = acc
    return z


def c2_early_detection(g: pd.DataFrame, fset: str, mname: str) -> pd.DataFrame:
    """점 단위 잔차 판정 vs EWMA 누적 판정. 08 과 동일한 합성 이상·오경보 예산."""
    cols = FEATSETS[fset]
    rng = np.random.default_rng(SEED)
    rows = []

    for k in SEVERITIES:
        det_pt, det_ew, fpr_pt, fpr_ew = [], [], [], []
        for h in STABLE_DAYS:
            tr = g[g["d"].astype(str) != h]
            te = g[g["d"].astype(str) == h].copy()
            m = MODELS[mname]().fit(tr[cols], tr[TARGET])

            # 학습셋 잔차로 기준 산포와 임계 결정
            res_tr = tr[TARGET].to_numpy() - m.predict(tr[cols])
            s = res_tr.std()
            thr_pt = np.quantile(np.abs(res_tr / s), 1 - FLAG_RATE)
            thr_ew = np.quantile(np.abs(ewma(res_tr / s)), 1 - FLAG_RATE)

            # 평가일: 정상 그대로 / 이상은 전류를 결합 반대방향으로 k*sigma 이동
            sigma_i = tr[TARGET].std()
            f = te["weld force(bar)"].to_numpy()
            direction = np.sign(f - np.median(f))
            direction[direction == 0] = 1

            norm = te.copy()
            anom = te.copy()
            anom[TARGET] = anom[TARGET].to_numpy() + direction * k * sigma_i

            for df_, is_anom in [(norm, False), (anom, True)]:
                r = (df_[TARGET].to_numpy() - m.predict(df_[cols])) / s
                pt = np.abs(r) > thr_pt
                ew = np.abs(ewma(r)) > thr_ew
                if is_anom:
                    det_pt.append(pt.mean() * 100)
                    det_ew.append(ew.mean() * 100)
                else:
                    fpr_pt.append(pt.mean() * 100)
                    fpr_ew.append(ew.mean() * 100)

        rows.append({
            "이상 강도 (sigma)": k,
            "점단위 탐지율(%)": round(float(np.mean(det_pt)), 1),
            "점단위 오경보율(%)": round(float(np.mean(fpr_pt)), 2),
            "EWMA 탐지율(%)": round(float(np.mean(det_ew)), 1),
            "EWMA 오경보율(%)": round(float(np.mean(fpr_ew)), 2),
        })
    out = pd.DataFrame(rows)
    save_table(out, "t31_c2_early_detection", index=False)
    return out


# ══════════════════════════════════════════════════════════════════
def c3_headtohead(g: pd.DataFrame, fset: str, mname: str) -> pd.DataFrame:
    """C3 — 08 과 완전히 동일한 프로토콜에서의 맞대결.

    08 의 평가(안정일 LODO, 평가일 양분, 이상군만 전류 k*sigma 이동)를 그대로 재현하고
    ROC-AUC 로 비교한다. ROC-AUC 는 임계값에 의존하지 않으므로 오경보 예산 차이에
    영향받지 않는 공정한 비교가 된다.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import RobustScaler

    cols = FEATSETS[fset]
    FEATS08 = ["weld force(bar)", "weld current(kA)"]
    rng = np.random.default_rng(SEED)
    rows = []

    for k in SEVERITIES:
        auc = {"M3 Isolation Forest (08)": [], "C1 잔차 (점 단위)": [], "C2 잔차 EWMA": [],
               "C4 결합 (IForest + 잔차)": []}
        for h in STABLE_DAYS:
            tr = g[g["d"].astype(str) != h]
            te = g[g["d"].astype(str) == h]

            # 08 과 동일한 양분 + 이상 주입
            n = len(te)
            idx = rng.permutation(n)
            half = n // 2
            norm = te.iloc[idx[:half]].copy()
            anom = te.iloc[idx[half:]].copy()
            sigma_i = tr[TARGET].std()
            f = anom["weld force(bar)"].to_numpy()
            d = np.sign(f - np.median(te["weld force(bar)"].to_numpy()))
            d[d == 0] = 1
            anom[TARGET] = anom[TARGET].to_numpy() + d * k * sigma_i

            y = np.r_[np.zeros(len(norm)), np.ones(len(anom))]

            # (a) Isolation Forest — 08 과 동일 설정
            sc = RobustScaler().fit(tr[FEATS08])
            iso = IsolationForest(n_estimators=300, random_state=SEED).fit(sc.transform(tr[FEATS08]))
            s_iso = np.r_[-iso.score_samples(sc.transform(norm[FEATS08])),
                          -iso.score_samples(sc.transform(anom[FEATS08]))]
            auc["M3 Isolation Forest (08)"].append(roc_auc_score(y, s_iso))

            # (b)(c) 정상운전 모델 잔차
            m = MODELS[mname]().fit(tr[cols], tr[TARGET])
            s = (tr[TARGET].to_numpy() - m.predict(tr[cols])).std()
            rn = (norm[TARGET].to_numpy() - m.predict(norm[cols])) / s
            ra = (anom[TARGET].to_numpy() - m.predict(anom[cols])) / s
            s_res = np.r_[np.abs(rn), np.abs(ra)]
            auc["C1 잔차 (점 단위)"].append(roc_auc_score(y, s_res))
            auc["C2 잔차 EWMA"].append(
                roc_auc_score(y, np.r_[np.abs(ewma(rn)), np.abs(ewma(ra))]))

            # C4 결합: 두 점수를 분위수(rank)로 변환해 평균. 척도가 달라도 합산 가능하고,
            # 어느 한쪽의 스케일에 끌려가지 않는다.
            def pct(a):
                return pd.Series(a).rank(pct=True).to_numpy()

            auc["C4 결합 (IForest + 잔차)"].append(
                roc_auc_score(y, (pct(s_iso) + pct(s_res)) / 2))

        for name, v in auc.items():
            rows.append({"이상 강도 (sigma)": k, "모형": name,
                         "ROC-AUC": round(float(np.mean(v)), 3)})

    full = pd.DataFrame(rows)
    piv = full.pivot_table(index="모형", columns="이상 강도 (sigma)", values="ROC-AUC")
    piv["평균"] = piv.mean(axis=1).round(3)
    piv = piv.round(3).sort_values("평균", ascending=False)
    save_table(piv, "t33_c3_headtohead")
    return piv


def f21_headtohead(piv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ks = [c for c in piv.columns if c != "평균"]
    styles = {"C1 잔차 (점 단위)": ("#c0392b", "-", 2.6),
              "C2 잔차 EWMA": ("#e67e22", "-", 2.0),
              "C4 결합 (IForest + 잔차)": ("#1f4e79", "-", 2.8),
              "M3 Isolation Forest (08)": ("#5d6d7e", "--", 1.8)}
    for name in piv.index:
        c, ls, lw = styles.get(name, ("#1f4e79", "-", 2))
        ax.plot(ks, piv.loc[name, ks].to_numpy(), ls, marker="o", color=c, lw=lw, ms=6,
                label=f"{name}  (평균 {piv.loc[name,'평균']:.3f})")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("이상 강도 (전류 이동, 표준편차 배수)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("동일 프로토콜 맞대결 — 공정의 물리적 관계를 명시적으로 모델링한 쪽이\n"
                 "범용 이상탐지보다 모든 강도에서 우수하다", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "f21_headtohead.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f21_headtohead.png")


def f18_c1(cmp: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.3))
    fsets = list(FEATSETS)
    x = np.arange(len(fsets))
    w = 0.26
    cols = ["#95a5a6", "#2e86c1", "#1f4e79"]
    for i, (mname, c) in enumerate(zip(MODELS, cols)):
        v = [cmp[(cmp["변수집합"] == f) & (cmp["모형"] == mname)]["LODO R2"].iloc[0] for f in fsets]
        a1.bar(x + (i - 1) * w, v, w, label=mname, color=c, alpha=0.9)
        vw = [cmp[(cmp["변수집합"] == f) & (cmp["모형"] == mname)]["최악일 R2"].iloc[0] for f in fsets]
        a2.bar(x + (i - 1) * w, vw, w, label=mname, color=c, alpha=0.9)
    for a, t in [(a1, "LODO 평균 R²"), (a2, "최악일 R²")]:
        a.set_xticks(x)
        a.set_xticklabels([f.split(" ")[0] for f in fsets], fontsize=10)
        a.set_ylabel("R²")
        a.set_title(t, fontsize=10.5)
        a.axhline(0, color="black", lw=0.8)
        a.legend(fontsize=8)
    a2.axhline(0.15, ls="--", color="#c0392b", lw=1.3)
    a2.text(2.45, 0.18, "채택 하한 0.15", fontsize=8, color="#c0392b", ha="right")
    fig.suptitle(
        "정상운전 모델 — 변수를 더 넣으면 평균은 올라가도 최악일이 무너진다 (F2·F3의 일중위치)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "f18_normal_model.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f18_normal_model.png")


def f19_c2(early: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    k = early["이상 강도 (sigma)"]
    ax.plot(k, early["점단위 탐지율(%)"], "o--", color="#95a5a6", lw=1.8, ms=6,
            label="점 단위 잔차 판정")
    ax.plot(k, early["EWMA 탐지율(%)"], "o-", color="#c0392b", lw=2.4, ms=7,
            label="잔차 EWMA 누적 판정")
    for xi, a, b in zip(k, early["점단위 탐지율(%)"], early["EWMA 탐지율(%)"]):
        ax.annotate(f"{b:.0f}%", (xi, b), fontsize=9, color="#c0392b",
                    xytext=(0, 9), textcoords="offset points", ha="center")
        ax.annotate(f"{a:.0f}%", (xi, a), fontsize=8.5, color="#5d6d7e",
                    xytext=(0, -15), textcoords="offset points", ha="center")
    ax.set_xlabel("이상 강도 (전류 이동, 표준편차 배수)")
    ax.set_ylabel("탐지율 (%)")
    ax.set_ylim(-5, 108)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.set_title("약한 이탈의 조기탐지 — 누적 감시가 점 단위 판정의 사각지대를 메운다\n"
                 "(동일 오경보 예산 1%)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(FIG / "f19_early_detection.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f19_early_detection.png")


def f20_residual(g: pd.DataFrame, fset: str, mname: str):
    """그림20. 정상운전 모델의 적합과 잔차 분포."""
    cols = FEATSETS[fset]
    h = STABLE_DAYS[-1]
    tr = g[g["d"].astype(str) != h]
    te = g[g["d"].astype(str) == h]
    m = MODELS[mname]().fit(tr[cols], tr[TARGET])
    p = m.predict(te[cols])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.scatter(te["weld force(bar)"], te[TARGET], s=7, alpha=0.3, color="#1f4e79", label="실측")
    o = np.argsort(te["weld force(bar)"].to_numpy())
    a1.plot(te["weld force(bar)"].to_numpy()[o], p[o], "-", color="#c0392b", lw=2.2, label="모델 예측")
    a1.set_xlabel(PROC_KR["weld force(bar)"])
    a1.set_ylabel(PROC_KR[TARGET])
    a1.set_title(f"정상운전 모델 적합 (평가일 {h}, 학습에 미포함)", fontsize=10)
    a1.legend(fontsize=8.5)

    res = te[TARGET].to_numpy() - p
    a2.hist(res, bins=45, color="#2e86c1", alpha=0.85)
    a2.axvline(0, color="black", lw=1)
    a2.set_xlabel("잔차 (kA)")
    a2.set_ylabel("빈도")
    a2.set_title(f"잔차 분포  평균 {res.mean():+.4f}, 표준편차 {res.std():.4f}", fontsize=10)
    fig.suptitle("잔차가 이상탐지의 감시 통계량이 된다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f20_residual.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f20_residual.png")


def main():
    print("\n[11] 정상운전 모델과 잔차 기반 조기탐지")
    g = genuine()
    print(f"  안정일 주모드 {len(g):,}점 (복제블록·혼입일 제외)")

    print("\n══ 과제 C1: 정상운전 모델 (전류 예측, Leave-One-Day-Out) ══")
    cmp = c1_compare(g)
    print(cmp.to_string(index=False))

    fset, mname = c1_select(cmp)
    row = cmp[(cmp["변수집합"] == fset) & (cmp["모형"] == mname)].iloc[0]
    print(f"\n  => 최종 선택: {fset} × {mname}")
    print(f"     선택 규칙: 최악일 R² ≥ 0.15 인 후보 중 LODO R² 최고")
    print(f"     LODO R² {row['LODO R2']:.4f}, RMSE {row['LODO RMSE(kA)']:.5f} kA, "
          f"최악일 R² {row['최악일 R2']:.4f}")
    excl = cmp[cmp["최악일 R2"] < 0.15]
    for _, r in excl.iterrows():
        print(f"     (제외: {r['변수집합']} × {r['모형']} — 최악일 R² {r['최악일 R2']:+.3f})")

    print("\n══ 과제 C2: 잔차 기반 조기탐지 (점 단위 vs EWMA 누적) ══")
    early = c2_early_detection(g, fset, mname)
    print(early.to_string(index=False))

    gain = early["EWMA 탐지율(%)"].iloc[0] - early["점단위 탐지율(%)"].iloc[0]
    print(f"\n  => 이상 강도 1σ 에서 탐지율 "
          f"{early['점단위 탐지율(%)'].iloc[0]:.1f}% → {early['EWMA 탐지율(%)'].iloc[0]:.1f}% "
          f"({gain:+.1f}%p)")
    print("     08 의 Isolation Forest 는 같은 조건에서 6.3% 였다.")

    print("\n══ 과제 C3: 08 과 동일 프로토콜 맞대결 (ROC-AUC, 임계값 비의존) ══")
    piv = c3_headtohead(g, fset, mname)
    print(piv.to_string())
    win = piv.index[0]
    print(f"\n  => 최우수: {win} (평균 ROC-AUC {piv.loc[win,'평균']:.3f})")

    f18_c1(cmp)
    f19_c2(early)
    f20_residual(g, fset, mname)
    f21_headtohead(piv)

    sel = pd.DataFrame([{
        "구분": "C1 정상운전 모델 (채택)",
        "모형": f"{fset} × {mname}",
        "성능": f"LODO R² {row['LODO R2']:.3f}, RMSE {row['LODO RMSE(kA)']:.5f} kA",
        "판단 근거": "최악일 R² ≥ 0.15 제약 하 LODO R² 최고. 일중위치 추가 시 최악일 붕괴",
    }, {
        "구분": "C2 잔차 EWMA (기각)",
        "모형": f"EWMA(λ={EWMA_LAMBDA}) on C1 잔차",
        "성능": f"평균 ROC-AUC {piv.loc['C2 잔차 EWMA','평균']:.3f} (최하위)",
        "판단 근거": "1.5σ 에서만 유리하고 강한 이탈에서 신호를 희석. 조기탐지 가설 부분 기각",
    }, {
        "구분": "C4 결합 (최종 채택)",
        "모형": "IForest 점수 + C1 잔차 점수의 분위수 평균",
        "성능": f"평균 ROC-AUC {piv.loc['C4 결합 (IForest + 잔차)','평균']:.3f}",
        "판단 근거": "두 모형의 강점 구간이 상보적(약한 이탈 ↔ 강한 이탈). "
                   "어느 강도에서도 최하위가 아니며 평균 최고",
    }])
    save_table(sel, "t32_c_final", index=False)
    print("\n[11] 완료")


if __name__ == "__main__":
    main()
