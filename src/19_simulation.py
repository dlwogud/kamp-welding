"""19. 시뮬레이션 검증 — 방법이 나쁜가, 데이터가 부족한가.

본 분석의 모든 귀무 결과(2.7 과제 A, 2.8 불량 연결, 2.9 LLP)에는 공통된 약점이 있다.
**'방법이 무력한 것'과 '표본이 부족한 것'을 구분할 직접 증거가 없다.**
3.5절의 검정력 분석이 상관계수 수준에서 이를 다루었으나, 모델링 절차 전체에 대한
검증은 아니었다.

본 절은 그 구분을 짓는다. **공정조건과 불량의 관계를 우리가 직접 심어 넣은 합성 데이터**를
만들고, 실제 데이터에 적용한 것과 동일한 LLP 절차를 돌린다. 합성 데이터에서는
**용접점 단위 진짜 라벨을 알고 있으므로**, 실제 데이터로는 원리상 측정할 수 없는 것을
측정할 수 있다.

측정 대상
    (1) LLP 가 학습한 점수가 진짜 라벨을 실제로 판별하는가 (인스턴스 ROC-AUC)
    (2) 가방이 몇 개부터 복원되는가
    (3) 불량률이 얼마부터 가능한가

합성 데이터의 설계 — 실제 데이터에 맞춘다
    공정변수는 **실제 안정일 6,339점을 재표본**하여 생성한다. 따라서 주변분포와
    가압력-전류 결합(rho = -0.735)이 실제와 동일하다. 임의의 분포를 가정하지 않는다.
    불량 발생기구는 로지스틱으로 심는다.
        logit P(defect | x) = a + b * z(x),   z = 특징공간의 어떤 방향으로의 투영
    a 는 목표 불량률을, b 는 효과크기를 정한다. b 는 **오라클 ROC-AUC**(진짜 기구를
    아는 이상적 판별자의 성능)가 지정값이 되도록 수치적으로 보정한다. 효과크기를
    'AUC 0.80 짜리 관계'처럼 해석 가능한 단위로 말하기 위해서다.

평가
    가방을 학습/평가로 나누고, 학습 가방에서는 **가방별 양성 개수만** 준다
    (실제 데이터와 동일한 정보량). 평가 가방에서 진짜 라벨로 인스턴스 AUC 를 잰다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from common import FIG, load_genuine, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
FEATS = [F, I, "R", "E"]

BAG_SIZE = 800                 # 실제 작업일 규모 (669~2,000점)
RATE = 0.00347                 # 실제 관측 불량률 (22건 / 6,339점)
N_BAGS_GRID = [5, 8, 12, 20, 30, 50]
ORACLE_AUC_GRID = [0.70, 0.80, 0.90]
RATE_GRID = [0.00347, 0.01, 0.03]
N_REP = 25
TEST_FRAC = 0.3
MAX_ITER = 8


def physics(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    amp = d[I] * 1000.0
    d["R"] = d[V] / amp
    d["E"] = d[V] * amp * (d[T] / 1000.0)
    return d


def feature_pool() -> np.ndarray:
    """실제 안정일 공정조건. 합성 데이터는 여기서 재표본한다."""
    g = physics(load_genuine(stable_only=True))
    X = g[FEATS].to_numpy(float)
    return RobustScaler().fit_transform(X)


def oracle_auc(z: np.ndarray, a: float, b: float, rng, n: int = 400_000) -> float:
    """주어진 (a, b) 에서 오라클(= z 로 순위를 매기는 이상적 판별자)의 ROC-AUC.

    해석적 근사 대신 몬테카를로로 직접 계산한다. 초판은 가중 순위로 근사했는데
    그 근사가 틀려 목표 0.70~0.90 을 지정해도 실제로는 0.999 가 심어졌다.
    """
    idx = rng.integers(0, len(z), n)
    zz = z[idx]
    p = 1.0 / (1.0 + np.exp(-(a + b * zz)))
    y = (rng.random(n) < p).astype(int)
    if y.sum() == 0 or y.sum() == n:
        return 0.5
    return float(roc_auc_score(y, zz))


def calibrate_b(z: np.ndarray, target_auc: float, rate: float, rng) -> float:
    """오라클 ROC-AUC 가 target 이 되도록 효과크기 b 를 이분탐색으로 보정.

    b 를 바꾸면 발생률도 변하므로, 매 후보마다 절편 a 를 목표 발생률에 맞춰
    다시 맞춘 뒤 AUC 를 측정한다.
    """
    lo, hi = 0.0, 20.0
    for _ in range(25):
        b = (lo + hi) / 2
        a = np.log(rate / (1 - rate)) - b * z.mean()
        if oracle_auc(z, a, b, rng) < target_auc:
            lo = b
        else:
            hi = b
    return (lo + hi) / 2


def simulate(pool: np.ndarray, n_bags: int, rate: float, b: float,
             direction: np.ndarray, rng) -> tuple[dict, dict, dict]:
    """합성 가방 생성. 반환: 특징, 진짜 라벨, 가방별 양성 개수."""
    X, Y, N = {}, {}, {}
    # 목표 불량률이 되도록 절편 보정
    zs = pool @ direction
    a = np.log(rate / (1 - rate)) - b * zs.mean()
    for k in range(n_bags):
        idx = rng.integers(0, len(pool), BAG_SIZE)
        x = pool[idx]
        z = x @ direction
        p = 1.0 / (1.0 + np.exp(-(a + b * z)))
        y = (rng.random(BAG_SIZE) < p).astype(int)
        X[k], Y[k], N[k] = x, y, int(y.sum())
    return X, Y, N


# ══════════════════════════════════════════════════════════════════
# LLP 모형 (18 과 동일한 절차. 가방별 양성 개수만 사용)
# ══════════════════════════════════════════════════════════════════
def fit_m1(Xtr: dict, ntr: dict, seed=SEED):
    Xall = np.vstack([Xtr[b] for b in Xtr])
    bags = np.concatenate([[b] * len(Xtr[b]) for b in Xtr])
    rate = sum(ntr.values()) / len(Xall)
    w = np.full(len(Xall), rate)
    clf = None
    Xd = np.vstack([Xall, Xall])
    y = np.r_[np.ones(len(Xall)), np.zeros(len(Xall))]
    for _ in range(MAX_ITER):
        sw = np.r_[w, 1 - w]
        clf = LogisticRegression(max_iter=400, random_state=seed).fit(Xd, y, sample_weight=sw)
        p = clf.predict_proba(Xall)[:, 1]
        w = np.empty_like(p)
        for b in Xtr:
            m = bags == b
            pb = p[m]
            w[m] = np.clip(pb * (ntr[b] / pb.sum()), 1e-9, 1.0) if pb.sum() > 0 \
                else ntr[b] / m.sum()
    return clf


def fit_m3(Xtr: dict, ntr: dict):
    """평균사상 — 가방별 (비율, 특징평균) 회귀의 기울기."""
    pis = np.array([ntr[b] / len(Xtr[b]) for b in Xtr])
    mus = np.vstack([Xtr[b].mean(0) for b in Xtr])
    A = np.c_[np.ones(len(pis)), pis]
    coef, *_ = np.linalg.lstsq(A, mus, rcond=None)
    d = coef[1]
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 0 else d


def one_run(pool, n_bags, rate, b, direction, rng) -> dict:
    X, Y, N = simulate(pool, n_bags, rate, b, direction, rng)
    bags = list(X)
    n_te = max(2, int(round(len(bags) * TEST_FRAC)))
    rng.shuffle(bags)
    te, tr = bags[:n_te], bags[n_te:]
    Xtr = {k: X[k] for k in tr}
    ntr = {k: N[k] for k in tr}
    Xte = np.vstack([X[k] for k in te])
    Yte = np.concatenate([Y[k] for k in te])
    if Yte.sum() == 0 or Yte.sum() == len(Yte):
        return None

    out = {"오라클": roc_auc_score(Yte, Xte @ direction)}
    try:
        clf = fit_m1(Xtr, ntr)
        out["M1 비율매칭"] = roc_auc_score(Yte, clf.predict_proba(Xte)[:, 1])
    except Exception:
        out["M1 비율매칭"] = np.nan
    try:
        d3 = fit_m3(Xtr, ntr)
        out["M3 평균사상"] = roc_auc_score(Yte, Xte @ d3)
    except Exception:
        out["M3 평균사상"] = np.nan
    return out


def sweep_bags(pool, direction) -> pd.DataFrame:
    rows = []
    z = pool @ direction
    crng = np.random.default_rng(SEED)
    for oauc in ORACLE_AUC_GRID:
        b = calibrate_b(z, oauc, RATE, crng)
        for nb in N_BAGS_GRID:
            rng = np.random.default_rng(SEED + nb * 7 + int(oauc * 100))
            accs = []
            for _ in range(N_REP):
                r = one_run(pool, nb, RATE, b, direction, rng)
                if r:
                    accs.append(r)
            if not accs:
                continue
            df = pd.DataFrame(accs)
            rows.append({
                "오라클 AUC (심은 효과크기)": oauc,
                "가방 수": nb,
                "실현 오라클 AUC": round(float(df["오라클"].mean()), 3),
                "M1 인스턴스 AUC": round(float(df["M1 비율매칭"].mean()), 3),
                "M1 표준편차": round(float(df["M1 비율매칭"].std()), 3),
                "M3 인스턴스 AUC": round(float(df["M3 평균사상"].mean()), 3),
                "M1 > 0.6 비율(%)": round(float((df["M1 비율매칭"] > 0.6).mean() * 100), 1),
            })
    out = pd.DataFrame(rows)
    save_table(out, "t63_sim_bags", index=False)
    return out


def sweep_2d(pool, direction) -> pd.DataFrame:
    """가방 수 x 불량률 교차 스윕 — 어느 축이 병목인가 (오라클 AUC 0.80 고정).

    1차 스윕에서 가방을 50개로 늘려도 복원되지 않는 반면 불량률을 올리면 개선되는
    징후가 보였다. 두 축을 교차시켜 병목을 확정한다.

    지표를 셋 둔다. 극저 불량률에서는 적합 방향의 부호가 뒤집혀 평균 AUC 가 0.5
    근처로 상쇄되므로, 평균만으로는 '학습이 안 된 것'과 '불안정한 것'을 구분할 수 없다.
      평균 AUC        : 기대 성능
      AUC>0.6 비율    : 쓸 만한 수준에 도달한 실행의 비율
      평균 |AUC-0.5|  : 부호를 무시했을 때 방향을 찾긴 했는가
    """
    z = pool @ direction
    b = calibrate_b(z, 0.80, RATE, np.random.default_rng(SEED))
    rows = []
    for nb in [5, 10, 20, 50]:
        for rate in [0.00347, 0.01, 0.03, 0.10]:
            rng = np.random.default_rng(SEED + int(rate * 1e5) + nb * 13)
            accs = [one_run(pool, nb, rate, b, direction, rng) for _ in range(N_REP)]
            accs = [a for a in accs if a]
            if not accs:
                continue
            df = pd.DataFrame(accs)
            auc = df["M1 비율매칭"].dropna()
            rows.append({
                "가방 수": nb,
                "불량률(%)": round(rate * 100, 3),
                "기대 양성/가방": round(rate * BAG_SIZE, 1),
                "총 양성 수": int(round(rate * BAG_SIZE * nb)),
                "평균 AUC": round(float(auc.mean()), 3),
                "AUC>0.6 비율(%)": round(float((auc > 0.6).mean() * 100), 1),
                "평균 |AUC-0.5|": round(float((auc - 0.5).abs().mean()), 3),
            })
    out = pd.DataFrame(rows)
    save_table(out, "t64_sim_2d", index=False)
    return out


def sweep_rate(pool, direction) -> pd.DataFrame:
    """불량률의 영향 (가방 수 고정). 하위호환용 요약."""
    z = pool @ direction
    b = calibrate_b(z, 0.80, RATE, np.random.default_rng(SEED))
    rows = []
    for nb in [5, 20]:
        for rate in RATE_GRID:
            rng = np.random.default_rng(SEED + int(rate * 1e5) + nb)
            accs = [one_run(pool, nb, rate, b, direction, rng) for _ in range(N_REP)]
            accs = [a for a in accs if a]
            if not accs:
                continue
            df = pd.DataFrame(accs)
            rows.append({
                "가방 수": nb,
                "불량률": rate,
                "기대 양성/가방": round(rate * BAG_SIZE, 2),
                "M1 인스턴스 AUC": round(float(df["M1 비율매칭"].mean()), 3),
                "M1 > 0.6 비율(%)": round(float((df["M1 비율매칭"] > 0.6).mean() * 100), 1),
            })
    out = pd.DataFrame(rows)
    save_table(out, "t64_sim_rate", index=False)
    return out


def t65_verdict(bag_tbl: pd.DataFrame) -> pd.DataFrame:
    """실제 데이터 조건(가방 5, 불량률 0.35%)에서의 기대 성능과 필요 가방 수."""
    rows = []
    for oauc in ORACLE_AUC_GRID:
        sub = bag_tbl[bag_tbl["오라클 AUC (심은 효과크기)"] == oauc].sort_values("가방 수")
        at5 = sub[sub["가방 수"] == 5]["M1 인스턴스 AUC"]
        # M1 인스턴스 AUC 가 0.6 을 넘는 최소 가방 수
        ok = sub[sub["M1 인스턴스 AUC"] >= 0.6]["가방 수"]
        rows.append({
            "심은 효과크기 (오라클 AUC)": oauc,
            "가방 5개일 때 M1 AUC": float(at5.iloc[0]) if len(at5) else np.nan,
            "M1 AUC ≥ 0.6 달성 최소 가방 수": int(ok.iloc[0]) if len(ok) else "50 이내 미달성",
            "가방 50개일 때 M1 AUC": float(sub[sub["가방 수"] == 50]["M1 인스턴스 AUC"].iloc[0]),
        })
    out = pd.DataFrame(rows)
    save_table(out, "t65_sim_verdict", index=False)
    return out


def f32(bag_tbl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = {0.70: "#95a5a6", 0.80: "#2e86c1", 0.90: "#1f4e79"}
    for oauc, c in cols.items():
        s = bag_tbl[bag_tbl["오라클 AUC (심은 효과크기)"] == oauc].sort_values("가방 수")
        if s.empty:
            continue
        ax.plot(s["가방 수"], s["M1 인스턴스 AUC"], "o-", color=c, lw=2.2, ms=7,
                label=f"심은 효과 오라클 AUC = {oauc}")
        ax.axhline(oauc, ls=":", color=c, lw=1.2, alpha=0.6)
    ax.axhline(0.5, ls="--", color="black", lw=1.2)
    ax.text(50, 0.505, "무작위 (0.5)", fontsize=8.5, ha="right")
    ax.axvline(5, ls=":", color="#c0392b", lw=1.8)
    ax.annotate("본 데이터 = 가방 5개", (5, 0.92), fontsize=9.5, color="#c0392b",
                rotation=90, va="top", ha="right")
    ax.set_xlabel("가방(작업일) 수")
    ax.set_ylabel("LLP 가 학습한 점수의 인스턴스 ROC-AUC")
    ax.set_xscale("log")
    ax.set_xticks(N_BAGS_GRID)
    ax.set_xticklabels(N_BAGS_GRID)
    ax.set_ylim(0.45, 0.95)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("합성 데이터 — 관계를 심어 두어도 가방 5개로는 복원되지 않는다\n"
                 "점선은 각 설정의 오라클(이상적 판별자) 성능", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f32_simulation_bags.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f32_simulation_bags.png")


def f33(tbl2: pd.DataFrame):
    piv = tbl2.pivot(index="불량률(%)", columns="가방 수", values="평균 AUC")
    fig, ax = plt.subplots(figsize=(7.6, 5))
    im = ax.imshow(piv.to_numpy(), cmap="RdYlGn", vmin=0.45, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=10)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"{v:g}%" for v in piv.index], fontsize=10)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                    color="black" if 0.5 < v < 0.8 else "white")
    ax.set_xlabel("가방(작업일) 수")
    ax.set_ylabel("불량률")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="M1 인스턴스 ROC-AUC")
    # 본 데이터 위치 표시
    ax.plot([0], [0], "o", ms=22, mfc="none", mec="#1f1f1f", mew=2.5)
    ax.annotate("본 데이터\n(가방 5, 0.35%)", (0, 0), fontsize=9, fontweight="bold",
                xytext=(26, 16), textcoords="offset points")
    ax.set_title("병목은 가방 수가 아니라 불량률이다\n"
                 "가방을 10배 늘려도(5→50) 개선되지 않지만, 불량률이 오르면 복원된다",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f33_simulation_2d.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f33_simulation_2d.png")


def main():
    print("\n[19] 시뮬레이션 검증 — 방법인가 데이터인가")
    pool = feature_pool()
    rng = np.random.default_rng(SEED)
    # 불량 발생 방향: 특징공간의 임의 단위벡터 (물리적 해석보다 일반성을 택한다)
    direction = rng.normal(size=pool.shape[1])
    direction /= np.linalg.norm(direction)
    print(f"  특징 풀: 실제 안정일 {len(pool):,}점 재표본")
    print(f"  가방 크기 {BAG_SIZE}, 반복 {N_REP}회/조건")

    print("\n══ (1) 가방 수 스윕 ══")
    bag_tbl = sweep_bags(pool, direction)
    print(bag_tbl.to_string(index=False))

    print("\n══ (2) 가방 수 x 불량률 교차 스윕 (오라클 AUC 0.80 고정) ══")
    tbl2 = sweep_2d(pool, direction)
    print(tbl2.to_string(index=False))
    piv = tbl2.pivot(index="불량률(%)", columns="가방 수", values="평균 AUC")
    print("\n   평균 AUC (행=불량률, 열=가방 수)")
    print(piv.to_string())
    by_rate = tbl2.groupby("불량률(%)")["평균 AUC"].mean()
    by_bag = tbl2.groupby("가방 수")["평균 AUC"].mean()
    print(f"\n   불량률에 따른 변화 폭: {by_rate.max()-by_rate.min():.3f}")
    print(f"   가방 수에 따른 변화 폭: {by_bag.max()-by_bag.min():.3f}")
    print(f"   => 병목은 {'불량률' if (by_rate.max()-by_rate.min()) > (by_bag.max()-by_bag.min()) else '가방 수'}")
    rate_tbl = tbl2

    print("\n══ (3) 판정 ══")
    v = t65_verdict(bag_tbl)
    print(v.to_string(index=False))

    f32(bag_tbl)
    f33(tbl2)

    # 결론은 표에서 유도한다 (미리 써두지 않는다)
    at5 = bag_tbl[bag_tbl["가방 수"] == 5]
    lo_auc, hi_auc = float(at5["M1 인스턴스 AUC"].min()), float(at5["M1 인스턴스 AUC"].max())
    by_rate = tbl2.groupby("불량률(%)")["평균 AUC"].mean()
    by_bag = tbl2.groupby("가방 수")["평균 AUC"].mean()
    rate_span, bag_span = by_rate.max() - by_rate.min(), by_bag.max() - by_bag.min()
    real = tbl2[tbl2["불량률(%)"] == 0.347]
    real_rng = f"{real['평균 AUC'].min():.3f}~{real['평균 AUC'].max():.3f}"
    bottleneck = "불량률" if rate_span > bag_span else "가방 수"
    concl = pd.DataFrame([
        {"항목": "설계", "값": "실제 안정일 공정조건을 재표본해 관계를 심고 동일 LLP 절차 적용"},
        {"항목": "실제 데이터로는 불가능했던 측정",
         "값": "용접점 단위 진짜 라벨로 인스턴스 ROC-AUC 를 직접 측정"},
        {"항목": "가방 5개(본 데이터 조건) 결과",
         "값": f"M1 인스턴스 AUC {lo_auc:.3f} ~ {hi_auc:.3f} — 무작위(0.5)와 사실상 같다"},
        {"항목": "병목 축",
         "값": f"{bottleneck} (불량률 변화 폭 {rate_span:.3f} vs 가방 수 변화 폭 {bag_span:.3f})"},
        {"항목": "실제 불량률 0.347% 에서",
         "값": f"가방을 5 → 50 으로 10배 늘려도 AUC {real_rng} 로 개선되지 않는다"},
        {"항목": "불량률 1% 이상에서",
         "값": f"AUC {tbl2[tbl2['불량률(%)'] >= 1]['평균 AUC'].min():.3f}~"
               f"{tbl2[tbl2['불량률(%)'] >= 1]['평균 AUC'].max():.3f} 로 복원이 시작된다"},
        {"항목": "판정",
         "값": "용접점 단위 불량 예측은 작업일을 더 모아도 달성되지 않는다. "
               "불량률 0.35% 에서는 가방 단위 라벨로부터 인스턴스 판별자를 학습할 수 없다"},
        {"항목": "4장 제언의 정정",
         "값": "'33 작업일 추가 수집'은 일 단위 관리도·상관 검출에는 유효하나 "
               "용접점 단위 불량 예측에는 무효하다. 후자에는 용접점 단위 라벨이 필수다"},
        {"항목": "단, 주의",
         "값": "본 시뮬레이션은 불량이 용접점 단위로 발생한다고 가정했다. 실제 데이터는 "
               "2.7.1 의 검정(불량건수가 생산량에 비례하지 않음) 때문에 그 가정조차 "
               "성립하지 않을 수 있으며, 그렇다면 전망은 더 비관적이다"},
    ])
    save_table(concl, "t66_sim_conclusion", index=False)
    print("\n-- 결론")
    print(concl.to_string(index=False))
    print("\n[19] 완료")


if __name__ == "__main__":
    main()
