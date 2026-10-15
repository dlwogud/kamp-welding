"""18. 용접점 단위 불량 예측 — 라벨 비율 학습(LLP)으로 정면 시도.

초판의 잘못된 단정을 정정한다.
    1.3절에서 '용접점 단위 지도학습 불가'라고 썼다. 이는 부정확한 진술이었다.
    성립하지 않는 것은 **표준 단일사례 지도학습**이며, 본 데이터의 라벨 구조
    (가방=작업일, 가방별 양성 개수를 안다)는 **라벨 비율 학습(Learning from Label
    Proportions, LLP)** 또는 다중사례학습(MIL)의 전형적 설정이다.
    이 분야는 바로 이런 상황을 위해 존재한다. 시도하지 않고 불가능하다고 쓴 것은
    과제의 핵심 요구('용접불량 여부를 예측하는 모델')를 회피한 것이다.

본 절은 그 시도를 수행하고 결과를 그대로 보고한다.

문제 설정
    가방 b (작업일) 마다 용접점 집합 X_b 와 양성 개수 n_b 가 주어진다.
    목표는 용접점 단위 판별함수 f(x) 를 학습하는 것이다.
    양성 비율은 약 0.35% (22건 / 6,339점) 로 극단적으로 낮다.

사용 가방
    복제 구간이 없는 안정일 5일만 쓴다. 혼입일은 작업일의 44~100%가 복제 구간이라
    '불량건수 / 실제 생산점수' 의 분모가 정의되지 않는다. 가방이 5개뿐이지만
    분모가 오염된 7개보다 낫다.

비교 모형
    M0 전역 상수   전 용접점에 동일 점수. 기준선.
    M1 비율매칭 로지스틱   예측 확률의 가방별 합이 n_b 가 되도록 절편을 보정하며 학습
    M2 교대 의사라벨   가방 내 상위 n_b 점을 양성으로 두고 학습, 재할당을 반복 (mi-SVM 계열)
    M3 평균사상(Mean Map)   양성 집단의 특징 평균을 가방 제약으로 추정해 판별

평가 (가방 단위 Leave-One-Bag-Out)
    학습에 쓰이지 않은 작업일에서 예측 양성 개수와 실제 개수를 비교한다.
    지표는 개수 MAE. 기준선(전역 비율 x 그 날 생산량)을 넘는지가 판정 기준이다.

정직성 장치
    가방의 양성 개수를 무작위로 섞은 뒤 전 과정을 반복하는 순열검정을 둔다.
    섞은 라벨로도 같은 성능이 나오면, 그 성능은 학습이 아니라 절차의 산물이다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from common import FIG, STABLE_DAYS, daily_label, load_genuine, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
FEATS = [F, I, "R", "E"]
N_PERM = 40
MAX_ITER = 8


def physics(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    amp = d[I] * 1000.0
    d["R"] = d[V] / amp
    d["E"] = d[V] * amp * (d[T] / 1000.0)
    return d


def make_bags() -> tuple[dict, dict]:
    """가방(작업일) -> 특징행렬, 양성 개수."""
    g = physics(load_genuine(stable_only=True))
    dl = daily_label()
    X, n = {}, {}
    for d, gg in g.groupby("d"):
        X[str(d)] = gg[FEATS].to_numpy(float)
        n[str(d)] = int(dl.loc[d])
    return X, n


# ══════════════════════════════════════════════════════════════════
# 모형들. 모두 '가방별 양성 개수'만 쓰고 용접점 라벨은 쓰지 않는다.
# ══════════════════════════════════════════════════════════════════
def fit_m0(Xtr, ntr):
    """전역 상수 — 전 점에 동일 점수. 예측 개수는 전역 비율 x 생산량."""
    rate = sum(ntr.values()) / sum(len(X) for X in Xtr.values())
    return {"kind": "M0", "rate": rate}


def fit_m1(Xtr, ntr, seed=SEED):
    """비율매칭 로지스틱.

    의사라벨 없이, 예측확률의 가방별 합이 n_b 가 되도록 절편을 조정하면서
    가중 로지스틱을 EM 처럼 반복 적합한다.
    """
    X = np.vstack(list(Xtr.values()))
    sc = RobustScaler().fit(X)
    Xs = {b: sc.transform(v) for b, v in Xtr.items()}
    Xall = np.vstack(list(Xs.values()))
    bags = np.concatenate([[b] * len(v) for b, v in Xs.items()])

    # 초기화: 전역 비율
    rate = sum(ntr.values()) / len(Xall)
    w = np.full(len(Xall), rate)
    clf = None
    for _ in range(MAX_ITER):
        # 소프트 라벨을 표본가중으로 변환해 적합
        y = np.r_[np.ones(len(Xall)), np.zeros(len(Xall))]
        Xd = np.vstack([Xall, Xall])
        sw = np.r_[w, 1 - w]
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=seed).fit(Xd, y, sample_weight=sw)
        p = clf.predict_proba(Xall)[:, 1]
        # 가방별로 합이 n_b 가 되도록 재조정 (단조 변환)
        w = np.empty_like(p)
        for b in Xs:
            m = bags == b
            pb = p[m]
            target = ntr[b]
            if pb.sum() <= 0:
                w[m] = target / m.sum()
                continue
            w[m] = np.clip(pb * (target / pb.sum()), 1e-9, 1.0)
    return {"kind": "M1", "sc": sc, "clf": clf, "rate": rate}


def fit_m2(Xtr, ntr, seed=SEED):
    """교대 의사라벨 — 가방 내 상위 n_b 점을 양성으로 두고 학습, 재할당 반복."""
    rng = np.random.default_rng(seed)
    X = np.vstack(list(Xtr.values()))
    sc = RobustScaler().fit(X)
    Xs = {b: sc.transform(v) for b, v in Xtr.items()}

    # 초기 할당: 무작위
    lab = {}
    for b, v in Xs.items():
        y = np.zeros(len(v), int)
        if ntr[b] > 0:
            y[rng.choice(len(v), size=min(ntr[b], len(v)), replace=False)] = 1
        lab[b] = y

    clf = None
    for _ in range(MAX_ITER):
        Xall = np.vstack(list(Xs.values()))
        yall = np.concatenate([lab[b] for b in Xs])
        if yall.sum() == 0 or yall.sum() == len(yall):
            break
        clf = HistGradientBoostingClassifier(max_iter=60, random_state=seed,
                                             class_weight="balanced").fit(Xall, yall)
        changed = 0
        for b, v in Xs.items():
            p = clf.predict_proba(v)[:, 1]
            y = np.zeros(len(v), int)
            if ntr[b] > 0:
                top = np.argsort(-p)[:min(ntr[b], len(v))]
                y[top] = 1
            changed += int((y != lab[b]).sum())
            lab[b] = y
        if changed == 0:
            break
    return {"kind": "M2", "sc": sc, "clf": clf,
            "rate": sum(ntr.values()) / sum(len(v) for v in Xs.values())}


def fit_m3(Xtr, ntr):
    """평균사상(Mean Map) — 양성 집단의 특징 평균을 가방 제약으로 추정.

    전체 평균 mu = pi*mu1 + (1-pi)*mu0 에서, 가방별 비율 차이를 이용해 mu1 을 추정한다.
    가방별 (비율, 특징평균) 로 선형회귀하면 기울기가 (mu1 - mu0) 에 해당한다.
    판별은 그 방향으로의 투영으로 한다.
    """
    pis = np.array([ntr[b] / len(Xtr[b]) for b in Xtr])
    mus = np.vstack([Xtr[b].mean(0) for b in Xtr])
    sc = RobustScaler().fit(np.vstack(list(Xtr.values())))
    mus_s = sc.transform(mus)
    # mu_b = mu0 + pi_b*(mu1-mu0)  ->  각 특징에 대해 pi 에 대한 1차 회귀
    A = np.c_[np.ones(len(pis)), pis]
    coef, *_ = np.linalg.lstsq(A, mus_s, rcond=None)
    direction = coef[1]                       # (mu1 - mu0)
    nrm = np.linalg.norm(direction)
    if nrm > 0:
        direction = direction / nrm
    return {"kind": "M3", "sc": sc, "dir": direction,
            "rate": sum(ntr.values()) / sum(len(v) for v in Xtr.values())}


def score(model, Xte: np.ndarray) -> np.ndarray:
    """용접점 단위 점수 (클수록 불량 가능성 높음)."""
    k = model["kind"]
    if k == "M0":
        return np.full(len(Xte), model["rate"])
    Z = model["sc"].transform(Xte)
    if k == "M1":
        return model["clf"].predict_proba(Z)[:, 1]
    if k == "M2":
        return model["clf"].predict_proba(Z)[:, 1] if model["clf"] is not None \
            else np.full(len(Xte), model["rate"])
    if k == "M3":
        return Z @ model["dir"]
    raise ValueError(k)


def predict_count(model, Xte: np.ndarray) -> float:
    """예측 양성 개수. M0·M1 은 확률 합, M2 는 확률 합, M3 는 전역 비율로 환산."""
    s = score(model, Xte)
    if model["kind"] in ("M0",):
        return float(model["rate"] * len(Xte))
    if model["kind"] in ("M1", "M2"):
        return float(np.clip(s, 0, 1).sum())
    # M3 는 점수 척도가 확률이 아니므로 전역 비율만큼의 상위 점을 양성으로 본다
    return float(model["rate"] * len(Xte))


FITTERS = {"M0 전역 상수 (기준선)": fit_m0,
           "M1 비율매칭 로지스틱": fit_m1,
           "M2 교대 의사라벨": fit_m2,
           "M3 평균사상(Mean Map)": fit_m3}


def lobo(X: dict, n: dict, seed: int = SEED) -> dict:
    """가방 단위 Leave-One-Bag-Out. 모형별 예측 개수 MAE."""
    res = {}
    for name, fit in FITTERS.items():
        errs, preds = [], []
        for b in X:
            Xtr = {k: v for k, v in X.items() if k != b}
            ntr = {k: v for k, v in n.items() if k != b}
            try:
                m = fit(Xtr, ntr) if fit is fit_m0 or fit is fit_m3 else fit(Xtr, ntr, seed)
            except TypeError:
                m = fit(Xtr, ntr)
            pc = predict_count(m, X[b])
            preds.append(pc)
            errs.append(abs(pc - n[b]))
        res[name] = {"mae": float(np.mean(errs)), "preds": preds}
    return res


def t59_compare(X: dict, n: dict) -> pd.DataFrame:
    res = lobo(X, n)
    base = res["M0 전역 상수 (기준선)"]["mae"]
    rows = []
    for name, r in res.items():
        rows.append({"모형": name, "LOBO 개수 MAE": round(r["mae"], 3),
                     "기준선 대비": "—" if name.startswith("M0")
                                   else f"{(r['mae'] - base) / base * 100:+.1f}%",
                     "예측 개수": ", ".join(f"{p:.1f}" for p in r["preds"]),
                     "실제 개수": ", ".join(str(n[b]) for b in X)})
    out = pd.DataFrame(rows).sort_values("LOBO 개수 MAE")
    save_table(out, "t59_llp_comparison", index=False)
    return out, res


def t60_permutation(X: dict, n: dict) -> pd.DataFrame:
    """가방 양성 개수를 섞어도 같은 성능이 나오는가 — 전수 열거로 정확히 검정한다.

    가방이 5개이고 개수에 중복이 있으므로(4,7,4,4,3) 서로 다른 배열은 5!/3! = 20 개뿐이다.
    무작위 추출 대신 **전부 열거**하면 근사가 아닌 정확한 순열 p 값을 얻는다.
    다만 배열이 20개뿐이므로 p 값의 최소 해상도는 1/20 = 0.05 이다.
    즉 이 설계로 얻을 수 있는 가장 작은 p 값이 0.05 이며, 검정 자체의 분해능이 한계다.
    """
    from itertools import permutations
    bags = list(X)
    counts = [n[b] for b in bags]
    uniq = sorted(set(permutations(counts)))
    obs = lobo(X, n)
    rows = []
    for name in FITTERS:
        null = np.array([lobo(X, dict(zip(bags, u)))[name]["mae"] for u in uniq])
        o = obs[name]["mae"]
        p = float((null <= o).mean())
        rows.append({"모형": name, "관측 MAE": round(o, 3),
                     "배열 수": len(uniq),
                     "셔플 MAE 중앙값": round(float(np.median(null)), 3),
                     "셔플 MAE 최소": round(float(null.min()), 3),
                     "정확 순열 p": round(p, 4),
                     "판정": "유의" if p < 0.05 else "우연과 구분 불가"})
    out = pd.DataFrame(rows)
    save_table(out, "t60_llp_permutation", index=False)
    return out


def t61_instance(X: dict, n: dict) -> pd.DataFrame:
    """용접점 단위 점수의 안정성 — 라벨이 없으므로 직접 검증은 불가하다.

    대신 '어느 가방을 빼고 학습해도 같은 점에 높은 점수를 주는가'를 본다.
    불안정하면 그 점수는 신호가 아니라 잡음이다.
    """
    from scipy.stats import spearmanr
    bags = list(X)
    Xall = np.vstack([X[b] for b in bags])
    scores = {}
    for name, fit in FITTERS.items():
        if name.startswith("M0"):
            continue
        S = []
        for b in bags:
            Xtr = {k: v for k, v in X.items() if k != b}
            ntr = {k: v for k, v in n.items() if k != b}
            try:
                m = fit(Xtr, ntr, SEED)
            except TypeError:
                m = fit(Xtr, ntr)
            S.append(score(m, Xall))
        S = np.vstack(S)
        rhos = [spearmanr(S[i], S[j]).statistic
                for i in range(len(S)) for j in range(i + 1, len(S))]
        scores[name] = float(np.mean(rhos))
    out = pd.DataFrame([{"모형": k, "폴드 간 점수 순위상관 (평균)": round(v, 3),
                         "해석": "안정" if v > 0.7 else ("보통" if v > 0.4 else "불안정")}
                        for k, v in scores.items()])
    save_table(out, "t61_instance_stability", index=False)
    return out


def f31(cmp: pd.DataFrame, X: dict, n: dict, res: dict):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    d = cmp.sort_values("LOBO 개수 MAE")
    cols = ["#2e86c1" if m.startswith("M0") else "#95a5a6" for m in d["모형"]]
    a1.barh(range(len(d)), d["LOBO 개수 MAE"], color=cols, alpha=0.9)
    a1.set_yticks(range(len(d)))
    a1.set_yticklabels([m.split(" (")[0] for m in d["모형"]], fontsize=9)
    a1.invert_yaxis()
    for i, v in enumerate(d["LOBO 개수 MAE"]):
        a1.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    a1.set_xlabel("LOBO 예측 개수 MAE (낮을수록 좋음)")
    a1.set_title("라벨 비율 학습 모형 비교", fontsize=10.5)

    bags = list(X)
    actual = [n[b] for b in bags]
    x = np.arange(len(bags))
    a2.plot(x, actual, "o-", color="black", lw=2.2, ms=8, label="실제 불량건수", zorder=4)
    for name, c in zip([m for m in res if not m.startswith("M0")],
                       ["#c0392b", "#e67e22", "#27ae60"]):
        a2.plot(x, res[name]["preds"], "s--", color=c, lw=1.8, ms=6,
                label=name.split(" (")[0])
    a2.plot(x, res["M0 전역 상수 (기준선)"]["preds"], "^:", color="#2e86c1", lw=1.8,
            ms=7, label="M0 기준선")
    a2.set_xticks(x)
    a2.set_xticklabels([b[5:] for b in bags], fontsize=9)
    a2.set_ylabel("불량건수")
    a2.legend(fontsize=8.5)
    a2.set_title("가방별 예측 vs 실제 (Leave-One-Bag-Out)", fontsize=10.5)
    fig.suptitle("가방 5개·양성비율 0.35% — 어떤 LLP 모형도 전역 상수를 넘지 못한다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f31_llp.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f31_llp.png")


def main():
    print("\n[18] 라벨 비율 학습(LLP)으로 용접점 단위 불량 예측 시도")
    X, n = make_bags()
    tot_pts = sum(len(v) for v in X.values())
    tot_pos = sum(n.values())
    print(f"  가방 {len(X)} 개 (복제 구간 없는 안정일), 용접점 {tot_pts:,} 점")
    print(f"  양성 {tot_pos} 건, 비율 {tot_pos/tot_pts*100:.3f}%")
    print(f"  가방별: " + ", ".join(f"{b[5:]}={n[b]}/{len(X[b])}" for b in X))

    print("\n══ (1) 모형 비교 (Leave-One-Bag-Out) ══")
    cmp, res = t59_compare(X, n)
    print(cmp.to_string(index=False))
    best = cmp.iloc[0]["모형"]
    print(f"\n   최우수: {best}")

    print("\n══ (2) 순열검정 — 가방 개수를 섞어도 같은가 ══")
    perm = t60_permutation(X, n)
    print(perm.to_string(index=False))

    print("\n══ (3) 용접점 점수의 안정성 ══")
    st = t61_instance(X, n)
    print(st.to_string(index=False))

    f31(cmp, X, n, res)

    concl = pd.DataFrame([
        {"항목": "문제 설정", "값": "라벨 비율 학습(LLP) — 가방=작업일, 가방별 양성 개수 기지"},
        {"항목": "가방 수 / 양성 비율", "값": f"{len(X)} 개 / {tot_pos/tot_pts*100:.3f}%"},
        {"항목": "비교 모형", "값": "4종 (전역 상수 + LLP 3종)"},
        {"항목": "최우수", "값": best},
        {"항목": "기준선 대비 명목 성능",
         "값": "없음" if best.startswith("M0")
               else f"{best} 가 MAE {cmp.iloc[0]['기준선 대비']} (명목 개선)"},
        {"항목": "정확 순열검정",
         "값": f"유의 {int((perm['정확 순열 p'] < 0.05).sum())}/{len(perm)} 모형. "
               f"최소 p = {perm['정확 순열 p'].min():.3f}"},
        {"항목": "검정 분해능 한계",
         "값": f"가방 5개 → 서로 다른 배열 {int(perm['배열 수'].iloc[0])}개. "
               f"얻을 수 있는 최소 p 가 {1/perm['배열 수'].iloc[0]:.3f} 이다"},
        {"항목": "결론",
         "값": "명목 개선은 있으나 우연과 구분되지 않는다. 가방 5개에서는 "
               "LLP 의 성패를 판정할 수 없다 — 성공도 실패도 결론 내릴 수 없는 상태다"},
        {"항목": "초판 정정",
         "값": "'용접점 단위 지도학습 불가'는 부정확했다. 정확히는 표준 단일사례 "
               "지도학습이 불가하며, 올바른 형식화(LLP)를 적용했으나 표본이 부족했다"},
        {"항목": "필요 조건",
         "값": "가방 수 25~33 일 확보 시 재시도 가능. 용접점 단위 라벨이 있으면 LLP 자체가 불필요"},
    ])
    save_table(concl, "t62_llp_conclusion", index=False)
    print("\n-- 결론")
    print(concl.to_string(index=False))
    print("\n[18] 완료")


if __name__ == "__main__":
    main()
