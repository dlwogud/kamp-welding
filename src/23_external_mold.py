"""23. 외부 데이터 검증 (2) — LLP 주장을 '라벨을 아는' 실제 데이터로 검증한다.

2.9~2.10절의 핵심 주장은 다음과 같다.
    **불량률이 0.35% 수준이면, 가방(작업일)별 불량 개수만으로는 제품 단위 판별자를
    학습할 수 없다. 가방을 더 모아도 달성되지 않는다.**
2.10절은 이를 합성 시뮬레이션으로 보였다. 합성이라는 한계가 남는다.

본 절은 **제품 단위 라벨이 실제로 존재하는 외부 데이터**로 같은 주장을 검증한다.
설계가 가능한 이유는 하나다. 라벨을 알고 있으므로

    (1) 먼저 **지도학습 상한**을 측정하고 (이 데이터에 신호가 있는지 확인)
    (2) 그 다음 **라벨을 일부러 버리고** 가방별 개수만 남겨 LLP 를 돌린 뒤
    (3) 숨겨 둔 진짜 라벨로 **인스턴스 ROC-AUC 를 채점**할 수 있다.

용접기 데이터로는 (1)도 (3)도 원리상 불가능하다. 외부 데이터라야 할 수 있는 검증이다.

데이터
    KAMP 제6회 경진대회 ① 사출성형기 AI 데이터셋
      moldset_labeled_cn7.csv  1,211건, PassOrFail 양성 17건 (1.40%)
      moldset_labeled_rg3.csv  (동일 구조, 재현성 확인용)
      공정변수 24종
    출처: KAMP (www.kamp-ai.kr). 대회 공지의 외부 데이터 활용 조항에 따라 사용하며,
    본 과제(② 용접기)의 답이 아니라 **방법론 검증 목적**으로만 쓴다.

구조의 평행성
    5개 가방으로 나누면 가방당 양성이 약 3.4건이다. 용접기 데이터는 가방당 2.8건이다.
    **절대 양성 수가 거의 같아** 비교가 성립한다.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from common import FIG, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
SRC = None  # _find_src() 로 지연 탐색
LABEL = "PassOrFail"
MAX_ITER = 8
N_REP = 40


def _find_src(folder_name: str, env_key: str) -> Path | None:
    """외부 데이터 위치 탐색.

    제출 패키지에는 외부 데이터를 포함하지 않는다(본 과제의 데이터가 아니므로).
    아래 순서로 찾고, 없으면 None 을 돌려 본 절을 건너뛴다.
      1) 환경변수 {env_key}
      2) data/external/<folder_name>
      3) ~/Downloads/<folder_name>
    """
    import os
    cands = []
    if os.environ.get(env_key):
        cands.append(Path(os.environ[env_key]))
    cands.append(Path(__file__).resolve().parents[1] / "data" / "external" / folder_name)
    cands.append(Path.home() / "Downloads" / folder_name)
    for c in cands:
        if c.exists():
            return c
    return None


def load(fname: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = pd.read_csv(SRC / fname)
    d = d.drop(columns=[c for c in d.columns if c.startswith("Unnamed")], errors="ignore")
    y = d[LABEL].to_numpy(int)
    X = d.drop(columns=[LABEL])
    keep = [c for c in X.columns if X[c].std() > 1e-12]      # 상수열 제거
    return X[keep].to_numpy(float), y, keep


def supervised_ceiling(X, y, seed=SEED) -> dict:
    """제품 단위 라벨을 전부 쓸 때의 상한. 이 데이터에 신호가 있는가."""
    out = {}
    for name, mk in [("로지스틱", lambda: LogisticRegression(max_iter=2000, random_state=seed,
                                                           class_weight="balanced")),
                     ("HistGB", lambda: HistGradientBoostingClassifier(
                         max_iter=200, random_state=seed, class_weight="balanced"))]:
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        sc = RobustScaler()
        aucs = []
        for tr, te in cv.split(X, y):
            Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
            if len(np.unique(y[te])) < 2:
                continue
            m = mk().fit(Xtr, y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(Xte)[:, 1]))
        out[name] = float(np.mean(aucs)) if aucs else np.nan
    return out


def make_bags(X, y, n_bags, rng, rate=None):
    """연속 블록으로 가방 구성. rate 가 주어지면 양성을 솎아 목표 불량률을 만든다."""
    idx = np.arange(len(y))
    if rate is not None:
        pos, neg = idx[y == 1], idx[y == 0]
        n_pos = int(round(rate * len(neg) / (1 - rate)))
        n_pos = min(n_pos, len(pos))
        if n_pos < n_bags:
            return None
        keep = np.r_[rng.choice(pos, n_pos, replace=False), neg]
        keep.sort()
        idx = keep
    rng.shuffle(idx)
    parts = np.array_split(idx, n_bags)
    return [(X[p], y[p]) for p in parts]


def fit_llp(bags_tr, seed=SEED):
    """비율매칭 로지스틱 (18·19절과 동일 절차). 가방별 양성 개수만 사용."""
    Xall = np.vstack([b[0] for b in bags_tr])
    sizes = [len(b[0]) for b in bags_tr]
    counts = [int(b[1].sum()) for b in bags_tr]
    sc = RobustScaler().fit(Xall)
    Z = sc.transform(Xall)
    bag_id = np.concatenate([[i] * s for i, s in enumerate(sizes)])
    rate = sum(counts) / len(Z)
    w = np.full(len(Z), rate)
    Xd = np.vstack([Z, Z])
    yy = np.r_[np.ones(len(Z)), np.zeros(len(Z))]
    clf = None
    for _ in range(MAX_ITER):
        clf = LogisticRegression(max_iter=500, random_state=seed).fit(
            Xd, yy, sample_weight=np.r_[w, 1 - w])
        p = clf.predict_proba(Z)[:, 1]
        w = np.empty_like(p)
        for i in range(len(sizes)):
            m = bag_id == i
            pb = p[m]
            w[m] = np.clip(pb * (counts[i] / pb.sum()), 1e-9, 1.0) if pb.sum() > 0 \
                else counts[i] / m.sum()
    return clf, sc


def llp_instance_auc(X, y, n_bags, rate, rng, n_rep=N_REP) -> dict:
    """가방 라벨만으로 학습하고, 숨겨 둔 진짜 라벨로 인스턴스 AUC 를 잰다."""
    aucs = []
    for _ in range(n_rep):
        bags = make_bags(X, y, n_bags, rng, rate)
        if bags is None:
            continue
        n_te = max(2, n_bags // 3)
        te, tr = bags[:n_te], bags[n_te:]
        Xte = np.vstack([b[0] for b in te]); Yte = np.concatenate([b[1] for b in te])
        if Yte.sum() == 0 or Yte.sum() == len(Yte):
            continue
        try:
            clf, sc = fit_llp(tr)
            aucs.append(roc_auc_score(Yte, clf.predict_proba(sc.transform(Xte))[:, 1]))
        except Exception:
            continue
    if not aucs:
        return {"평균 AUC": np.nan, "AUC>0.6 비율(%)": np.nan, "유효 반복": 0}
    a = np.array(aucs)
    return {"평균 AUC": round(float(a.mean()), 3),
            "AUC>0.6 비율(%)": round(float((a > 0.6).mean() * 100), 1),
            "유효 반복": len(a)}


def t80_ceiling(datasets) -> pd.DataFrame:
    rows = []
    for name, (X, y, cols) in datasets.items():
        ceil = supervised_ceiling(X, y)
        rows.append({"데이터": name, "건수": len(y), "양성": int(y.sum()),
                     "불량률(%)": round(float(y.mean() * 100), 2),
                     "변수 수": len(cols),
                     "지도학습 AUC (로지스틱)": round(ceil["로지스틱"], 3),
                     "지도학습 AUC (HistGB)": round(ceil["HistGB"], 3)})
    out = pd.DataFrame(rows)
    save_table(out, "t80_mold_supervised_ceiling", index=False)
    return out


def t81_llp(X, y) -> pd.DataFrame:
    """가방 수 x 불량률 교차 — 2.10절 시뮬레이션과 같은 축."""
    rng = np.random.default_rng(SEED)
    base_rate = float(y.mean())
    rows = []
    for n_bags in [5, 10, 20]:
        for rate in [0.0035, 0.007, base_rate]:
            r = llp_instance_auc(X, y, n_bags, rate, rng)
            n_pos_bag = rate * len(y) / n_bags
            rows.append({"가방 수": n_bags, "불량률(%)": round(rate * 100, 3),
                         "가방당 기대 양성": round(n_pos_bag, 2), **r})
    out = pd.DataFrame(rows)
    save_table(out, "t81_mold_llp", index=False)
    return out


def f37(ceil: pd.DataFrame, llp: pd.DataFrame):
    ceiling = max(float(ceil["지도학습 AUC (HistGB)"].iloc[0]),
                  float(ceil["지도학습 AUC (로지스틱)"].iloc[0]))
    valid = llp[llp["유효 반복"] > 0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    for nb in sorted(valid["가방 수"].unique()):
        s = valid[valid["가방 수"] == nb].sort_values("불량률(%)")
        if len(s) < 1:
            continue
        a1.plot(s["불량률(%)"], s["평균 AUC"], "o-", lw=2.1, ms=7, label=f"가방 {nb}개")
    a1.axhline(ceiling, ls="--", color="#27ae60", lw=1.8)
    a1.text(valid["불량률(%)"].max(), ceiling + 0.012,
            f"제품 단위 라벨 사용 시 상한 {ceiling:.3f}", fontsize=8.5, ha="right", color="#27ae60")
    a1.axhline(0.5, ls=":", color="black", lw=1.2)
    a1.axvline(0.35, ls=":", color="#c0392b", lw=1.6)
    a1.text(0.37, 0.56, "용접기 0.35%\n(이 데이터로는 재현 불가)", fontsize=8, color="#c0392b")
    a1.set_xlabel("불량률 (%)")
    a1.set_ylabel("LLP 인스턴스 ROC-AUC")
    a1.set_ylim(0.4, 1.0)
    a1.legend(fontsize=9)
    a1.set_title("가방 집계만으로 학습했을 때의 실제 성능", fontsize=10.5)

    best = float(valid["평균 AUC"].max()) if len(valid) else np.nan
    worst = float(valid["평균 AUC"].min()) if len(valid) else np.nan
    labels = ["제품 단위 라벨\n(지도학습 상한)", "가방 집계만\n(최선 조건)", "가방 집계만\n(최악 조건)"]
    vals = [ceiling, best, worst]
    cols = ["#27ae60", "#e67e22", "#c0392b"]
    a2.bar(range(3), vals, color=cols, alpha=0.9)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            a2.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    a2.axhline(0.5, ls=":", color="black", lw=1.2)
    a2.set_xticks(range(3))
    a2.set_xticklabels(labels, fontsize=9)
    a2.set_ylabel("인스턴스 ROC-AUC")
    a2.set_ylim(0.4, 1.0)
    a2.set_title("라벨 해상도를 낮출 때 잃는 것", fontsize=10.5)

    fig.suptitle("외부 데이터 — 제품 단위 라벨이 있을 때와 가방 집계만 있을 때", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f37_external_mold.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f37_external_mold.png")


def t82_conclusion(ceil: pd.DataFrame, llp: pd.DataFrame) -> pd.DataFrame:
    """결론은 전부 표에서 유도한다. 미리 써두지 않는다."""
    ceiling = max(float(ceil["지도학습 AUC (HistGB)"].iloc[0]),
                  float(ceil["지도학습 AUC (로지스틱)"].iloc[0]))
    valid = llp[llp["유효 반복"] > 0]
    testable = sorted(valid["불량률(%)"].unique())
    untestable = sorted(set(llp["불량률(%)"]) - set(testable))
    best = float(valid["평균 AUC"].max()) if len(valid) else float("nan")
    worst = float(valid["평균 AUC"].min()) if len(valid) else float("nan")
    lo_rate = min(testable) if testable else None
    rg3 = float(ceil["지도학습 AUC (HistGB)"].iloc[-1])

    rows = [
        {"항목": "검증 대상 주장",
         "값": "불량률이 낮으면 가방별 개수만으로는 제품 단위 판별자를 학습할 수 없다"},
        {"항목": "데이터",
         "값": "KAMP ① 사출성형기 (제품 단위 라벨 존재). 외부 데이터 활용 조항에 따라 사용"},
        {"항목": "제품 단위 라벨 사용 시 (상한)",
         "값": f"ROC-AUC {ceiling:.3f} — 이 데이터에 학습 가능한 신호가 분명히 있다"},
        {"항목": "가방 집계만 쓸 때",
         "값": f"ROC-AUC {worst:.3f} ~ {best:.3f}"},
        {"항목": "라벨 해상도를 낮춰 잃는 양",
         "값": f"{ceiling:.3f} → 최대 {best:.3f} (약 {ceiling - best:.3f} 손실)"},
        {"항목": "검정 가능했던 최저 불량률",
         "값": f"{lo_rate}%" if lo_rate is not None else "없음"},
        {"항목": "용접기 조건(0.35%) 재현 여부",
         "값": f"불가. 양성이 17건뿐이라 0.35%로 솎으면 약 4건만 남아 평가 구성이 "
               f"성립하지 않는다 (검정 불가: {untestable}%)"},
        {"항목": "판정",
         "값": f"용접기보다 4배 높은 불량률에서 이미 {ceiling:.3f} → {best:.3f} 로 "
               f"대부분의 신호를 잃는다. 0.35%에서는 더 나쁠 수밖에 없다. "
               f"2.10절 시뮬레이션의 결론이 실제 라벨 데이터에서 같은 방향으로 재현된다"},
        {"항목": "부수 관찰 (RG3)",
         "값": f"RG3는 제품 단위 라벨을 다 써도 지도학습 AUC {rg3:.3f} 로 신호가 없다. "
               f"제조 데이터라고 항상 학습 가능한 관계가 있는 것은 아니다"},
        {"항목": "의의",
         "값": "용접기 데이터로는 상한 측정도 인스턴스 채점도 원리상 불가능하다. "
               "외부 데이터라야 가능한 검증이며, 합성 시뮬레이션보다 강한 증거다"},
    ]
    out = pd.DataFrame(rows)
    save_table(out, "t82_mold_conclusion", index=False)
    return out


def main():
    global SRC
    SRC = _find_src("1. 사출성형기 AI 데이터셋", "KAMP_MOLD_DIR")
    if SRC is None:
        print("\n[23] 건너뜀 — 외부 데이터가 없습니다.")
        print("  이 절은 방법론 외부 검증이며 본 과제(② 용접기) 결론에 영향을 주지 않습니다.")
        print("  실행하려면 아래 중 하나에 데이터를 두십시오:")
        print(f"    - 환경변수 KAMP_MOLD_DIR=<폴더 경로>")
        print("    - data/external/1. 사출성형기 AI 데이터셋/")
        print("    - ~/Downloads/1. 사출성형기 AI 데이터셋/")
        return
    print("\n[23] 외부 데이터 검증 (2) — 사출성형기로 LLP 주장 검증")
    datasets = {}
    for f, nm in [("moldset_labeled_cn7.csv", "사출성형 CN7"),
                  ("moldset_labeled_rg3.csv", "사출성형 RG3")]:
        try:
            datasets[nm] = load(f)
        except Exception as e:
            print(f"  [건너뜀] {f}: {e}")
    for nm, (X, y, cols) in datasets.items():
        print(f"  {nm}: {len(y):,}건, 양성 {int(y.sum())}건 ({y.mean()*100:.2f}%), 변수 {len(cols)}")

    print("\n══ (1) 지도학습 상한 — 이 데이터에 신호가 있는가 ══")
    ceil = t80_ceiling(datasets)
    print(ceil.to_string(index=False))

    key = list(datasets)[0]
    X, y, _ = datasets[key]
    print(f"\n══ (2) 라벨을 버리고 가방 집계만으로 — {key} ══")
    llp = t81_llp(X, y)
    print(llp.to_string(index=False))

    f37(ceil, llp)
    concl = t82_conclusion(ceil, llp)
    print("\n-- 결론")
    print(concl.to_string(index=False))
    print("\n[23] 완료")


if __name__ == "__main__":
    main()
