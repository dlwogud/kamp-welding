"""04. 대조 분석 — '용접점 단위 이진분류'를 하면 왜 안 되는가.

본 데이터에는 용접점 단위 불량 라벨이 없다. 그럼에도 11,939행 이진분류를 수행하려면
작업일 라벨을 행으로 끌어내려야(전가해야) 하는데, 이는 두 가지 방식뿐이다.

  방식 A (건수 전가) : 그 날 불량건수만큼의 행을 임의로 골라 1 로 둔다 (39/11,939 = 0.33%)
  방식 B (일자 전가) : 불량이 많았던 날의 모든 행을 1 로 둔다

두 방식 모두 라벨을 만들어낸 것이지 관측한 것이 아니다. 이 스크립트는
그렇게 만든 라벨로 학습했을 때 나오는 높은 AUC 가 실제 판별력이 아니라
'작업일 식별'에 불과함을 세 가지 대조로 증명한다.

  대조 1. 분할 방식 : 행 무작위 분할 vs 작업일 그룹 분할 (GroupKFold)
  대조 2. 라벨 셔플 : 라벨을 무작위로 섞은 뒤 같은 절차를 반복
  대조 3. 순열검정 : 관측 AUC 가 귀무분포 안에 들어오는지

결론이 '하지 말아야 한다'로 나오더라도, 그 근거를 수치로 남기는 것이 이 절의 목적이다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

from common import FIG, PROC, daily_label, load_raw, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
N_PERM = 25


def make_labels(raw: pd.DataFrame, seed: int = SEED):
    """방식 A / 방식 B 라벨 생성."""
    rng = np.random.default_rng(seed)
    dl = daily_label()
    lab = raw[raw["d"].isin(dl.index)].copy()

    # 방식 A: 그 날 불량건수만큼 임의 행을 1 로
    yA = np.zeros(len(lab), dtype=int)
    pos = lab.reset_index(drop=True)
    for dd, k in dl.items():
        idx = pos.index[pos["d"] == dd].to_numpy()
        pick = rng.choice(idx, size=int(k), replace=False)
        yA[pick] = 1

    # 방식 B: 불량건수 중앙값 초과일의 모든 행을 1 로
    med = dl.median()
    hi_days = set(dl.index[dl > med])
    yB = lab["d"].isin(hi_days).astype(int).to_numpy()

    lab = lab.reset_index(drop=True)
    lab["yA"], lab["yB"] = yA, yB
    return lab, sorted(hi_days)


def cv_scores(X, y, groups, scheme: str, seed: int = SEED):
    """교차검증 AUC/PR-AUC. scheme in {'random','group'}."""
    if scheme == "random":
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        splits = cv.split(X, y)
    else:
        cv = GroupKFold(n_splits=min(5, len(np.unique(groups))))
        splits = cv.split(X, y, groups)

    aucs, aps = [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, random_state=seed, class_weight="balanced"
        ).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
    if not aucs:
        return np.nan, np.nan
    return float(np.mean(aucs)), float(np.mean(aps))


def run(lab: pd.DataFrame) -> pd.DataFrame:
    X = lab[PROC].to_numpy()
    groups = lab["d"].astype(str).to_numpy()
    rows = []
    for tag, col in [("방식 A (건수 전가)", "yA"), ("방식 B (일자 전가)", "yB")]:
        y = lab[col].to_numpy()
        for scheme, sname in [("random", "행 무작위 분할"), ("group", "작업일 그룹 분할")]:
            auc, ap = cv_scores(X, y, groups, scheme)
            # 라벨 셔플 대조
            rng = np.random.default_rng(SEED)
            auc_s, ap_s = cv_scores(X, rng.permutation(y), groups, scheme)
            rows.append(
                {
                    "라벨 생성 방식": tag,
                    "양성 비율(%)": round(y.mean() * 100, 2),
                    "분할 방식": sname,
                    "ROC-AUC": round(auc, 3),
                    "PR-AUC": round(ap, 4),
                    "라벨셔플 ROC-AUC": round(auc_s, 3),
                    "셔플 대비 초과": round(auc - auc_s, 3),
                }
            )
    out = pd.DataFrame(rows)
    save_table(out, "t13_naive_binary_contrast", index=False)
    return out


def perm_auc(lab: pd.DataFrame, col: str, scheme: str, n_perm: int = N_PERM):
    """순열검정: 라벨을 반복 셔플해 AUC 귀무분포를 만든다."""
    X = lab[PROC].to_numpy()
    y = lab[col].to_numpy()
    groups = lab["d"].astype(str).to_numpy()
    obs, _ = cv_scores(X, y, groups, scheme)
    rng = np.random.default_rng(SEED)
    null = []
    for i in range(n_perm):
        a, _ = cv_scores(X, rng.permutation(y), groups, scheme, seed=SEED + i)
        if not np.isnan(a):
            null.append(a)
    null = np.array(null)
    p = float(((null >= obs).sum() + 1) / (len(null) + 1))
    return obs, null, p


def f7_contrast(res: pd.DataFrame):
    """그림7. 분할 방식 하나만 바꿔도 AUC 가 무너진다."""
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    tags = res["라벨 생성 방식"].unique()
    w = 0.35
    x = np.arange(len(tags))
    rnd = [res[(res["라벨 생성 방식"] == t) & (res["분할 방식"] == "행 무작위 분할")]["ROC-AUC"].iloc[0] for t in tags]
    grp = [res[(res["라벨 생성 방식"] == t) & (res["분할 방식"] == "작업일 그룹 분할")]["ROC-AUC"].iloc[0] for t in tags]
    ax.bar(x - w / 2, rnd, w, label="행 무작위 분할 (잘못된 검증)", color="#c0392b", alpha=0.85)
    ax.bar(x + w / 2, grp, w, label="작업일 그룹 분할 (올바른 검증)", color="#2e86c1", alpha=0.85)
    for xi, (a, b) in enumerate(zip(rnd, grp)):
        ax.text(xi - w / 2, a + 0.015, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(xi + w / 2, b + 0.015, f"{b:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.text(len(tags) - 0.5, 0.515, "무작위 추측 수준 (0.5)", fontsize=8, color="gray", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("전가 라벨의 높은 AUC 는 판별력이 아니라 '작업일 암기'다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f7_naive_contrast.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f7_naive_contrast.png")


def f8_perm(obs: float, null: np.ndarray, p: float, title: str, fname: str):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.hist(null, bins=30, color="#95a5a6", alpha=0.85, label=f"라벨 셔플 귀무분포 (n={len(null)})")
    ax.axvline(obs, color="#c0392b", lw=2.2, label=f"관측 AUC = {obs:.3f}")
    ax.axvline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("빈도")
    ax.set_title(f"{title}\n순열 p = {p:.3f}", fontsize=10.5)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)
    print(f"  saved -> outputs/figures/{fname}")


def main():
    print("\n[04] 대조 분석 — 용접점 단위 이진분류의 허구성 검증")
    raw = load_raw()
    lab, hi_days = make_labels(raw)
    print(f"  대상 {len(lab):,} 행 (라벨 있는 8일)")
    print(f"  방식 A 양성 {int(lab['yA'].sum())} 건 ({lab['yA'].mean()*100:.2f}%)")
    print(f"  방식 B 양성 {int(lab['yB'].sum()):,} 건 ({lab['yB'].mean()*100:.2f}%) "
          f"— 고불량일 {[str(d) for d in hi_days]}")

    print("\n-- 대조 1·2: 분할 방식 x 라벨 셔플")
    res = run(lab)
    print(res.to_string(index=False))

    print("\n-- 대조 3: 순열검정 (방식 B, 행 무작위 분할)")
    obs, null, p = perm_auc(lab, "yB", "random")
    print(f"   관측 AUC {obs:.3f} / 귀무 평균 {null.mean():.3f} / p = {p:.3f}")
    f8_perm(obs, null, p, "방식 B + 행 무작위 분할", "f8_perm_random.png")

    print("\n-- 대조 3: 순열검정 (방식 B, 작업일 그룹 분할)")
    obs2, null2, p2 = perm_auc(lab, "yB", "group")
    print(f"   관측 AUC {obs2:.3f} / 귀무 평균 {null2.mean():.3f} / p = {p2:.3f}")
    f8_perm(obs2, null2, p2, "방식 B + 작업일 그룹 분할", "f8_perm_group.png")

    f7_contrast(res)

    summary = pd.DataFrame(
        [
            {
                "검증": "방식 B · 행 무작위 분할",
                "ROC-AUC": round(obs, 3),
                "순열 p": round(p, 3),
                "해석": "같은 작업일의 행이 학습·평가에 동시에 들어가 작업일을 암기한다",
            },
            {
                "검증": "방식 B · 작업일 그룹 분할",
                "ROC-AUC": round(obs2, 3),
                "순열 p": round(p2, 3),
                "해석": "미지의 작업일에는 일반화되지 않는다. 판별력의 실체가 없다",
            },
        ]
    )
    save_table(summary, "t14_naive_conclusion", index=False)
    print("\n-- 결론")
    print(summary.to_string(index=False))
    print("\n[04] 완료")


if __name__ == "__main__":
    main()
