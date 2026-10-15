"""22. 외부 데이터 검증 (1) — 라벨이 있는 실제 제조 이상탐지 데이터에 동일 파이프라인 적용.

본 분석의 가장 큰 약점은 명확하다.
    **제2장의 성능(7유형 평균 ROC-AUC 0.971)은 전부 '우리가 설계하고 주입한 합성 이상'에
    대한 것이다.** 이상을 우리가 정의하고, 주입하고, 탐지했다. 2.10절의 시뮬레이션도
    합성이기는 마찬가지다.

본 절은 그 약점을 **외부 데이터로** 메운다.

데이터
    KAMP 제6회 경진대회 ③ 소성가공 예지보전 AI 데이터셋
      press_data_normal.csv  정상 20,000 레코드 (Equipment_state = 0)
      outlier_data.csv       이상    600 레코드 (Equipment_state = 1)
      변수: AI0_Vibration, AI1_Vibration, AI2_Current (3채널)
    출처: KAMP (www.kamp-ai.kr), 제6회 경진대회 공개 데이터셋.
    대회 공지의 '문제 출제 데이터 외의 데이터 활용 가능' 조항에 따라 사용하며,
    본 과제(② 용접기)의 답으로 쓰는 것이 아니라 **방법론 검증 목적**으로만 쓴다.

왜 이 데이터인가
    (1) 이상 라벨이 **실제로** 존재한다. 우리가 만든 것이 아니다.
    (2) 설비 공정의 연속 센서 데이터로, 용접기 데이터와 문제 구조가 같다
        (정상 운전에서 벗어난 상태를 라벨 없이 탐지).
    (3) 특징 차원이 3으로, 우리 최종 특징공간(가압력·R·E)과 같다.

검증 원칙 — 아무것도 튜닝하지 않는다
    모형, 이웃 수 k=3, 스케일러, 교정 비율, 컨포멀 절차까지 **용접기 분석에서 확정한
    설정을 그대로** 쓴다. 이 데이터에 맞춰 조정하면 검증의 의미가 사라진다.
    학습·교정에는 **정상 데이터만** 쓰고 이상 라벨은 평가에만 쓴다(용접기와 동일한 정보 조건).

시계열 처리
    100ms 간격 연속 측정이므로 관측이 독립이 아니다. 무작위 분할은 인접 시점이
    학습·평가에 함께 들어가 성능을 부풀린다. **시간순 분할**을 쓴다.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from common import FIG, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
SRC = None  # _find_src() 로 지연 탐색
FEATS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
K_NB = 3            # 용접기 분석에서 중첩 교차검증으로 선택한 값. 바꾸지 않는다.
CAL_FRAC = 0.3      # 용접기 분석과 동일
TRAIN_FRAC = 0.5    # 정상 데이터의 앞 50% 를 참조셋으로


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


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    nm = pd.read_csv(SRC / "press_data_normal.csv")
    ot = pd.read_csv(SRC / "outlier_data.csv")
    for d in (nm, ot):
        d["TimeStamp"] = pd.to_datetime(d["TimeStamp"])
    return nm.sort_values("TimeStamp"), ot.sort_values("TimeStamp")


def split_chrono(nm: pd.DataFrame):
    """시간순 분할. 참조 / 교정 / 평가(정상)."""
    n = len(nm)
    i1 = int(n * TRAIN_FRAC)
    i2 = i1 + int(n * CAL_FRAC)
    return nm.iloc[:i1], nm.iloc[i1:i2], nm.iloc[i2:]


def score_knn(ref: np.ndarray, k: int = K_NB):
    """참조셋으로 적합한 kNN 채점 함수를 돌려준다."""
    sc = RobustScaler().fit(ref)
    nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(ref))
    return (lambda Z: nn.kneighbors(sc.transform(Z))[0].mean(1)), sc


def t76_main(ref, cal, te_nm, ot) -> pd.DataFrame:
    """최종 모형(kNN k=3 + 컨포멀)의 실제 라벨 기준 성능."""
    f, _ = score_knn(ref[FEATS].to_numpy())
    s_cal = np.sort(f(cal[FEATS].to_numpy()))
    s_nm = f(te_nm[FEATS].to_numpy())
    s_ot = f(ot[FEATS].to_numpy())
    y = np.r_[np.zeros(len(s_nm)), np.ones(len(s_ot))]
    s = np.r_[s_nm, s_ot]
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    n = len(s_cal)
    pv = lambda v: (1.0 + (n - np.searchsorted(s_cal, v, side="left"))) / (n + 1.0)
    p_nm, p_ot = pv(s_nm), pv(s_ot)
    rows = []
    for a in [0.01, 0.05]:
        rows.append({"컨포멀 임계 alpha": a,
                     "실제 오경보율(%)": round(float((p_nm < a).mean()) * 100, 3),
                     "이상 탐지율(%)": round(float((p_ot < a).mean()) * 100, 1),
                     "보장 충족": "예" if (p_nm < a).mean() <= a else "아니오"})
    conf = pd.DataFrame(rows)
    save_table(conf, "t77_external_conformal", index=False)

    out = pd.DataFrame([{
        "데이터": "소성가공 예지보전 (외부, 실제 라벨)",
        "참조셋(정상)": len(ref), "교정셋(정상)": len(cal),
        "평가 정상": len(te_nm), "평가 이상": len(ot),
        "이상 비율(%)": round(len(ot) / (len(te_nm) + len(ot)) * 100, 2),
        "ROC-AUC": round(float(auc), 3),
        "PR-AUC": round(float(ap), 3),
    }])
    save_table(out, "t76_external_press", index=False)
    return out, conf, (s_nm, s_ot, p_nm, p_ot)


def t78_models(ref, te_nm, ot) -> pd.DataFrame:
    """모형 순위가 외부 데이터에서도 유지되는가.

    용접기 분석(2.4절)에서 kNN 거리가 최소최대 기준 1위였다. 그 선택이 우리 합성
    벤치마크에만 특화된 것인지, 실제 라벨 데이터에서도 유효한지 확인한다.
    설정은 전부 용접기 분석과 동일하게 둔다.
    """
    A = ref[FEATS].to_numpy()
    B = np.r_[te_nm[FEATS].to_numpy(), ot[FEATS].to_numpy()]
    y = np.r_[np.zeros(len(te_nm)), np.ones(len(ot))]
    sc = RobustScaler().fit(A)
    As, Bs = sc.transform(A), sc.transform(B)

    res = {}
    nn = NearestNeighbors(n_neighbors=K_NB).fit(As)
    res["M8 kNN 거리 (용접기 최종 선정)"] = nn.kneighbors(Bs)[0].mean(1)
    mu, S = As.mean(0), np.cov(As.T) + np.eye(As.shape[1]) * 1e-9
    d = Bs - mu
    res["M2 Hotelling T2"] = np.einsum("ij,jk,ik->i", d, np.linalg.inv(S), d)
    try:
        res["M3 마할라노비스(MCD)"] = MinCovDet(random_state=SEED,
                                             support_fraction=0.8).fit(As).mahalanobis(Bs)
    except Exception:
        pass
    res["M5 Isolation Forest"] = -IsolationForest(n_estimators=300,
                                                  random_state=SEED).fit(As).score_samples(Bs)
    res["M6 LOF"] = -LocalOutlierFactor(n_neighbors=35, novelty=True).fit(As).score_samples(Bs)
    step = max(1, len(As) // 3000)
    res["M7 One-Class SVM"] = -OneClassSVM(kernel="rbf", nu=0.02,
                                           gamma="scale").fit(As[::step]).score_samples(Bs)
    for i, c in enumerate(FEATS):
        med = np.median(As[:, i]); mad = np.median(np.abs(As[:, i] - med)) or 1e-9
        res[f"M0 단변량({c})"] = np.abs(Bs[:, i] - med) / mad

    rows = [{"모형": k, "ROC-AUC": round(float(roc_auc_score(y, v)), 3),
             "PR-AUC": round(float(average_precision_score(y, v)), 3)}
            for k, v in res.items()]
    out = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    save_table(out, "t78_external_model_ranking", index=False)
    return out


def f36(s_nm, s_ot, p_nm, p_ot, main):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    lo = min(s_nm.min(), s_ot.min()); hi = np.quantile(np.r_[s_nm, s_ot], 0.999)
    bins = np.linspace(lo, hi, 60)
    a1.hist(s_nm, bins=bins, alpha=0.75, color="#2e86c1", label=f"정상 (n={len(s_nm):,})", density=True)
    a1.hist(s_ot, bins=bins, alpha=0.75, color="#c0392b", label=f"이상 (n={len(s_ot):,})", density=True)
    a1.set_xlabel("kNN 이상점수")
    a1.set_ylabel("밀도")
    a1.legend(fontsize=9)
    a1.set_title(f"실제 라벨 기준 점수 분포  ROC-AUC = {main['ROC-AUC'].iloc[0]:.3f}", fontsize=10.5)

    from sklearn.metrics import roc_curve
    y = np.r_[np.zeros(len(s_nm)), np.ones(len(s_ot))]
    fpr, tpr, _ = roc_curve(y, np.r_[s_nm, s_ot])
    a2.plot(fpr, tpr, color="#1f4e79", lw=2.4)
    a2.plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
    a2.set_xlabel("오경보율")
    a2.set_ylabel("탐지율")
    a2.set_title("ROC 곡선 (외부 데이터, 실제 라벨)", fontsize=10.5)
    fig.suptitle("용접기 분석에서 확정한 파이프라인을 튜닝 없이 그대로 적용한 결과", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f36_external_press.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f36_external_press.png")


def main():
    global SRC
    SRC = _find_src("3. 소성가공 예지보전 AI 데이터셋", "KAMP_PRESS_DIR")
    if SRC is None:
        print("\n[22] 건너뜀 — 외부 데이터가 없습니다.")
        print("  이 절은 방법론 외부 검증이며 본 과제(② 용접기) 결론에 영향을 주지 않습니다.")
        print("  실행하려면 아래 중 하나에 데이터를 두십시오:")
        print(f"    - 환경변수 KAMP_PRESS_DIR=<폴더 경로>")
        print("    - data/external/3. 소성가공 예지보전 AI 데이터셋/")
        print("    - ~/Downloads/3. 소성가공 예지보전 AI 데이터셋/")
        return
    print("\n[22] 외부 데이터 검증 (1) — 소성가공 예지보전")
    nm, ot = load()
    print(f"  정상 {len(nm):,} 레코드 / 이상 {len(ot):,} 레코드, 변수 {FEATS}")
    ref, cal, te_nm = split_chrono(nm)
    print(f"  시간순 분할: 참조 {len(ref):,} / 교정 {len(cal):,} / 평가(정상) {len(te_nm):,}")
    print(f"  설정은 용접기 분석과 동일 (k={K_NB}, RobustScaler, 교정비율 {CAL_FRAC}). 튜닝하지 않음")

    print("\n══ (1) 최종 모형의 실제 라벨 기준 성능 ══")
    out, conf, (s_nm, s_ot, p_nm, p_ot) = t76_main(ref, cal, te_nm, ot)
    print(out.to_string(index=False))
    print("\n   컨포멀 임계별")
    print(conf.to_string(index=False))

    print("\n══ (2) 모형 순위가 외부 데이터에서도 유지되는가 ══")
    rank = t78_models(ref, te_nm, ot)
    print(rank.to_string(index=False))
    top = rank.iloc[0]["모형"]
    knn_rank = int(rank.reset_index(drop=True).index[
        rank.reset_index(drop=True)["모형"].str.startswith("M8")][0]) + 1
    print(f"\n   1위: {top}")
    print(f"   용접기에서 선정한 kNN 의 외부 데이터 순위: {knn_rank}/{len(rank)}")

    f36(s_nm, s_ot, p_nm, p_ot, out)

    auc = float(out["ROC-AUC"].iloc[0])
    concl = pd.DataFrame([
        {"항목": "목적",
         "값": "합성 이상에서만 검증된 파이프라인을 실제 라벨 데이터로 외부 검증"},
        {"항목": "데이터 출처",
         "값": "KAMP 제6회 경진대회 ③ 소성가공 예지보전 AI 데이터셋 (공개). "
               "본 과제의 답이 아니라 방법론 검증 목적"},
        {"항목": "튜닝 여부",
         "값": f"없음. k={K_NB}, 스케일러, 교정비율 모두 용접기 분석 설정 그대로"},
        {"항목": "실제 라벨 기준 ROC-AUC", "값": f"{auc:.3f}"},
        {"항목": "kNN 의 외부 데이터 순위", "값": f"{knn_rank}/{len(rank)}"},
        {"항목": "해석",
         "값": ("합성 이상에서의 성능이 실제 이상에서도 재현된다. "
                "제2장의 수치가 합성 설계에만 특화된 것이 아님을 뒷받침한다"
                if auc >= 0.8 else
                "실제 이상에서는 성능이 낮다. 제2장의 수치는 합성 설계에 의존적일 수 있다")},
        {"항목": "한계",
         "값": "공정이 다르므로 성능 수치 자체가 용접기 데이터로 전이되지는 않는다. "
               "전이되는 것은 '절차가 실제 이상도 탐지한다'는 사실이다"},
    ])
    save_table(concl, "t79_external_conclusion", index=False)
    print("\n-- 결론")
    print(concl.to_string(index=False))
    print("\n[22] 완료")


if __name__ == "__main__":
    main()
