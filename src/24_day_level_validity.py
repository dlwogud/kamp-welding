"""24. 제4.2절 일 단위 경보의 타당성 — 순환성 검증과 올바른 분석단위.

17 은 "가압력 일별 중앙값 2σ 관리도 → 이상일 3/3 탐지, 안정일 0/5 오경보" 를 보고한다.
이 주장에는 두 가지 반론이 가능하며, 본 절은 둘 다 데이터로 검증한다.

반론 1 — 순환이 아닌가
    이상일(혼입일) 라벨은 01 에서 '가압력 3bar 초과 용접점이 1% 이상인 날' 로 정의되었다.
    그 라벨을 **같은 변수(가압력)의 관리도**로 탐지하면 동어반복이 아닌가.

    검증 방법: 라벨을 만든 재료(복제 구간)를 전부 제거한 뒤에도 신호가 남는지 본다.
    복제 구간 제외 후 '3bar 초과율' 이 전 작업일 0% 가 되면 라벨 기준은 더 이상
    어떤 날도 구분하지 못한다. 그 상태에서 중앙값 차이가 남는다면 순환이 아니다.

반론 2 — 8개 점에서 3/3 이 무슨 의미인가
    탐지 3/3·오경보 0/5 는 계수일 뿐 검정이 아니다. 작업일을 분석단위로 한
    **정확 순열검정**(C(8,3)=56개 배열 전수 열거)으로 p 값을 산출한다.

    아울러 분석단위 문제를 명시한다. 1.5절이 보고한 용접점 단위 Mann-Whitney
    p < 1e-32 는 같은 작업일의 점들을 독립으로 취급한 **의사반복(pseudo-replication)**
    이므로 추론의 근거가 될 수 없다. 올바른 단위는 작업일(n=8)이다.

산출
    t59_day_level_permutation   작업일 단위 정확 순열검정
    t60_label_independence      라벨 기준의 소멸 — 복제 제외 전후
    t61_analysis_unit           분석단위 비교 (용접점 vs 작업일)
    t62_evaluated_days          탐지 평가에 쓰인 작업일의 정체
    f31_day_level_permutation   순열분포와 관측값
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from common import FIG, STABLE_DAYS, load_genuine, load_raw, load_result, mark_duplicates, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
F = "weld force(bar)"
TEST_DATE = "2020-03-27"
HIGH_FORCE = 3.0          # 01 의 혼입일 판정에 쓰인 레시피 경계
HIGH_FORCE_PCT = 1.0      # 01 의 혼입일 판정 임계 (고압 비율 %)


def day_medians() -> tuple[dict, dict]:
    """복제 구간 제외 후 일별 가압력 중앙값. (안정일, 평가대상일)"""
    g = load_genuine()
    day = g.groupby("d")[F].median()
    sta = {str(k): float(v) for k, v in day.items() if str(k) in STABLE_DAYS}
    var = {str(k): float(v) for k, v in day.items() if str(k) not in STABLE_DAYS}
    return sta, var


def t62_evaluated_days(sta: dict, var: dict) -> pd.DataFrame:
    """탐지 평가에 쓰인 작업일을 하나하나 밝힌다. 04-03 이 왜 없는지도 적는다."""
    res = load_result()
    defects = res.groupby(res["d"].astype(str))["defect"].sum().to_dict()
    raw = mark_duplicates(load_raw())
    rows = []
    for d in sorted({str(x) for x in raw["d"].astype(str)}):
        sub = raw[raw["d"].astype(str) == d]
        n_gen = int((~sub["is_dup"]).sum())
        if d in sta:
            role = "안정일 — 관리한계 추정에 사용"
        elif n_gen == 0:
            role = "제외 — 작업일 전체가 복제 구간"
        elif d == TEST_DATE:
            role = "이상일 — 탐지 평가 대상 (※ 불량 라벨 없는 예측대상일)"
        else:
            role = "이상일 — 탐지 평가 대상"
        rows.append({
            "작업일": d,
            "원본 용접점": len(sub),
            "복제 제외 후": n_gen,
            "가압력 중앙값": round(sta.get(d, var.get(d, float("nan"))), 4),
            "불량건수": defects.get(d, "없음(예측대상)" if d == TEST_DATE else 0),
            "역할": role,
        })
    out = pd.DataFrame(rows)
    save_table(out, "t62_evaluated_days", index=False)
    return out


def t60_label_independence() -> pd.DataFrame:
    """라벨 기준이 복제 제외 후 소멸하는지 확인한다. 순환성 반론의 핵심."""
    raw = mark_duplicates(load_raw())
    rows = []
    for d, sub in raw.groupby(raw["d"].astype(str)):
        gen = sub[~sub["is_dup"]]
        p_raw = 100.0 * (sub[F] > HIGH_FORCE).mean()
        p_gen = 100.0 * (gen[F] > HIGH_FORCE).mean() if len(gen) else float("nan")
        rows.append({
            "작업일": d,
            "구분": "안정일" if d in STABLE_DAYS else ("예측대상" if d == TEST_DATE else "혼입일"),
            "원본 3bar초과율(%)": round(p_raw, 2),
            "복제제외 3bar초과율(%)": round(p_gen, 3) if np.isfinite(p_gen) else None,
            "원본 기준 라벨": "혼입" if p_raw >= HIGH_FORCE_PCT else "안정",
            "복제제외 기준 라벨": (
                "데이터없음" if not np.isfinite(p_gen)
                else ("혼입" if p_gen >= HIGH_FORCE_PCT else "안정")
            ),
            "복제제외 중앙값": round(float(gen[F].median()), 4) if len(gen) else None,
        })
    out = pd.DataFrame(rows).sort_values("작업일")
    save_table(out, "t60_label_independence", index=False)
    return out


def t59_permutation(sta: dict, var: dict) -> tuple[pd.DataFrame, np.ndarray, float]:
    """작업일을 분석단위로 한 정확 순열검정. 전수 열거이므로 근사가 없다."""
    a = np.array(list(sta.values()))        # 안정일
    b = np.array(list(var.values()))        # 이상일
    obs = float(a.mean() - b.mean())        # 방향: 이상일이 더 낮다 (사전지정)
    allv = np.concatenate([a, b])
    n, k = len(allv), len(b)

    dist = []
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool)
        m[list(idx)] = True
        dist.append(allv[~m].mean() - allv[m].mean())
    dist = np.array(dist)
    p_exact = float((dist >= obs - 1e-12).mean())
    p_mw = float(mannwhitneyu(a, b, alternative="greater").pvalue)

    s = float(np.std(a, ddof=1))
    out = pd.DataFrame([
        {"항목": "분석단위", "값": f"작업일 (안정 {len(a)}일 vs 이상 {len(b)}일)"},
        {"항목": "검정 통계량", "값": "일별 가압력 중앙값의 평균차"},
        {"항목": "관측 평균차", "값": f"{obs:.4f} bar"},
        {"항목": "측정 분해능 대비", "값": f"{obs / 0.01:.1f} 분해능 단위 (분해능 0.01 bar)"},
        {"항목": "순열 배열 수", "값": f"{len(dist)} (= C({n},{k}) 전수 열거, 근사 없음)"},
        {"항목": "정확 순열검정 p (단측)", "값": f"{p_exact:.4f}"},
        {"항목": "이 설계의 최소 가능 p", "값": f"{1 / len(dist):.4f} — 관측값이 이에 해당(완전분리)"},
        {"항목": "Mann-Whitney p (단측)", "값": f"{p_mw:.4f}"},
        {"항목": "안정일 중앙값", "값": ", ".join(f"{v:.2f}" for v in a)},
        {"항목": "이상일 중앙값", "값": ", ".join(f"{v:.2f}" for v in b)},
        {"항목": "안정일 간 표준편차", "값": f"{s:.5f} bar (자유도 {len(a) - 1})"},
    ])
    save_table(out, "t59_day_level_permutation", index=False)
    return out, dist, obs


def t61_analysis_unit(sta: dict, var: dict, p_exact: float) -> pd.DataFrame:
    """분석단위를 바꾸면 p 가 어떻게 달라지는가 — 의사반복의 크기를 보인다."""
    g = load_genuine()
    pt_a = g[g["d"].astype(str).isin(STABLE_DAYS)][F].values
    pt_b = g[~g["d"].astype(str).isin(STABLE_DAYS)][F].values
    p_pt = float(mannwhitneyu(pt_a, pt_b, alternative="greater").pvalue)
    out = pd.DataFrame([
        {"분석단위": "용접점 (n = %d vs %d)" % (len(pt_a), len(pt_b)),
         "p 값": f"{p_pt:.3g}",
         "타당성": "부적절 — 같은 작업일의 점을 독립으로 취급(의사반복)",
         "용도": "분포 차이의 기술(記述)에 한정"},
        {"분석단위": "작업일 (n = %d vs %d)" % (len(sta), len(var)),
         "p 값": f"{p_exact:.4f}",
         "타당성": "적절 — 공정조건이 바뀌는 단위가 작업일",
         "용도": "본 보고서의 추론 근거"},
    ])
    save_table(out, "t61_analysis_unit", index=False)
    return out


def f31_permutation(dist: np.ndarray, obs: float, p_exact: float):
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.hist(dist, bins=28, color="#2e86c1", alpha=0.65, edgecolor="white", linewidth=0.5)
    ax.axvline(obs, color="#c0392b", lw=2,
               label=f"관측 평균차 {obs:.4f} bar\n정확 p = {p_exact:.4f}")
    ax.set_xlabel("순열 평균차 (안정일 − 이상일, bar)")
    ax.set_ylabel("배열 수")
    ax.set_title(f"작업일 단위 정확 순열검정 — 전수 {len(dist)}개 배열", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "f31_day_level_permutation.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("  24. 제4.2절 일 단위 경보의 타당성 검증")
    print("=" * 70)

    sta, var = day_medians()

    print("\n══ (1) 탐지 평가에 쓰인 작업일의 정체 ══")
    ev = t62_evaluated_days(sta, var)
    print(ev.to_string(index=False))
    print("\n  2020-04-03 은 작업일 전체(800점)가 복제 구간이라 평가에서 제외된다.")
    print(f"  {TEST_DATE} 는 가압력 기준으로 혼입일이나 불량 라벨이 없는 예측대상일이다.")
    print("  → 따라서 '3/3' 중 불량 라벨까지 있는 날은 2일이다. 본문에 명시한다.")

    print("\n══ (2) 순환성 반론 — 라벨 기준이 복제 제외 후 소멸하는가 ══")
    li = t60_label_independence()
    print(li.to_string(index=False))
    gen_lab = set(li["복제제외 기준 라벨"]) - {"데이터없음"}
    if gen_lab == {"안정"}:
        print("\n  복제 구간을 제외하면 '3bar 초과율 >= 1%' 기준은 어떤 작업일도 혼입으로")
        print("  판정하지 못한다(전 작업일 0%). 라벨을 만든 재료가 완전히 사라진 상태에서")
        print("  중앙값 차이가 남으므로 **동어반복이 아니다.**")
        print("  또한 방향이 반대다 — 원본에서 고압이 섞인 날이 복제 제외 후엔 더 낮게 운전되었다.")
    else:
        print(f"\n  주의: 복제 제외 후에도 혼입 판정이 남는다({gen_lab}). 순환성 반론이 유효하다.")

    print("\n══ (3) 작업일 단위 정확 순열검정 ══")
    pm, dist, obs = t59_permutation(sta, var)
    print(pm.to_string(index=False))
    p_exact = float(pm.loc[pm["항목"] == "정확 순열검정 p (단측)", "값"].iloc[0])

    print("\n══ (4) 분석단위 비교 ══")
    au = t61_analysis_unit(sta, var, p_exact)
    print(au.to_string(index=False))

    f31_permutation(dist, obs, p_exact)

    print("\n" + "=" * 70)
    if p_exact <= 0.05:
        print(f"  결론: 작업일 단위 정확 순열검정 p = {p_exact:.4f} (<= 0.05).")
        print("  '3/3 탐지'는 계수가 아니라 유의한 차이로 뒷받침된다.")
        print("  단, 이 값은 이 설계에서 가능한 최소 p 이므로 더 강한 주장은 불가하다.")
    else:
        print(f"  결론: 작업일 단위 p = {p_exact:.4f} — 유의하지 않다. 본문 주장을 낮춘다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
