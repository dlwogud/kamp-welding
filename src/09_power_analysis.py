"""09. 검정력 분석 — 공정조건-불량 관계를 검출하려면 며칠이 필요한가.

03 과 07 에서 공정조건 변수는 일별 불량건수를 설명하지 못했다. 그러나
'관계가 없다'와 '관계를 검출할 표본이 없다'는 전혀 다른 결론이다.
본 절은 둘을 구분한다.

  (1) 현재 표본(8일)의 검출력 : 어느 정도 크기의 효과까지 검출 가능한가
  (2) 필요 표본 : 통상적 효과크기를 80% 검출력으로 잡으려면 며칠이 필요한가

몬테카를로로 추정한다. 귀무분포는 라벨 순열, 대립분포는 지정한 모상관에서
이변량 정규를 생성해 Spearman 상관의 검정력을 구한다.
결과는 제4장의 데이터 수집 개선 제언에 직접 연결된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from common import FIG, save_table, setup_mpl

plt = setup_mpl()
RNG = np.random.default_rng(42)
N_SIM = 4000
ALPHA = 0.05
CURRENT_N = 8


def power_at(n: int, rho: float, n_sim: int = N_SIM) -> float:
    """표본 n, 모상관 rho 에서 Spearman 검정이 유의를 내는 비율."""
    mean = [0.0, 0.0]
    cov = [[1.0, rho], [rho, 1.0]]
    hits = 0
    for _ in range(n_sim):
        xy = RNG.multivariate_normal(mean, cov, size=n)
        _, p = spearmanr(xy[:, 0], xy[:, 1])
        if p < ALPHA:
            hits += 1
    return hits / n_sim


def detectable_rho(n: int, target_power: float = 0.80) -> float:
    """표본 n 에서 target_power 로 검출 가능한 최소 모상관 (이분탐색)."""
    lo, hi = 0.0, 0.99
    for _ in range(14):
        mid = (lo + hi) / 2
        if power_at(n, mid, n_sim=1500) < target_power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def required_n(rho: float, target_power: float = 0.80) -> int:
    """모상관 rho 를 target_power 로 검출하는 데 필요한 최소 표본."""
    for n in range(5, 200):
        if power_at(n, rho, n_sim=1200) >= target_power:
            return n
    return -1


def t28_power_curve() -> pd.DataFrame:
    rows = []
    for rho in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p8 = power_at(CURRENT_N, rho)
        need = required_n(rho)
        rows.append(
            {
                "모상관 rho": rho,
                "효과 크기": {0.3: "작음", 0.4: "작음~중간", 0.5: "중간", 0.6: "중간~큼",
                            0.7: "큼", 0.8: "매우 큼"}[rho],
                f"현재 표본(n={CURRENT_N}) 검정력": f"{p8*100:.1f}%",
                "80% 검정력 필요 작업일": need,
                "추가 수집 필요": max(0, need - CURRENT_N),
            }
        )
    t = pd.DataFrame(rows)
    save_table(t, "t28_power_analysis", index=False)
    return t


def t29_summary() -> pd.DataFrame:
    rho_min = detectable_rho(CURRENT_N)
    obs_max = 0.558  # 03 에서 관측된 최대 |rho| (전류_표준편차)
    t = pd.DataFrame(
        [
            {"항목": "현재 라벨 표본", "값": f"{CURRENT_N} 작업일"},
            {"항목": "80% 검정력으로 검출 가능한 최소 상관",
             "값": f"|rho| >= {rho_min:.2f}"},
            {"항목": "실제 관측된 최대 상관 (전류_표준편차)",
             "값": f"{obs_max:.3f}"},
            {"항목": "판정",
             "값": "관측 상관이 검출 가능 하한에 못 미친다 — "
                   "'관계 없음'이 아니라 '검출 불가'로 보고해야 한다"},
            {"항목": "중간 효과(rho=0.5) 검출에 필요한 작업일",
             "값": f"{required_n(0.5)} 일"},
            {"항목": "현행 수집속도 기준 소요기간",
             "값": f"약 {required_n(0.5)/5:.0f} 주 (주 5 작업일 가정)"},
        ]
    )
    save_table(t, "t29_power_summary", index=False)
    return t


def f17(curve: pd.DataFrame):
    ns = list(range(5, 61))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))

    for rho, c in zip([0.4, 0.5, 0.6, 0.7], ["#95a5a6", "#2e86c1", "#1f4e79", "#27ae60"]):
        ps = [power_at(n, rho, n_sim=800) * 100 for n in ns]
        a1.plot(ns, ps, "-", color=c, lw=2, label=f"ρ = {rho}")
    a1.axhline(80, ls="--", color="#c0392b", lw=1.3)
    a1.text(56, 82, "80% 검정력", fontsize=8, color="#c0392b", ha="right")
    a1.axvline(CURRENT_N, ls=":", color="black", lw=1.5)
    a1.text(CURRENT_N + 1, 8, f"현재 n={CURRENT_N}", fontsize=8.5)
    a1.set_xlabel("라벨이 있는 작업일 수")
    a1.set_ylabel("검정력 (%)")
    a1.set_title("표본 크기별 검정력 (Spearman, α=0.05)", fontsize=10.5)
    a1.legend(fontsize=8.5)

    x = np.arange(len(curve))
    a2.bar(x, curve["80% 검정력 필요 작업일"], color="#1f4e79", alpha=0.85)
    a2.axhline(CURRENT_N, ls="--", color="#c0392b", lw=1.5)
    a2.text(len(curve) - 0.4, CURRENT_N + 1.5, f"현재 보유 {CURRENT_N}일",
            fontsize=8.5, color="#c0392b", ha="right")
    for i, v in enumerate(curve["80% 검정력 필요 작업일"]):
        a2.text(i, v + 1, str(v), ha="center", fontsize=9)
    a2.set_xticks(x)
    a2.set_xticklabels([f"ρ={r}" for r in curve["모상관 rho"]], fontsize=9)
    a2.set_ylabel("필요 작업일 수")
    a2.set_title("80% 검정력 달성에 필요한 표본", fontsize=10.5)

    fig.suptitle(
        "공정조건과 불량의 관계를 검출하려면 현재보다 훨씬 많은 라벨이 필요하다",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIG / "f17_power_analysis.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f17_power_analysis.png")


def main():
    print("\n[09] 검정력 분석")
    print("  (몬테카를로 시뮬레이션 — 수십 초 소요)")
    curve = t28_power_curve()
    print("\n-- 효과크기별 검정력과 필요 표본")
    print(curve.to_string(index=False))

    print("\n-- 요약")
    print(t29_summary().to_string(index=False))

    f17(curve)
    print("\n[09] 완료")


if __name__ == "__main__":
    main()
