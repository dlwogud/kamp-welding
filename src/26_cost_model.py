"""26. 금액 단위 손익 — 현장이 자기 단가를 넣어 바로 판단하게 만든다.

왜 필요한가
    17 의 손익분기는 **농축배수 r\\*** 라는 무차원 임계로만 제시되어 있다. 이론적으로는
    충분하나 현장에서 "그래서 얼마냐" 에 답하지 못한다. 반대로 임의의 단가를 가정해
    "연 N천만원 절감" 을 적으면 **근거 없는 수치**가 된다.

    그래서 택한 방식: **물량은 데이터에서 실측하고, 단가는 모수로 남긴다.**
    현장이 자기 재검사 단가(c_i)와 불량 유출 비용(c_d)을 넣으면 금액이 확정된다.

본 절이 산출하는 것
    (1) 재검사 물량 — 임계별 일/월/연 환산 (실측)
    (2) 비용식과 손익분기 단가 — c_i 를 모수로 둔 폐형식
    (3) 금액 표 — 현실적 단가 구간에서의 일/연 비용과 순이익 (모수 조합 전개)
    (4) 일 단위 경보의 가치 — 작업일 격리 대안과의 비교

지키는 원칙
    - 실측값과 가정값을 표에서 분리 표기한다.
    - 검출률(민감도)은 추정할 수 없으므로 **가정하지 않는다.** 대신 손익분기에
      필요한 농축배수 r 로 표현하고, r 은 현장이 1개월 재검사 기록으로 측정할 수 있다.
    - "절감액" 을 단일 수치로 제시하지 않는다. 전부 조건부로 적는다.

산출
    t64_rework_volume      재검사 물량 (실측)
    t65_breakeven_price    손익분기 재검사 단가
    t66_cost_grid          단가 조합별 일/연 비용과 순이익
    t67_day_alarm_value    일 단위 경보의 비용 구조
    f32_cost_breakeven     손익분기선
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from common import FIG, OUT, STABLE_DAYS, load_genuine, load_result, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
F = "weld force(bar)"

WORKDAYS_PER_MONTH = 20     # 본 데이터가 8 작업일/약 2주 → 월 20일 가정. 가정값으로 표기한다.
WORKDAYS_PER_YEAR = 240


ALPHA_FLAG = 0.01    # 17 과 동일한 운영 임계


def load_scores() -> pd.DataFrame:
    """17 과 **같은 기준**을 쓴다 — LODO 채점본의 안정일만.

    기준을 맞추는 것이 중요하다. 최종 산출물 weld_anomaly_scores.csv 는 평가일을
    포함하고 교정 방식도 달라 p<0.01 플래그 비율이 17.8% 로 나온다. 그 값을 쓰면
    4.5절의 f = 1.83% 와 어긋나고, 비용 계산이 과대해진다.

    운영 부하의 올바른 기준은 **정상 가동일에 몇 점을 재검사하는가** 이므로
    안정일 LODO 채점본을 쓴다.
    """
    s = pd.read_csv(OUT / "weld_scores_lodo.csv", encoding="utf-8-sig")
    s["d"] = pd.to_datetime(s["d"]).dt.date
    return s[s["d"].astype(str).isin(STABLE_DAYS)].copy()


def t64_volume(s: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """재검사 물량. 전부 실측이다."""
    n_total = len(s)
    day_n = g[g["d"].astype(str).isin(STABLE_DAYS)].groupby("d").size()
    rows = []
    for name, thr in [("p < 0.0038 (팽창계수 반영 명목값)", 0.0038),
                      ("p < 0.01 (4.5절 기준)", 0.01),
                      ("p < 0.05", 0.05)]:
        flagged = int((s["conformal_p"] < thr).sum())
        frac = flagged / n_total
        rows.append({
            "재검사 임계": name,
            f"플래그 점수(안정일 {n_total:,} 중)": flagged,
            "플래그 비율 f": f"{frac * 100:.2f}%",
            "실현가능 최대 농축배수 1/f": round(1 / frac, 1) if frac > 0 else np.inf,
            "일평균 재검사 점수": round(frac * day_n.mean(), 1),
            "일최대 재검사 점수": round(frac * day_n.max(), 1),
            "월 환산(20일)": round(frac * day_n.mean() * WORKDAYS_PER_MONTH, 0),
            "연 환산(240일)": round(frac * day_n.mean() * WORKDAYS_PER_YEAR, 0),
            "구분": "실측 × 작업일수 가정",
        })
    out = pd.DataFrame(rows)
    save_table(out, "t64_rework_volume", index=False)
    return out


def breakeven_ci(c_d: float, q: float, r: float, f: float) -> float:
    """손익분기 재검사 단가 c_i*.

    17 의 조건 c_i·f = c_d·q·r·f / (r·f + 1 − f) 를 c_i 에 대해 풀면
        c_i* = c_d · q · r / (r·f + 1 − f)
    c_i < c_i* 이면 재검사가 이득이다. f·c_i 가 양변에서 약분되므로
    c_i* 는 플래그 '비율' 에 직접 비례하지 않는다.
    """
    return c_d * q * r / (r * f + 1.0 - f)


def t65_breakeven_price(f: float) -> pd.DataFrame:
    """불량 유출 비용과 농축배수를 모수로 둔 손익분기 재검사 단가."""
    rows = []
    for c_d in (10_000, 50_000, 100_000, 500_000):
        for q in (0.001, 0.005, 0.01):
            rec = {"불량 유출 1건 비용 c_d": f"{c_d:,}원", "전체 불량률 q": q}
            for r in (2, 5, 10, 30):
                rec[f"r={r}"] = f"{breakeven_ci(c_d, q, r, f):,.0f}원"
            rows.append(rec)
    out = pd.DataFrame(rows)
    save_table(out, "t65_breakeven_price", index=False)
    return out


def t66_cost_grid(s: pd.DataFrame, g: pd.DataFrame, obs_defects_per_day: float) -> pd.DataFrame:
    """단가 조합별 일/연 비용과 순이익. 전부 조건부이며 실현가능성을 검사한다.

    **반드시 지켜야 하는 물리 제약**: 재검사로 잡는 불량이 그 날 발생한 불량보다
    많을 수 없다. 잡는 건수 = n_day·f·q·r, 발생 건수 = n_day·q 이므로 제약은 r <= 1/f 다.
    이를 어기는 조합은 계산하지 않고 '실현 불가' 로 적는다.
    """
    day_n = float(g[g["d"].astype(str).isin(STABLE_DAYS)].groupby("d").size().mean())
    rows = []
    for thr, tname in [(0.0038, "p<0.0038"), (0.01, "p<0.01")]:
        f = float((s["conformal_p"] < thr).mean())
        n_day = f * day_n
        r_max = 1.0 / f
        for c_i in (500, 1_000, 3_000):
            cost_day = n_day * c_i
            for c_d, q, r in [(100_000, 0.005, 5), (100_000, 0.005, 10),
                              (500_000, 0.005, 5), (500_000, 0.01, 10)]:
                produced = day_n * q                     # 그 날 발생하는 불량 건수
                caught = n_day * q * r                   # 재검사로 잡는 건수
                feasible = r <= r_max and caught <= produced + 1e-9
                # q 가 실측 일평균 불량건수와 정합한가. 2.7.1 에서 불량건수가 생산량에
                # 비례하지 않음을 확인했으므로 q 는 식별되지 않는다. 관측 대비만 적는다.
                q_ratio = produced / obs_defects_per_day
                benefit_day = caught * c_d
                rows.append({
                    "임계": tname,
                    "일 재검사 점수": round(n_day, 1),
                    "재검사 단가 c_i": f"{c_i:,}원",
                    "유출비용 c_d": f"{c_d:,}원",
                    "q": q, "r": r,
                    "일 발생 불량(가정 q)": round(produced, 2),
                    "실측 4.875건 대비": f"{q_ratio:.2f}배",
                    "일 적출 불량": round(caught, 2),
                    "일 재검사 비용": f"{cost_day:,.0f}원",
                    "일 기대 회피손실": f"{benefit_day:,.0f}원",
                    "일 순이익": f"{benefit_day - cost_day:+,.0f}원",
                    "연 순이익(240일)": f"{(benefit_day - cost_day) * WORKDAYS_PER_YEAR:+,.0f}원",
                    "판정": ("이득" if benefit_day > cost_day else "손실") if feasible else "실현 불가",
                })
    out = pd.DataFrame(rows)
    save_table(out, "t66_cost_grid", index=False)
    return out


def t67_day_alarm(g: pd.DataFrame) -> pd.DataFrame:
    """일 단위 경보가 만드는 선택지의 비용 구조. 경보 1회당으로 적는다."""
    day_n = g.groupby("d").size()
    mean_n, max_n = float(day_n.mean()), float(day_n.max())
    rows = [
        {"대응": "① 작업일 전량 재검사", "대상 점수": f"{mean_n:,.0f} (최대 {max_n:,.0f})",
         "경보 1회 비용": "c_i × 1,092 (최대 2,000)",
         "성격": "확실하나 가장 비싸다"},
        {"대응": "② 컨포멀 상위만 재검사 (p<0.01)", "대상 점수": f"{mean_n * 0.0183:,.0f}",
         "경보 1회 비용": f"c_i × {mean_n * 0.0183:,.0f}",
         "성격": "전량의 1.83%. 다만 검출률 미지"},
        {"대응": "③ 설비조건 점검 후 재가동", "대상 점수": "0",
         "경보 1회 비용": "점검 인시 + 정지손실",
         "성격": "**권고.** 경보가 공정 이동을 가리키므로 원인 제거가 맞다"},
        {"대응": "④ 무대응", "대상 점수": "0", "경보 1회 비용": "0",
         "성격": "이동이 불량으로 이어진다는 근거는 없으나, 사양 이탈 위험을 방치"},
    ]
    out = pd.DataFrame(rows)
    save_table(out, "t67_day_alarm_value", index=False)
    return out


def f32_plot(f: float):
    """손익분기선 — 현장이 자기 점을 찍어 위/아래를 보면 된다."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    r = np.linspace(1, 40, 300)
    for c_d, q, c in [(100_000, 0.005, "#2e86c1"), (500_000, 0.005, "#c0392b"),
                      (100_000, 0.010, "#27ae60"), (500_000, 0.010, "#8e44ad")]:
        ax.plot(r, [breakeven_ci(c_d, q, x, f) for x in r], color=c, lw=1.9,
                label=f"c_d={c_d // 10000}만원, q={q:.3f}")
    ax.set_xlabel("농축배수 r  (플래그 점이 정상 점보다 몇 배 불량한가)")
    ax.set_ylabel("손익분기 재검사 단가 $c_i^*$ (원/점)")
    ax.set_title(f"선 아래쪽이면 재검사가 이득이다 (플래그 비율 f = {f * 100:.2f}%)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "f32_cost_breakeven.png", dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("  26. 금액 단위 손익 — 물량은 실측, 단가는 모수")
    print("=" * 70)

    s, g = load_scores(), load_genuine()
    f = float((s["conformal_p"] < ALPHA_FLAG).mean())
    res = load_result()
    obs_dpd = float(res["defect"].sum()) / res["d"].nunique()
    print(f"\n  기준: 안정일 LODO 채점본 {len(s):,}점, p < {ALPHA_FLAG} → f = {f * 100:.2f}%")
    print(f"  (4.5절의 f = 1.83% 와 같은 기준이다)")
    print(f"  실현가능 최대 농축배수 1/f = {1 / f:.1f}")
    print(f"  실측 일평균 불량 {obs_dpd:.3f}건 → 이를 넘는 적출량은 물리적으로 불가능하다")

    print("\n══ (1) 재검사 물량 (실측) ══")
    print(t64_volume(s, g).to_string(index=False))

    print("\n══ (2) 손익분기 재검사 단가 c_i* ══")
    print("  c_i* = c_d · q · r / (r·f + 1 − f),  f = %.4f (실측)" % f)
    print("  c_i < c_i* 이면 재검사가 이득이다.")
    print(t65_breakeven_price(f).to_string(index=False))

    print("\n══ (3) 단가 조합별 일/연 순이익 (전부 조건부) ══")
    grid = t66_cost_grid(s, g, obs_dpd)
    print(grid.to_string(index=False))
    n_win = int((grid["판정"] == "이득").sum())
    n_bad = int((grid["판정"] == "실현 불가").sum())
    print(f"\n  제시한 {len(grid)}개 조합 중 이득 {n_win}개, 손실 {len(grid)-n_win-n_bad}개, 실현 불가 {n_bad}개.")
    print(f"  q = 0.005 는 일 발생 불량 6.34건으로 실측 {obs_dpd:.3f}건의 1.30배,")
    print(f"  q = 0.010 은 12.68건으로 2.60배다. 후자는 관측보다 크므로 보수적 가정이 아니다.")
    print("  2.7.1 에서 불량건수가 생산량에 비례하지 않음을 확인했으므로 q 는 식별되지 않는다.")
    print("  어느 조합이 현장에 해당하는지는 c_i·c_d·q·r 를 현장이 넣어야 결정된다.")

    print("\n══ (4) 일 단위 경보의 대응 선택지 ══")
    print(t67_day_alarm(g).to_string(index=False))

    f32_plot(f)

    print("\n" + "=" * 70)
    print("  r 은 현장이 1개월 재검사 기록으로 측정할 수 있는 유일한 미지수다.")
    print("  본 데이터로는 추정할 수 없으므로 임계와 조건만 제시한다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
