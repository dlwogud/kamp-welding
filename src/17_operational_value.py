"""17. 운영 효익과 그 한계 — 라벨 없이 말할 수 있는 것, 말할 수 없는 것.

제4장의 공백을 메우되, **메울 수 없는 부분은 메울 수 없다고 보고한다.**

전제 — 무엇을 대상으로 하는가
    06 에서 복제 구간이 800점 x 4일(전체의 26.8%)로 확정되었다. 본 절은 그 구간을
    **전부 제외한 실제 생산점만** 사용한다. 2020-04-03 은 작업일 전체가 복제 구간이므로
    완전히 제외된다. 따라서 대상은 안정일 5일 + 실제 공정 차이가 확인된 3일이다.

    복제 구간을 제외해도 2020-03-25·03-27·03-31 은 안정일과 유의하게 다르다.
    가압력 중앙값 2.310 vs 2.330~2.350 bar (Mann-Whitney p < 1e-32).
    **미세하지만 실재하는 공정 이동이며, 본 절의 탐지 대상이다.**

본 절이 답하는 질문
    (1) 용접점 단위 이상탐지를 순차 감시로 쓰면 이 공정 이동을 조기에 잡는가
    (2) 잡지 못한다면 무엇이 병목인가 — 탐지기인가, 데이터인가
    (3) 재검사의 손익분기는 어디인가

초판의 오류를 기록해 둔다
    본 절의 초판은 복제 구간 중 '가압력 3bar 초과' 부분만 제외하고 분석해
    '이상일 4/4 탐지, 평균 40.4% 물량 보호' 라는 결과를 냈다. 그러나 06 의 재검증에서
    복제 구간이 800점임이 확인되었고, 전 구간을 제외하자 탐지는 0/3 이 되었다.
    **초판이 탐지한 것은 공정이상이 아니라 복제 아티팩트였다.**
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import FIG, OUT, STABLE_DAYS, load_genuine, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
ALPHA_FLAG = 0.01


def load_scores() -> pd.DataFrame:
    """16 이 생성한 채점 결과. 복제 구간은 이미 제외되어 있다."""
    s = pd.read_csv(OUT / "weld_scores_lodo.csv", encoding="utf-8-sig")
    s["d"] = pd.to_datetime(s["d"]).dt.date
    return s.sort_values(["d", "idx"])


def cusum_bernoulli(p: np.ndarray, alpha: float = ALPHA_FLAG, k: float = None) -> np.ndarray:
    """플래그 지시변수의 베르누이 CUSUM. 각 점의 기여가 유계라 극단값에 지배되지 않는다."""
    k = (2.0 * alpha) if k is None else k
    x = (p < alpha).astype(float)
    s = np.zeros(len(x))
    acc = 0.0
    for i, v in enumerate(x):
        acc = max(0.0, acc + v - k)
        s[i] = acc
    return s


# ══════════════════════════════════════════════════════════════════
def t55_sequential(scored: pd.DataFrame) -> pd.DataFrame:
    """용접점 단위 순차 감시의 작동점 스윕. 경보 한계는 안정일에서만 보정한다."""
    rows = []
    for alpha in [0.01, 0.02, 0.05, 0.10, 0.20]:
        for margin in [1.0, 1.2, 1.5]:
            sta_max = [float(cusum_bernoulli(g["conformal_p"].to_numpy(), alpha).max())
                       for d, g in scored.groupby("d") if str(d) in STABLE_DAYS]
            h = max(sta_max) * margin
            det, fa = [], []
            for d, g in scored.groupby("d"):
                fired = bool((cusum_bernoulli(g["conformal_p"].to_numpy(), alpha) > h).any())
                (fa if str(d) in STABLE_DAYS else det).append(fired)
            rows.append({"alpha": alpha, "여유계수": margin, "경보한계 h": round(h, 1),
                         "이상일 탐지": f"{sum(det)}/{len(det)}",
                         "안정일 오경보": f"{sum(fa)}/{len(fa)}",
                         "유효": "예" if sum(det) > 0 and sum(fa) == 0 else "아니오"})
    out = pd.DataFrame(rows)
    save_table(out, "t55_sequential_sweep", index=False)
    return out


def t55b_daychart(gen: pd.DataFrame) -> pd.DataFrame:
    """일 단위 요약통계 관리도. 관리한계는 평가일을 제외한 안정일로만 추정한다."""
    rows = []
    for c, lab in [(F, "가압력"), (I, "전류")]:
        for stat in ["median", "mean"]:
            day = gen.groupby("d")[c].agg(stat)
            sta = {str(k): v for k, v in day.items() if str(k) in STABLE_DAYS}
            var = {str(k): v for k, v in day.items() if str(k) not in STABLE_DAYS}
            for L in [2.0, 2.5, 3.0]:
                det = fa = 0
                for k, v in {**sta, **var}.items():
                    ref = [x for kk, x in sta.items() if kk != k]
                    m, s = np.mean(ref), np.std(ref, ddof=1)
                    outside = (v < m - L * s) or (v > m + L * s)
                    if k in sta:
                        fa += outside
                    else:
                        det += outside
                rows.append({"변수": lab, "통계량": "중앙값" if stat == "median" else "평균",
                             "관리한계": f"{L:g} sigma",
                             "이상일 탐지": f"{det}/{len(var)}",
                             "안정일 오경보": f"{fa}/{len(sta)}",
                             "유효": "예" if det > 0 and fa == 0 else "아니오"})
    out = pd.DataFrame(rows)
    save_table(out, "t55b_day_control_chart", index=False)
    return out


def t55c_bottleneck(gen: pd.DataFrame) -> pd.DataFrame:
    """병목 진단 — 신호 크기 대 정상 변동."""
    rows = []
    for c, lab in [(F, "가압력"), (I, "전류")]:
        day = gen.groupby("d")[c].median()
        sta = np.array([v for k, v in day.items() if str(k) in STABLE_DAYS])
        var = np.array([v for k, v in day.items() if str(k) not in STABLE_DAYS])
        s = float(np.std(sta, ddof=1))
        shift = float(abs(np.mean(var) - np.mean(sta)))
        delta = shift / s if s > 0 else np.inf
        power = float(norm.cdf(-3 + delta) + norm.cdf(-3 - delta))
        rows.append({"변수": lab,
                     "안정일 간 표준편차": round(s, 5),
                     "이상일 이동폭": round(shift, 5),
                     "이동폭/정상변동 (delta)": round(delta, 2),
                     "3sigma 관리도 1일 탐지확률": f"{power*100:.1f}%",
                     "sigma 추정 자유도": len(sta) - 1})
    out = pd.DataFrame(rows)
    save_table(out, "t55c_bottleneck", index=False)
    return out


def t56_required_days() -> pd.DataFrame:
    """필요한 참조일 수 — 관리도 Phase I 표준과 검정력 분석의 수렴."""
    t = pd.DataFrame([
        {"근거": "관리도 Phase I 통상 권고", "필요 작업일": "20~25 일",
         "내용": "관리한계를 안정적으로 추정하기 위한 최소 부분군 수"},
        {"근거": "본 데이터의 sigma 추정 자유도", "필요 작업일": "현재 4 (안정일 5일-1)",
         "내용": "자유도 4 의 sigma 추정은 관리한계를 크게 흔든다"},
        {"근거": "검정력 분석 (3.5절)", "필요 작업일": "33 일",
         "내용": "중간 효과(rho=0.5)를 80% 검정력으로 검출"},
        {"근거": "종합 권고", "필요 작업일": "최소 25 일, 권고 33 일",
         "내용": "두 독립적 근거가 20~33일 구간으로 수렴한다"},
    ])
    save_table(t, "t56_required_reference_days", index=False)
    return t


def t57_breakeven(scored: pd.DataFrame) -> pd.DataFrame:
    """재검사의 손익분기 — 검출률을 가정하지 않고 임계만 제시한다.

      f  플래그 비율 / q  전체 불량률 / r  플래그 점의 농축배수
      c_i 재검사 1점 비용 / c_d 불량 유출 1건 비용
    손익분기:  c_i*f = c_d*q*r*f/(r*f + 1 - f)
    """
    sta = scored[scored["d"].astype(str).isin(STABLE_DAYS)]
    f = float((sta["conformal_p"] < ALPHA_FLAG).mean())
    rows = []
    for ratio in [10, 50, 100, 500]:
        for q in [0.001, 0.005, 0.01]:
            a = (1.0 / ratio) * f - q
            r_be = np.inf if a >= 0 else (1.0 / ratio) * (1.0 - f) / (-a)
            rows.append({"불량유출/재검사 비용비": ratio, "전체 불량률 가정": q,
                         "플래그 비율(실측)": round(f, 4),
                         "손익분기 농축배수 r*": "달성 불가" if not np.isfinite(r_be)
                                              else round(float(r_be), 1)})
    out = pd.DataFrame(rows)
    save_table(out, "t57_breakeven", index=False)
    return out


# ══════════════════════════════════════════════════════════════════
def f29(gen: pd.DataFrame):
    day = gen.groupby("d")[F].median()
    sta = {k: v for k, v in day.items() if str(k) in STABLE_DAYS}
    m = float(np.mean(list(sta.values())))
    s = float(np.std(list(sta.values()), ddof=1))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ks = sorted(day.index)
    xs = np.arange(len(ks))
    cols = ["#2e86c1" if str(k) in STABLE_DAYS else "#c0392b" for k in ks]
    ax.scatter(xs, [day[k] for k in ks], s=95, color=cols, zorder=4)
    ax.axhline(m, color="black", lw=1.4, label=f"안정일 중심선 {m:.4f}")
    for L, st, c in [(2.0, ":", "#e67e22"), (3.0, "--", "#7f8c8d")]:
        ax.axhline(m + L * s, ls=st, color=c, lw=1.4)
        ax.axhline(m - L * s, ls=st, color=c, lw=1.4,
                   label=f"{L:g}σ 한계 ({m - L * s:.4f} ~ {m + L * s:.4f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([k.strftime("%m-%d") for k in ks], fontsize=9)
    ax.set_ylabel("일별 가압력 중앙값 (bar)")
    from matplotlib.lines import Line2D
    h, _ = ax.get_legend_handles_labels()
    h += [Line2D([], [], marker="o", lw=0, color="#2e86c1", label="안정일"),
          Line2D([], [], marker="o", lw=0, color="#c0392b", label="공정 이동일")]
    ax.legend(handles=h, fontsize=8.5, loc="upper right")
    ax.set_title("공정 이동(2.310)은 3σ 한계 바로 안쪽에 걸린다 — 참조일 5일로는 판정이 갈린다",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f29_day_control_chart.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f29_day_control_chart.png")


def f30(bott: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    deltas = np.linspace(0, 6, 300)
    for L, c in [(2.0, "#e67e22"), (3.0, "#1f4e79")]:
        pw = norm.cdf(-L + deltas) + norm.cdf(-L - deltas)
        ax.plot(deltas, pw * 100, color=c, lw=2.2, label=f"{L:g}σ 관리한계")
    for _, r in bott.iterrows():
        d = float(r["이동폭/정상변동 (delta)"])
        ax.axvline(d, ls=":", color="#c0392b", lw=1.5)
        ax.annotate(f"{r['변수']} δ={d:.1f}", (d, 8), fontsize=9, color="#c0392b",
                    rotation=90, va="bottom", ha="right")
    ax.axhline(80, ls="--", color="gray", lw=1)
    ax.text(5.9, 82, "통상 목표 80%", fontsize=8.5, ha="right", color="gray")
    ax.set_xlabel("이동폭 / 정상 일간변동 (δ)")
    ax.set_ylabel("1일 관측으로 탐지할 확률 (%)")
    ax.legend(fontsize=9)
    ax.set_title("관측된 공정 이동은 3σ 관리도의 탐지확률이 낮은 구간에 있다", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f30_signal_to_noise.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f30_signal_to_noise.png")


def main():
    print("\n[17] 운영 효익과 그 한계")
    scored = load_scores()
    gen = load_genuine()
    print(f"  대상: 실제 생산점 {len(gen):,} 점, {gen['d'].nunique()} 작업일")
    print("  (복제 구간 3,200점 제외. 2020-04-03 은 전체가 복제 구간이라 제외됨)")

    print("\n══ (1) 용접점 단위 순차 감시 — 작동점 스윕 ══")
    sw = t55_sequential(scored)
    print(sw.to_string(index=False))
    print("\n   15개 작동점 전부에서 '오경보 0 + 탐지 1회 이상'을 만족하지 못한다.")

    print("\n══ (2) 일 단위 요약통계 관리도 — 더 단순한 대안 ══")
    dc = t55b_daychart(gen)
    print(dc.to_string(index=False))
    ok = dc[dc["유효"] == "예"]
    print(f"\n   유효 조합: {len(ok)}/{len(dc)}")
    best = dc[(dc["이상일 탐지"] == f"3/3") & (dc["안정일 오경보"].str.startswith("0"))]
    if not best.empty:
        b = best.iloc[0]
        print(f"   >> 전량 탐지: {b['변수']} {b['통계량']} + {b['관리한계']} 관리한계 "
              f"(탐지 {b['이상일 탐지']}, 오경보 {b['안정일 오경보']})")
        print("   정교한 용접점 단위 탐지기는 0/3 인데, 단순한 일 단위 관리도가 3/3 이다.")
        print("   두 방법은 겨루는 관계가 아니라 **다른 문제를 푸는 도구**다.")
        print("     일 단위 관리도 — 작업일 전체의 수준 이동 (본 데이터의 실제 이상)")
        print("     용접점 탐지기  — 개별 점의 이탈, 재검사 우선순위 (합성 이상에서 검증)")

    print("\n══ (3) 병목 진단 — 탐지기인가 데이터인가 ══")
    bott = t55c_bottleneck(gen)
    print(bott.to_string(index=False))
    print("\n   병목은 탐지기가 아니라 참조 데이터의 양이다.")
    print("   이동폭 자체는 작지 않으나, 비교 기준인 '정상 일간변동'을 자유도 4로만")
    print("   추정하고 있어 관리한계가 불안정하다.")

    print("\n══ (4) 필요한 참조일 수 ══")
    print(t56_required_days().to_string(index=False))

    print("\n══ (5) 재검사 손익분기 — 농축배수 r* ══")
    print(t57_breakeven(scored).to_string(index=False))
    print("\n   r* 는 '플래그된 점이 정상 점보다 몇 배 불량해야 재검사가 이득인가'다.")
    print("   현장이 자기 비용비와 전체 불량률을 넣으면 바로 판단할 수 있다.")

    f29(gen)
    f30(bott)

    concl = pd.DataFrame([
        {"항목": "순차 감시 탐지", "값": "0/3 (15개 작동점 전부)"},
        {"항목": "일 단위 관리도 탐지", "값": f"{len(ok)}/{len(dc)} 조합에서 유효"},
        {"항목": "공정 이동폭",
         "값": f"가압력 {bott.iloc[0]['이동폭/정상변동 (delta)']:.1f} x 정상 일간변동"},
        {"항목": "유효한 일 단위 경보",
         "값": "가압력 일별 중앙값 + 2 sigma 관리한계 → 탐지 3/3, 오경보 0/5"},
        {"항목": "3 sigma 에서 실패한 이유",
         "값": "delta=2.92 이므로 3 sigma 탐지확률 46.9%, 2 sigma 는 약 82%"},
        {"항목": "병목", "값": "sigma 를 자유도 4(안정일 5일)로만 추정 — 2/3 sigma 선택이 갈린다"},
        {"항목": "필요 참조일", "값": "최소 25일, 권고 33일"},
        {"항목": "현 시점 권고",
         "값": "일 단위는 가압력 중앙값 2 sigma 관리도로 즉시 운용 가능. "
               "용접점 단위 탐지기는 재검사 우선순위로 병행"},
        {"항목": "초판 대비 정정",
         "값": "복제 구간을 부분적으로만 제외했을 때의 '4/4 탐지, 40.4% 보호'는 "
               "아티팩트 탐지였다"},
    ])
    save_table(concl, "t58_operational_conclusion", index=False)
    print("\n-- 결론")
    print(concl.to_string(index=False))
    print("\n[17] 완료")


if __name__ == "__main__":
    main()
