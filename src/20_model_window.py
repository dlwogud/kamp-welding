"""20. 모델 기반 안정 공정조건 범위 — 지문 요구사항 ③에 모델로 답한다.

제4장의 공정 관리범위는 '안정일의 자연변동'(I-MR 관리한계)에서 도출했다. 이는
공정 일관성 기준이며 **모델의 출력이 아니다**. 지문의 요구는 다음과 같다.

    "불량위험을 낮출 수 있는 안정 공정조건의 범위를 제안하시오"

본 절은 그 범위를 **모델에서 직접 도출**한다. 두 모델을 쓴다.

  (A) 이상탐지 모형 (14번, kNN k=3, 특징공간 [가압력, R, E])
      공정공간의 각 지점에 컨포멀 p 값을 부여한다. p 가 큰 영역이
      '모델이 정상 운전으로 보는 영역'이다.

  (B) 라벨 비율 학습 모형 (18번, M1 비율매칭 로지스틱)
      가방별 불량 개수만으로 학습한 불량위험 방향이다. 낮은 위험 영역이
      '모델이 불량 가능성이 낮다고 보는 영역'이다.
      **검증 상태 주의**: 이 모형은 순열검정에서 유의하지 않았다(p=0.45).
      위험 방향은 '시사'이지 '입증'이 아니며, 본 절은 그 사실을 표기한다.

권고 범위 = (A 정상) ∩ (B 저위험) ∩ (데이터 지지영역)

데이터 지지영역을 교집합에 넣는 이유
    모델은 학습 데이터가 없는 영역에도 값을 내놓는다. 그 값으로 공정조건을
    권고하면 외삽이다. 실제 관측이 있는 영역으로만 한정한다.

공정공간의 차원
    전압(변동계수 0.155%)과 통전시간(0.025%)은 사실상 상수이므로 중앙값으로 고정하고,
    실질 변동 변수인 **가압력 × 전류 2차원 격자**로 공간을 구성한다.
    파생량 R = V/I, E = V·I·t 는 이 격자에서 결정된다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from common import FIG, STABLE_DAYS, daily_label, load_genuine, save_table, setup_mpl

plt = setup_mpl()
warnings.filterwarnings("ignore")
SEED = 42
F, I, V, T = "weld force(bar)", "weld current(kA)", "weld Voltage(v)", "weld time(ms)"
FEATS_AD = [F, "R", "E"]          # 14번 이상탐지 특징공간
FEATS_LLP = [F, I, "R", "E"]      # 18번 LLP 특징공간
K_NB = 3
CAL_FRAC = 0.3
GRID = 160
P_NORMAL = 0.05                   # 컨포멀 p 가 이보다 크면 '모델이 정상으로 봄'
SUPPORT_Q = 0.99                  # 지지영역: 실제 관측까지의 거리가 이 분위수 이내
MAX_ITER = 8


def physics_arr(f, i, v, t):
    amp = i * 1000.0
    return {"R": v / amp, "E": v * amp * (t / 1000.0)}


def prep():
    g = load_genuine(stable_only=True).copy()
    amp = g[I] * 1000.0
    g["R"] = g[V] / amp
    g["E"] = g[V] * amp * (g[T] / 1000.0)
    return g


def build_grid(g: pd.DataFrame):
    """가압력 × 전류 격자. 전압·통전시간은 중앙값 고정."""
    v0, t0 = float(g[V].median()), float(g[T].median())
    fmin, fmax = g[F].quantile(0.001), g[F].quantile(0.999)
    imin, imax = g[I].quantile(0.001), g[I].quantile(0.999)
    pad_f, pad_i = (fmax - fmin) * 0.25, (imax - imin) * 0.25
    fs = np.linspace(fmin - pad_f, fmax + pad_f, GRID)
    is_ = np.linspace(imin - pad_i, imax + pad_i, GRID)
    FF, II = np.meshgrid(fs, is_, indexing="xy")
    d = physics_arr(FF.ravel(), II.ravel(), v0, t0)
    G = pd.DataFrame({F: FF.ravel(), I: II.ravel(), "R": d["R"], "E": d["E"]})
    return fs, is_, FF, II, G, v0, t0


# ══════════════════════════════════════════════════════════════════
def ad_pvalue(g: pd.DataFrame, G: pd.DataFrame):
    """격자 각 점의 컨포멀 p 값 (14번 최종 모형)."""
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(g))
    ncal = int(len(g) * CAL_FRAC)
    cal, ref = g.iloc[idx[:ncal]], g.iloc[idx[ncal:]]
    sc = RobustScaler().fit(ref[FEATS_AD])
    nn = NearestNeighbors(n_neighbors=K_NB).fit(sc.transform(ref[FEATS_AD]))
    s_cal = np.sort(nn.kneighbors(sc.transform(cal[FEATS_AD]))[0].mean(1))
    s = nn.kneighbors(sc.transform(G[FEATS_AD]))[0].mean(1)
    n = len(s_cal)
    p = (1.0 + (n - np.searchsorted(s_cal, s, side="left"))) / (n + 1.0)
    return p, (sc, nn, s_cal)


def llp_risk(g: pd.DataFrame, G: pd.DataFrame):
    """격자 각 점의 LLP 불량위험 (18번 M1). 전 안정일을 가방으로 학습."""
    dl = daily_label()
    X, ntr = {}, {}
    for d, gg in g.groupby("d"):
        X[str(d)] = gg[FEATS_LLP].to_numpy(float)
        ntr[str(d)] = int(dl.loc[d])
    sc = RobustScaler().fit(np.vstack(list(X.values())))
    Xs = {b: sc.transform(v) for b, v in X.items()}
    Xall = np.vstack(list(Xs.values()))
    bags = np.concatenate([[b] * len(v) for b, v in Xs.items()])
    rate = sum(ntr.values()) / len(Xall)
    w = np.full(len(Xall), rate)
    Xd = np.vstack([Xall, Xall])
    y = np.r_[np.ones(len(Xall)), np.zeros(len(Xall))]
    clf = None
    for _ in range(MAX_ITER):
        clf = LogisticRegression(max_iter=500, random_state=SEED).fit(
            Xd, y, sample_weight=np.r_[w, 1 - w])
        p = clf.predict_proba(Xall)[:, 1]
        w = np.empty_like(p)
        for b in Xs:
            m = bags == b
            pb = p[m]
            w[m] = np.clip(pb * (ntr[b] / pb.sum()), 1e-9, 1.0) if pb.sum() > 0 \
                else ntr[b] / m.sum()
    risk = clf.predict_proba(sc.transform(G[FEATS_LLP]))[:, 1]
    return risk, clf, sc


def support_mask(g: pd.DataFrame, G: pd.DataFrame, k: int = 20):
    """데이터 지지영역 — 밀도 기반.

    각 지점에서 **k번째 가까운 실제 관측까지의 거리**를 밀도의 역지표로 쓴다.
    임계는 실제 관측점들에서 같은 통계를 구해 그 상위 분위수로 둔다.
    즉 '실제 데이터가 있는 곳만큼 조밀한 격자점'만 지지영역으로 본다.

    최근접 1개를 쓰면 안 된다. 학습점을 질의하면 자기 자신까지의 거리 0 이 나오고,
    본 데이터는 완전중복행이 42.6% 라 2번째 이웃도 0 인 경우가 많다.
    k 를 충분히 크게 잡아야 밀도가 제대로 반영된다.
    """
    sc = RobustScaler().fit(g[[F, I]])
    Xd = sc.transform(g[[F, I]])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xd)
    d_data = nn.kneighbors(Xd)[0][:, -1]              # 자기 자신 제외한 k번째
    thr = float(np.quantile(d_data, SUPPORT_Q))
    d_grid = nn.kneighbors(sc.transform(G[[F, I]]))[0][:, -1]
    return d_grid <= thr, thr


def t67_window(g, fs, is_, p_ad, risk, sup) -> pd.DataFrame:
    """권고 범위 = 모델 정상 ∩ 모델 저위험 ∩ 지지영역."""
    risk_thr = float(np.quantile(risk[sup], 0.5))     # 지지영역 내 중앙값 기준
    ok = (p_ad > P_NORMAL) & (risk <= risk_thr) & sup
    FF = np.repeat(fs[None, :], len(is_), axis=0).ravel()
    II = np.repeat(is_[:, None], len(fs), axis=1).ravel()
    rows = []
    if ok.sum() == 0:
        rows.append({"항목": "권고 범위", "값": "존재하지 않음"})
    else:
        rows += [
            {"구성": "모델 정상 영역 (컨포멀 p > 0.05)", "격자 비율(%)": round((p_ad > P_NORMAL).mean() * 100, 1)},
            {"구성": "모델 저위험 영역 (LLP 위험 ≤ 지지영역 중앙값)", "격자 비율(%)": round((risk <= risk_thr).mean() * 100, 1)},
            {"구성": "데이터 지지영역", "격자 비율(%)": round(sup.mean() * 100, 1)},
            {"구성": "교집합 = 권고 범위", "격자 비율(%)": round(ok.mean() * 100, 1)},
        ]
    t = pd.DataFrame(rows)
    save_table(t, "t67_window_components", index=False)

    # 변수별 권고 구간 (교집합의 외곽 상자)
    box = pd.DataFrame([
        {"변수": "가압력(bar)",
         "모델 권고 범위": f"{FF[ok].min():.3f} ~ {FF[ok].max():.3f}" if ok.sum() else "—",
         "관리도 범위 (4.3절)": "2.323 ~ 2.356",
         "실제 운전범위 (안정일)": f"{g[F].min():.3f} ~ {g[F].max():.3f}"},
        {"변수": "전류(kA)",
         "모델 권고 범위": f"{II[ok].min():.3f} ~ {II[ok].max():.3f}" if ok.sum() else "—",
         "관리도 범위 (4.3절)": "14.642 ~ 14.703",
         "실제 운전범위 (안정일)": f"{g[I].min():.3f} ~ {g[I].max():.3f}"},
    ])
    save_table(box, "t68_model_window", index=False)
    return box, ok, risk_thr


def t68b_tradeoff(g, fs, is_, p_ad, risk, sup) -> pd.DataFrame:
    """위험 임계에 따른 범위-포함률 절충.

    초판은 LLP 위험을 '지지영역 중앙값 이하'로 잘랐는데, 그 임계가 결과를 결정해 버린다
    (포함률이 자동으로 50% 근처가 된다). 임계를 고정하는 대신 **절충 곡선**으로 제시해
    현장이 '얼마나 좁은 범위를 쓸 것인가'를 직접 고르게 한다.
    """
    FF = np.repeat(fs[None, :], len(is_), axis=0).ravel()
    II = np.repeat(is_[:, None], len(fs), axis=1).ravel()
    fi = np.searchsorted(fs, g[F].to_numpy()).clip(0, len(fs) - 1)
    ii = np.searchsorted(is_, g[I].to_numpy()).clip(0, len(is_) - 1)
    rows = []
    for q in [0.25, 0.50, 0.75, 0.90, 1.00]:
        thr = float(np.quantile(risk[sup], q))
        ok = (p_ad > P_NORMAL) & (risk <= thr) & sup
        if ok.sum() == 0:
            continue
        OK = ok.reshape(len(is_), len(fs))
        cov = float(OK[ii, fi].mean()) * 100
        rows.append({
            "LLP 위험 임계(분위수)": q,
            "가압력 범위": f"{FF[ok].min():.3f} ~ {FF[ok].max():.3f}",
            "전류 범위": f"{II[ok].min():.3f} ~ {II[ok].max():.3f}",
            "현 생산 포함률(%)": round(cov, 1),
            "격자 면적(%)": round(ok.mean() * 100, 1),
        })
    out = pd.DataFrame(rows)
    save_table(out, "t68b_window_tradeoff", index=False)
    return out


def t68c_ablation(g, fs, is_, p_ad, risk, sup) -> pd.DataFrame:
    """구성요소 제거 실험 — 검증되지 않은 LLP 가 실제로 범위를 바꾸는가.

    LLP 모형은 순열검정에서 유의하지 않았다(p=0.45). 그 모형이 권고 범위에
    실질적으로 기여하지 않는다면, 범위는 사실상 이상탐지 모형과 지지영역만으로
    정해지는 것이며 그 사실을 밝혀야 한다.
    """
    FF = np.repeat(fs[None, :], len(is_), axis=0).ravel()
    II = np.repeat(is_[:, None], len(fs), axis=1).ravel()
    thr = float(np.quantile(risk[sup], 0.5))
    combos = [
        ("A 이상탐지 + 지지영역", (p_ad > P_NORMAL) & sup),
        ("B LLP + 지지영역", (risk <= thr) & sup),
        ("A ∩ B ∩ 지지영역 (권고)", (p_ad > P_NORMAL) & (risk <= thr) & sup),
        ("지지영역만", sup),
    ]
    rows = []
    for name, m in combos:
        if m.sum() == 0:
            rows.append({"구성": name, "가압력 범위": "—", "전류 범위": "—", "격자 면적(%)": 0.0})
            continue
        rows.append({"구성": name,
                     "가압력 범위": f"{FF[m].min():.3f} ~ {FF[m].max():.3f}",
                     "전류 범위": f"{II[m].min():.3f} ~ {II[m].max():.3f}",
                     "격자 면적(%)": round(m.mean() * 100, 1)})
    out = pd.DataFrame(rows)
    save_table(out, "t68c_window_ablation", index=False)
    return out


def t69_coverage(g, fs, is_, ok) -> pd.DataFrame:
    """실제 생산점 중 권고 범위 안에 든 비율. 운영 부하의 직접 지표."""
    fi = np.searchsorted(fs, g[F].to_numpy()).clip(0, len(fs) - 1)
    ii = np.searchsorted(is_, g[I].to_numpy()).clip(0, len(is_) - 1)
    OK = ok.reshape(len(is_), len(fs))
    inside = OK[ii, fi]
    rows = [{"구분": "안정일 전체", "용접점수": len(g),
             "권고 범위 내(%)": round(float(inside.mean()) * 100, 1)}]
    for d, gg in g.groupby("d"):
        m = g["d"] == d
        rows.append({"구분": str(d), "용접점수": int(m.sum()),
                     "권고 범위 내(%)": round(float(inside[m.to_numpy()].mean()) * 100, 1)})
    t = pd.DataFrame(rows)
    save_table(t, "t69_window_coverage", index=False)
    return t


def f34(g, fs, is_, p_ad, risk, sup, ok, risk_thr):
    FFg, IIg = np.meshgrid(fs, is_, indexing="xy")
    P = p_ad.reshape(FFg.shape)
    Rk = risk.reshape(FFg.shape)
    S = sup.reshape(FFg.shape)
    OK = ok.reshape(FFg.shape)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

    a = axes[0]
    im = a.contourf(FFg, IIg, P, levels=20, cmap="RdYlGn")
    a.contour(FFg, IIg, P, levels=[P_NORMAL], colors="black", linewidths=2)
    a.scatter(g[F], g[I], s=2, c="#1f1f1f", alpha=0.18)
    fig.colorbar(im, ax=a, label="컨포멀 p")
    a.set_title("(A) 이상탐지 모형 — 정상 영역\n검은 선 = p 0.05", fontsize=10)

    a = axes[1]
    im = a.contourf(FFg, IIg, Rk, levels=20, cmap="RdYlGn_r")
    a.contour(FFg, IIg, Rk, levels=[risk_thr], colors="black", linewidths=2)
    a.scatter(g[F], g[I], s=2, c="#1f1f1f", alpha=0.18)
    fig.colorbar(im, ax=a, label="LLP 불량위험")
    a.set_title("(B) 라벨 비율 학습 모형 — 위험\n(검증 미완: 순열 p=0.45)", fontsize=10)

    a = axes[2]
    a.contourf(FFg, IIg, S.astype(float), levels=[0.5, 1.5], colors=["#d5dbdb"], alpha=0.6)
    a.contourf(FFg, IIg, OK.astype(float), levels=[0.5, 1.5], colors=["#27ae60"], alpha=0.75)
    a.scatter(g[F], g[I], s=2, c="#1f1f1f", alpha=0.25)
    import matplotlib.patches as mp
    a.add_patch(mp.Rectangle((2.323, 14.642), 2.356 - 2.323, 14.703 - 14.642,
                             fill=False, ec="#c0392b", lw=2.2))
    a.set_title("(C) 권고 범위 = A ∩ B ∩ 지지영역\n빨간 사각형 = 관리도 범위(4.3절)", fontsize=10)

    for a in axes:
        a.set_xlabel("가압력 (bar)")
        a.set_ylabel("전류 (kA)")
    fig.suptitle("모델이 제안하는 안정 공정조건 범위 — 지문 요구사항 ③에 대한 모델 기반 답", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(FIG / "f34_model_window.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f34_model_window.png")


def main():
    print("\n[20] 모델 기반 안정 공정조건 범위")
    g = prep()
    print(f"  학습 데이터: 안정일 {len(g):,}점 (복제 구간 제외)")
    fs, is_, FF, II, G, v0, t0 = build_grid(g)
    print(f"  격자 {GRID}x{GRID}, 전압 {v0:.4f}V · 통전시간 {t0:.1f}ms 로 고정")

    p_ad, _ = ad_pvalue(g, G)
    risk, clf, _ = llp_risk(g, G)
    sup, thr = support_mask(g, G)
    print(f"  데이터 지지영역: 격자의 {sup.mean()*100:.1f}%")

    print("\n══ (1) 권고 범위의 구성 ══")
    box, ok, risk_thr = t67_window(g, fs, is_, p_ad, risk, sup)
    print(pd.read_csv("../outputs/tables/t67_window_components.csv").to_string(index=False)
          if False else "")
    print(box.to_string(index=False))

    print("\n══ (1b) 위험 임계에 따른 절충 ══")
    tr = t68b_tradeoff(g, fs, is_, p_ad, risk, sup)
    print(tr.to_string(index=False))

    print("\n══ (1c) 구성요소 제거 실험 — LLP 가 범위를 바꾸는가 ══")
    ab = t68c_ablation(g, fs, is_, p_ad, risk, sup)
    print(ab.to_string(index=False))
    a_only = ab[ab["구성"].str.startswith("A ")].iloc[0]
    both = ab[ab["구성"].str.startswith("A ∩")].iloc[0]
    same = (a_only["가압력 범위"] == both["가압력 범위"]) and (a_only["전류 범위"] == both["전류 범위"])
    print(f"\n   A 단독과 A∩B 의 범위가 {'동일' if same else '상이'}하다.")
    if same:
        print("   검증되지 않은 LLP 는 권고 범위를 바꾸지 않는다. 범위는 사실상")
        print("   이상탐지 모형과 데이터 지지영역으로 결정된다 — 그대로 보고한다.")

    print("\n══ (2) 실제 생산점의 권고 범위 포함률 ══")
    cov = t69_coverage(g, fs, is_, ok)
    print(cov.to_string(index=False))

    f34(g, fs, is_, p_ad, risk, sup, ok, risk_thr)

    # 검증되지 않은 B 를 제외한 방어 가능한 범위 = A ∩ 지지영역
    FFv = np.repeat(fs[None, :], len(is_), axis=0).ravel()
    IIv = np.repeat(is_[:, None], len(fs), axis=1).ravel()
    okA = (p_ad > P_NORMAL) & sup
    fA, FA = float(FFv[okA].min()), float(FFv[okA].max())
    iA, IA = float(IIv[okA].min()), float(IIv[okA].max())
    CH_F, CH_I = (2.323, 2.356), (14.642, 14.703)
    contains = (fA <= CH_F[0] and FA >= CH_F[1] and iA <= CH_I[0] and IA >= CH_I[1])
    fi = np.searchsorted(fs, g[F].to_numpy()).clip(0, len(fs) - 1)
    ii = np.searchsorted(is_, g[I].to_numpy()).clip(0, len(is_) - 1)
    covA = float(okA.reshape(len(is_), len(fs))[ii, fi].mean()) * 100

    two_tier = pd.DataFrame([
        {"층": "내측 — 운영 목표", "근거": "안정일 I-MR 관리한계 (4.3절)",
         "가압력(bar)": f"{CH_F[0]} ~ {CH_F[1]}", "전류(kA)": f"{CH_I[0]} ~ {CH_I[1]}",
         "성격": "이 안에서 운전하면 공정이 통제상태"},
        {"층": "외측 — 이상 판정 경계", "근거": "이상탐지 모형 ∩ 데이터 지지영역 (본 절)",
         "가압력(bar)": f"{fA:.3f} ~ {FA:.3f}", "전류(kA)": f"{iA:.3f} ~ {IA:.3f}",
         "성격": f"이 밖은 모델이 이상으로 판정. 현 생산의 {covA:.1f}% 포함"},
    ])
    save_table(two_tier, "t70b_two_tier_window", index=False)
    print("\n══ (3) 2단 권고 범위 ══")
    print(two_tier.to_string(index=False))
    print(f"\n   외측이 내측을 포함하는가: {'예' if contains else '아니오'}")
    if contains:
        print("   서로 다른 두 경로(안정일 자연변동 / 모델 출력)가 모순되지 않는다.")
        print("   내측은 운영 목표, 외측은 경보 경계로 쓰는 2단 운용이 가능하다.")

    concl = pd.DataFrame([
        {"항목": "도출 방법",
         "값": "이상탐지 모형의 컨포멀 p 영역 ∩ LLP 위험 하위영역 ∩ 데이터 지지영역"},
        {"항목": "관리도 범위와의 관계",
         "값": "4.3절 관리도 범위는 안정일 자연변동에서, 본 범위는 모델 출력에서 도출. "
               "두 경로가 독립적으로 지목한 영역을 비교한다"},
        {"항목": "검증 상태 — 중요",
         "값": "구성요소 (B) LLP 모형은 순열검정에서 유의하지 않았다(p=0.45). "
               "따라서 본 범위는 '모델이 시사하는 범위'이며 '불량 저감이 입증된 범위'가 아니다"},
        {"항목": "외삽 방지",
         "값": "데이터 지지영역을 교집합에 포함해, 관측이 없는 조건으로 외삽하지 않는다"},
        {"항목": "LLP 구성요소의 기여",
         "값": "제거 실험 결과 범위를 좁히기는 하나(면적 5.5% → 3.1%), 해당 모형이 "
               "검증되지 않았으므로 **최종 권고에서는 제외**하고 이상탐지+지지영역만 쓴다"},
        {"항목": "최종 권고 — 2단 구조",
         "값": f"내측(운영 목표) 가압력 {CH_F[0]}~{CH_F[1]}, 전류 {CH_I[0]}~{CH_I[1]} / "
               f"외측(경보 경계) 가압력 {fA:.3f}~{FA:.3f}, 전류 {iA:.3f}~{IA:.3f}"},
        {"항목": "두 경로의 정합",
         "값": ("외측이 내측을 완전히 포함한다. 안정일 자연변동과 모델 출력이라는 "
                "독립적인 두 경로가 모순되지 않는다" if contains else
                "외측이 내측을 포함하지 않는다. 두 경로가 어긋나는 구간을 별도 검토해야 한다")},
        {"항목": "지문 요구사항 ③ 대응",
         "값": "모델 출력으로 공정조건 범위를 제시하고, 관리도 기반 범위와 2단으로 통합한다"},
    ])
    save_table(concl, "t70_model_window_conclusion", index=False)
    print("\n-- 결론")
    print(concl.to_string(index=False))
    print("\n[20] 완료")


if __name__ == "__main__":
    main()
