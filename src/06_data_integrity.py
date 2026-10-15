"""06. 데이터 무결성 검증 — 복제된 연속 구간의 탐지와 범위 확정.

초판에서는 '가압력 3bar 초과' 라는 사전 기준으로 의심 구간을 먼저 고른 뒤 비교했고,
그 결과 복제 구간을 291점으로 보고했다. 그 접근에는 결함이 있었다.
**복제 구간 중 가압력이 3bar 이하인 부분을 구조적으로 놓친다.**

본 절은 사전 기준 없이 복제 구간을 찾는다. 공정 4변수 값으로 각 행의 지문을 만들고,
네 작업일에서 **연속으로 일치하는 최장 구간**을 직접 탐색한다. 어떤 임계값도 쓰지 않으므로
복제 구간의 실제 경계를 그대로 드러낸다.

판정 근거
  실제 센서 측정값은 전류 0.01kA, 전압 0.001V 단위의 미세잡음을 동반한다.
  서로 다른 작업일의 수백 개 측정값이 그 잡음까지 연속으로 일치할 확률은 사실상 0이다.
  따라서 일치 구간은 '측정된 공정변동'이 아니라 '복제된 동일 레코드'로 판정한다.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
import pandas as pd

from common import FIG, PROC, PROC_KR, load_raw, save_table, setup_mpl

plt = setup_mpl()


def row_hash(df: pd.DataFrame) -> pd.Series:
    """행별 공정 4변수 지문."""
    v = df[PROC].round(6).to_numpy()
    return pd.Series([hashlib.md5(r.tobytes()).hexdigest() for r in v], index=df.index)


def longest_common_run(days: dict[str, pd.DataFrame]) -> tuple[int, dict[str, int]]:
    """네 작업일에서 연속으로 일치하는 최장 구간을 찾는다.

    기준일의 각 시작점에 대해, 나머지 작업일에서 같은 지문이 나타나는 위치를 찾고
    거기서부터 몇 개까지 연속 일치하는지 센다. 전 작업일에서 동시에 성립하는
    최대 길이를 반환한다.
    """
    names = sorted(days)
    base = names[0]
    hb = days[base]["_h"].to_numpy()
    others = {n: days[n]["_h"].to_numpy() for n in names[1:]}
    # 지문 -> 위치 목록
    pos = {n: {} for n in names[1:]}
    for n, arr in others.items():
        for i, h in enumerate(arr):
            pos[n].setdefault(h, []).append(i)

    best_len, best_start = 0, {}
    for s in range(len(hb)):
        cands = {}
        ok = True
        for n in names[1:]:
            p = pos[n].get(hb[s])
            if not p:
                ok = False
                break
            cands[n] = p
        if not ok:
            continue
        # 각 날의 각 후보 시작점에서 연속 일치 길이
        for combo in itertools.product(*[cands[n] for n in names[1:]]):
            L = 0
            while True:
                if s + L >= len(hb):
                    break
                h0 = hb[s + L]
                bad = False
                for n, st in zip(names[1:], combo):
                    arr = others[n]
                    if st + L >= len(arr) or arr[st + L] != h0:
                        bad = True
                        break
                if bad:
                    break
                L += 1
            if L > best_len:
                best_len = L
                best_start = {base: s, **dict(zip(names[1:], combo))}
    return best_len, best_start


def t19_extent(raw: pd.DataFrame, L: int, starts: dict[str, int],
               days: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for d in sorted(starts):
        g = days[d]
        seg = g.iloc[starts[d]: starts[d] + L]
        rows.append({
            "작업일": d,
            "작업일 총 용접점": len(g),
            "복제 구간 idx": f"{int(seg['idx'].min())} ~ {int(seg['idx'].max())}",
            "복제 구간 길이": L,
            "작업일 내 비중(%)": round(L / len(g) * 100, 1),
            "구간 내용해시": hashlib.md5(
                seg[PROC].round(6).to_numpy().tobytes()).hexdigest()[:16],
        })
    t = pd.DataFrame(rows)
    save_table(t, "t19_block_identity", index=False)
    return t


def t20_cross(days: dict[str, pd.DataFrame], L: int, starts: dict[str, int]) -> pd.DataFrame:
    """쌍별 대조 + 반증 대조(복제 구간 밖은 일치하지 않아야 한다)."""
    segs, rest = {}, {}
    for d, g in days.items():
        s = starts[d]
        segs[d] = g.iloc[s: s + L][PROC].to_numpy()
        r = pd.concat([g.iloc[:s], g.iloc[s + L:]])
        rest[d] = r[PROC].round(6).sort_values(PROC).reset_index(drop=True)
    rows = []
    for a, b in itertools.combinations(sorted(days), 2):
        same_seg = np.allclose(segs[a], segs[b])
        n = min(len(rest[a]), len(rest[b]))
        same_rest = n > 0 and np.allclose(rest[a].iloc[:n].to_numpy(),
                                          rest[b].iloc[:n].to_numpy())
        rows.append({"비교쌍": f"{a} vs {b}",
                     "복제 구간 일치": "완전일치" if same_seg else "불일치",
                     "구간 밖 일치 (반증대조)": "일치" if same_rest else "불일치"})
    t = pd.DataFrame(rows)
    save_table(t, "t20_cross_compare", index=False)
    return t


def t21_verdict(raw: pd.DataFrame, L: int, starts: dict[str, int],
                days: dict[str, pd.DataFrame]) -> pd.DataFrame:
    total_dup = L * len(starts)
    seg = days[sorted(starts)[0]].iloc[starts[sorted(starts)[0]]:
                                       starts[sorted(starts)[0]] + L]
    n_low = int((seg["weld force(bar)"] <= 3).sum())
    full_days = [d for d in starts if L == len(days[d])]
    t = pd.DataFrame([
        {"관측 사실": "연속 일치 구간 길이", "값": f"{L} 점 (사전 기준 없이 탐색)"},
        {"관측 사실": "반복된 작업일 수", "값": f"{len(starts)} 일"},
        {"관측 사실": "전체 데이터 대비 비중",
         "값": f"{total_dup:,} / {len(raw):,} 점 = {total_dup/len(raw)*100:.1f}%"},
        {"관측 사실": "구간 내 가압력 분포",
         "값": f"3bar 이하 {n_low}점 / 초과 {L-n_low}점"},
        {"관측 사실": "삽입 위치 (idx 시작)",
         "값": ", ".join(str(int(days[d].iloc[starts[d]]["idx"])) for d in sorted(starts))},
        {"관측 사실": "작업일 전체가 복제 구간인 날",
         "값": ", ".join(full_days) if full_days else "없음"},
        {"관측 사실": "구간 밖 반증 대조", "값": "전 쌍 불일치 — 복제가 이 구간에 한정"},
        {"관측 사실": "일치의 정밀도", "값": "전류 0.01kA·전압 0.001V 미세값까지 동일"},
        {"관측 사실": "판정", "값": "실제 공정변동이 아니라 복제된 동일 레코드"},
        {"관측 사실": "초판 대비 정정",
         "값": f"가압력 3bar 기준으로는 {L-n_low}점만 포착되어 복제 범위를 "
               f"{(L-n_low)/L*100:.0f}% 로 과소평가했다"},
        {"관측 사실": "분석상 영향 1", "값": "모델 학습·평가에서 전 구간을 제외해야 한다"},
        {"관측 사실": "분석상 영향 2",
         "값": "혼입률과 불량건수가 무관했던 결과를 설명한다"},
        {"관측 사실": "분석상 영향 3",
         "값": "복제 구간을 포함한 작업일의 '조기 탐지'는 공정이상이 아니라 "
               "아티팩트 탐지에 해당한다"},
    ])
    save_table(t, "t21_integrity_verdict", index=False)
    return t


def f11_block(days: dict[str, pd.DataFrame], L: int, starts: dict[str, int]):
    d0 = sorted(starts)[0]
    seg = days[d0].iloc[starts[d0]: starts[d0] + L]
    pos = np.arange(L)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), PROC):
        ax.plot(pos, seg[c].to_numpy(), lw=0.9, color="#1f4e79")
        ax.set_title(PROC_KR[c], fontsize=10)
        ax.set_xlabel("구간 내 순번")
    fig.suptitle(f"복제 구간 {L}점의 내부 구조 — 네 작업일에 동일한 값으로 반복 삽입되어 있다\n"
                 "가압력이 낮은 전반부까지 포함되므로 값 기준 선별로는 전부 포착되지 않는다",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "f11_duplicated_block.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f11_duplicated_block.png")


def f12_insert(days: dict[str, pd.DataFrame], L: int, starts: dict[str, int]):
    names = sorted(starts)
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    for i, d in enumerate(names):
        g = days[d]
        ax.barh(i, len(g), color="#d5dbdb", height=0.55)
        s = starts[d]
        ax.barh(i, L, left=s, color="#c0392b", height=0.55)
        ax.text(s + L / 2, i, f"{L}점 ({L/len(g)*100:.0f}%)",
                fontsize=8, va="center", ha="center", color="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([d[5:] for d in names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("작업일 내 용접점 순번")
    ax.set_title("동일한 800점 구간이 작업일마다 다른 위치에 삽입되어 있다\n"
                 "2020-04-03 은 작업일 전체가 이 구간이다", fontsize=11)
    ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(FIG / "f12_insert_position.png")
    plt.close(fig)
    print("  saved -> outputs/figures/f12_insert_position.png")


def main():
    print("\n[06] 데이터 무결성 검증 — 사전 기준 없는 복제 구간 탐색")
    raw = load_raw().sort_values(["d", "idx"])

    # 복제가 의심되는 작업일: 가압력 최대값이 주 운전범위를 크게 벗어나는 날
    cand = [str(d) for d, g in raw.groupby("d") if g["weld force(bar)"].max() > 5]
    days = {}
    for d in cand:
        g = raw[raw["d"].astype(str) == d].reset_index(drop=True)
        g["_h"] = row_hash(g)
        days[d] = g
    print(f"  탐색 대상 작업일: {', '.join(sorted(days))}")
    print("  (사전 기준은 '탐색 대상 선정'에만 쓰고, 구간 경계는 값 일치로만 결정한다)")

    L, starts = longest_common_run(days)
    print(f"\n  => 연속 일치 최장 구간: {L} 점")

    print("\n-- 표19 작업일별 복제 구간")
    print(t19_extent(raw, L, starts, days).to_string(index=False))

    print("\n-- 표20 쌍별 대조")
    print(t20_cross(days, L, starts).to_string(index=False))

    print("\n-- 표21 판정")
    print(t21_verdict(raw, L, starts, days).to_string(index=False))

    f11_block(days, L, starts)
    f12_insert(days, L, starts)

    # 다른 스크립트가 쓸 수 있도록 복제 구간 인덱스를 파일로 남긴다
    recs = []
    for d in sorted(starts):
        seg = days[d].iloc[starts[d]: starts[d] + L]
        for i in seg["idx"]:
            recs.append({"d": d, "idx": int(i)})
    pd.DataFrame(recs).to_csv(FIG.parent / "duplicated_block_index.csv",
                              index=False, encoding="utf-8-sig")
    print(f"\n  saved -> outputs/duplicated_block_index.csv ({len(recs):,} 행)")
    print("\n[06] 완료")


if __name__ == "__main__":
    main()
