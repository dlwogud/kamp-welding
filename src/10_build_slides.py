"""10. 발표자료 생성 (PPTX).

제출 요건: 발표자료를 PDF 와 PPT 양쪽으로 제출.
본 스크립트는 PPTX 를 생성한다. PDF 는 PowerPoint 또는 한컴에서 내보내기로 변환한다.

구성은 보고서의 논리 순서를 따르되, 발표용으로 '발견 -> 재정의 -> 모델 -> 활용' 흐름을 강조한다.
블라인드 평가 요건에 따라 소속·로고를 넣지 않는다. 팀명/성명만 표지에서 채운다.
"""
from __future__ import annotations

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

from common import FIG, ROOT

OUT = ROOT / "report" / "발표자료.pptx"

NAVY = RGBColor(0x1F, 0x4E, 0x79)
RED = RGBColor(0xC0, 0x39, 0x2B)
GRAY = RGBColor(0x5D, 0x6D, 0x7E)
DARK = RGBColor(0x21, 0x2F, 0x3C)
LIGHT = RGBColor(0xF4, 0xF6, 0xF7)

FONT = "맑은 고딕"
W, H = Inches(13.333), Inches(7.5)  # 16:9


def _set(run, size, bold=False, color=DARK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT


def textbox(slide, x, y, w, h, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].alignment = align
    return tf


def para(tf, text, size, bold=False, color=DARK, space_before=0, first=False, align=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before = Pt(space_before)
    if align is not None:
        p.alignment = align
    r = p.add_run()
    r.text = text
    _set(r, size, bold, color)
    return p


def band(slide, color=NAVY, h=Inches(0.07), y=Inches(1.12)):
    from pptx.enum.shapes import MSO_SHAPE

    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.6), y, Inches(12.133), h)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def head(slide, title, sub=None):
    tf = textbox(slide, Inches(0.6), Inches(0.35), Inches(12.133), Inches(0.8))
    para(tf, title, 28, True, NAVY, first=True)
    band(slide)
    if sub:
        tf2 = textbox(slide, Inches(0.6), Inches(1.26), Inches(12.133), Inches(0.45))
        para(tf2, sub, 14, False, GRAY, first=True)


def pic(slide, name, x, y, w=None, h=None):
    p = FIG / name
    if not p.exists():
        return None
    kw = {}
    if w:
        kw["width"] = w
    if h:
        kw["height"] = h
    return slide.shapes.add_picture(str(p), x, y, **kw)


def kpi(slide, items, y=Inches(5.5), h=Inches(1.35)):
    """하단 KPI 박스들."""
    from pptx.enum.shapes import MSO_SHAPE

    n = len(items)
    total = Inches(12.133)
    gap = Inches(0.18)
    bw = Emu(int((total - gap * (n - 1)) / n))
    x = Inches(0.6)
    for label, value, color in items:
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, bw, h)
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT
        box.line.color.rgb = color
        box.line.width = Pt(1.5)
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_top = Pt(8)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = value
        _set(r, 22, True, color)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = label
        _set(r2, 11, False, GRAY)
        x = Emu(x + bw + gap)


# ══════════════════════════════════════════════════════════════════
def s01_title(prs):
    s = blank(prs)
    from pptx.enum.shapes import MSO_SHAPE

    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), W, Inches(2.6))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False

    tf = textbox(s, Inches(0.9), Inches(0.75), Inches(11.5), Inches(1.6))
    para(tf, "공정 결합관계 학습 기반 로봇용접 이상탐지", 34, True, RGBColor(0xFF, 0xFF, 0xFF), first=True)
    para(tf, "라벨 해상도 제약 하의 검증 설계", 20, False, RGBColor(0xD6, 0xE4, 0xF0), space_before=8)

    tf2 = textbox(s, Inches(0.9), Inches(3.1), Inches(11.5), Inches(2.2))
    para(tf2, "제6회 K-인공지능 제조데이터 분석 경진대회", 16, True, DARK, first=True)
    para(tf2, "② 용접기 AI 데이터셋 — 용접불량 예측 및 안정 공정조건 분석", 14, False, GRAY, space_before=6)
    para(tf2, "", 12, space_before=18)
    para(tf2, "팀명 :", 14, False, DARK, space_before=6)
    para(tf2, "발표자 :", 14, False, DARK, space_before=4)

    kpi(s, [
        ("용접점 (설명변수)", "11,939", NAVY),
        ("불량 라벨 (작업일)", "8", RED),
        ("합성이상 민감도", "0.971", NAVY),
        ("외부 실제라벨 검증", "0.899", NAVY),
    ], y=Inches(5.75), h=Inches(1.25))


def s02_problem(prs):
    s = blank(prs)
    head(s, "출제 요구사항과 데이터", "지문은 세 가지를 요구한다")
    tf = textbox(s, Inches(0.8), Inches(1.95), Inches(5.6), Inches(4.2))
    para(tf, "① 최종 품질검사 이전에", 17, True, NAVY, first=True)
    para(tf, "    용접불량 여부를 예측하는 모델", 15, False, DARK)
    para(tf, "② 주요 용접조건과", 17, True, NAVY, space_before=16)
    para(tf, "    모델의 실패조건 분석", 15, False, DARK)
    para(tf, "③ 불량위험을 낮추는", 17, True, NAVY, space_before=16)
    para(tf, "    안정 공정조건 범위 제안", 15, False, DARK)

    tf2 = textbox(s, Inches(6.9), Inches(1.95), Inches(5.8), Inches(4.2))
    para(tf2, "제공 데이터", 17, True, DARK, first=True)
    para(tf2, "Raw data   11,939행 — 용접점별 공정조건", 14, False, DARK, space_before=12)
    para(tf2, "result          23행 — 작업일 × 불량유형 집계", 14, False, RED, space_before=6)
    para(tf2, "scaled_data.csv — 라벨 없음, 스케일링본", 14, False, GRAY, space_before=6)
    para(tf2, "설비 1종 · 품목 1종 · 두께 0.7mm 고정(분산 0)", 13, False, GRAY, space_before=14)
    para(tf2, "→ 착수 단계에서 라벨 구조부터 확인했다", 15, True, RED, space_before=18)


def s03_label(prs):
    s = blank(prs)
    head(s, "발견 1 — 용접점 단위 불량 라벨이 없다", "설명변수 11,939개 vs 라벨 8개")
    pic(s, "f3_label_resolution.png", Inches(1.4), Inches(1.95), w=Inches(10.5))
    kpi(s, [
        ("라벨 총 건수", "39건", RED),
        ("라벨 유효 표본", "8 작업일", RED),
        ("라벨 관측단위", "작업일", RED),
        ("용접점 단위 지도학습", "불가", RED),
    ])


def s04_integrity(prs):
    s = blank(prs)
    head(s, "발견 2 — 800점 구간이 네 작업일에 반복", "전체 데이터의 26.8%. 2020-04-03 은 작업일 전체가 이 구간이다")
    pic(s, "f12_insert_position.png", Inches(0.7), Inches(1.95), w=Inches(6.3))

    tf = textbox(s, Inches(7.3), Inches(2.0), Inches(5.4), Inches(3.3))
    para(tf, "확인된 사실", 16, True, DARK, first=True)
    para(tf, "· 사전 기준 없이 연속 일치 구간을 직접 탐색", 13, False, DARK, space_before=10)
    para(tf, "· 일치 구간 길이 800점, 4일 해시 동일", 13, True, DARK, space_before=6)
    para(tf, "   f079d764609a5603  (6개 비교쌍 완전일치)", 12, False, GRAY, space_before=2)
    para(tf, "· 전체의 26.8% (3,200 / 11,939점)", 13, True, RED, space_before=6)
    para(tf, "· 2020-04-03 은 작업일 전체가 이 구간", 13, True, RED, space_before=6)
    para(tf, "· 반증 대조: 나머지 구간은 6쌍 전부 불일치", 13, False, DARK, space_before=6)
    para(tf, "· 전류 0.01kA·전압 0.001V 미세값까지 동일", 13, False, RED, space_before=6)
    para(tf, "→ 모델 학습·평가에서 제외", 15, True, RED, space_before=16)

    kpi(s, [
        ("반복된 레코드", "800 × 4일", RED),
        ("전체 대비", "26.8%", RED),
        ("04-03 비중", "100%", RED),
        ("반증 대조", "구간 밖 전부 불일치", NAVY),
    ])


def s05_redefine(prs):
    s = blank(prs)
    head(s, "문제 재정의 — 예측 단위를 라벨 단위에 맞춘다", "용접점 단위 이진분류를 강행하지 않는다")
    from pptx.enum.shapes import MSO_SHAPE

    for i, (title, rows, col) in enumerate([
        ("과제 A — 작업일 단위", [
            ("예측 단위", "작업일"),
            ("목표", "불량건수 예측"),
            ("라벨", "있음 (8일, 39건)"),
            ("방법", "지도학습 (Poisson 계열)"),
            ("평가", "Leave-One-Out CV"),
        ], NAVY),
        ("과제 B·C — 용접점 단위", [
            ("예측 단위", "용접점"),
            ("목표", "결합관계 이탈 탐지"),
            ("라벨", "없음 / 전류값은 있음"),
            ("방법", "이상탐지 11종 + 회귀 9종"),
            ("평가", "안정일 LODO + 합성이상"),
        ], RED),
    ]):
        x = Inches(0.8 + i * 6.2)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(2.0), Inches(5.6), Inches(3.9))
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT
        box.line.color.rgb = col
        box.line.width = Pt(2)
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Pt(16)
        tf.margin_top = Pt(12)
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = title
        _set(r, 18, True, col)
        for k, v in rows:
            pp = tf.add_paragraph()
            pp.space_before = Pt(9)
            r1 = pp.add_run()
            r1.text = f"{k}   "
            _set(r1, 12, False, GRAY)
            r2 = pp.add_run()
            r2.text = v
            _set(r2, 13, True, DARK)

    tf = textbox(s, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.8), PP_ALIGN.CENTER)
    para(tf, "모형 11종 × 특징공간 3종 × 합성 이상유형 6종을 동일 조건에서 비교했다", 15, True, DARK, first=True)
    para(tf, "최종 kNN 에는 A7 지속반복이상을 추가해 7유형 강건성을 점검했다", 12.5, False, GRAY, space_before=3)


def s06_taskA(prs):
    s = blank(prs)
    head(s, "과제 A — 작업일 불량건수 예측", "공정조건을 넣은 모형이 기준선을 넘지 못한다")
    pic(s, "f13_taskA_comparison.png", Inches(1.0), Inches(1.9), w=Inches(11.3))
    kpi(s, [
        ("M0 기준선 LOO-MAE", "1.536", NAVY),
        ("M2 Poisson GLM", "+11.5%", RED),
        ("M3 음이항 GLM", "+11.2%", RED),
        ("M1 비례 모형", "+42.6%", RED),
    ])


def s07_scaling(prs):
    s = blank(prs)
    head(s, "종속변수의 성격 — 불량건수는 생산량을 따라가지 않는다",
         "AIC 기준 상수 모형이 비례 모형을 이긴다 (ΔAIC = 3.05)")
    pic(s, "f6_scaling_check.png", Inches(1.0), Inches(1.95), w=Inches(11.3))
    tf = textbox(s, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.8), PP_ALIGN.CENTER)
    para(tf, "→ result 의 defect 는 전수검사 건수가 아니라 표본검사 적발건수에 가깝다. 종속변수를 불량건수로 둔다.",
         15, True, DARK, first=True)


def s08_flaw(prs):
    s = blank(prs)
    head(s, "과제 B — 1차 설계의 결함을 발견하고 수정했다", "평가설계가 모델 선택을 뒤집은 사례")
    from pptx.enum.shapes import MSO_SHAPE

    box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.0), Inches(5.6), Inches(3.5))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = GRAY
    box.line.width = Pt(1.5)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(16)
    tf.margin_top = Pt(12)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "1차 설계 (폐기)"
    _set(r, 17, True, GRAY)
    for t, sz, bold, c, sb in [
        ("복제 구간을 '이상'의 프록시로 사용", 13, False, DARK, 12),
        ("가압력 단변량 기준선", 13, False, DARK, 14),
        ("탐지율 100% / 오경보율 0.03%", 15, True, DARK, 4),
        ("모든 다변량 모형을 압도", 13, False, DARK, 8),
        ("문제: 프록시가 '가압력 높은 점'과 동일", 13, False, RED, 14),
        ("→ 성능이 아니라 프록시 정의를 측정한 순환 평가", 13, True, RED, 6),
    ]:
        pp = tf.add_paragraph()
        pp.space_before = Pt(sb)
        rr = pp.add_run()
        rr.text = t
        _set(rr, sz, bold, c)

    box2 = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.9), Inches(2.0), Inches(5.6), Inches(3.5))
    box2.fill.solid()
    box2.fill.fore_color.rgb = LIGHT
    box2.line.color.rgb = RED
    box2.line.width = Pt(2)
    box2.shadow.inherit = False
    tf2 = box2.text_frame
    tf2.word_wrap = True
    tf2.margin_left = Pt(16)
    tf2.margin_top = Pt(12)
    p2 = tf2.paragraphs[0]
    r2 = p2.add_run()
    r2.text = "재설계 (채택)"
    _set(r2, 17, True, RED)
    for t, sz, bold, c, sb in [
        ("정상 운전의 가압력–전류 결합(ρ=−0.735)을 기준으로", 13, False, DARK, 12),
        ("결합이 깨진 점을 이상으로 정의", 13, False, DARK, 4),
        ("합성 이상 주입: 전류만 k·σ 이동", 13, False, DARK, 10),
        ("가압력은 건드리지 않는다", 13, False, GRAY, 4),
        ("같은 가압력 단변량 기준선", 13, False, DARK, 14),
        ("ROC-AUC 0.498 — 무작위 수준", 15, True, RED, 4),
    ]:
        pp = tf2.add_paragraph()
        pp.space_before = Pt(sb)
        rr = pp.add_run()
        rr.text = t
        _set(rr, sz, bold, c)

    tf3 = textbox(s, Inches(0.8), Inches(5.85), Inches(11.7), Inches(0.9), PP_ALIGN.CENTER)
    para(tf3, "1차 설계였다면 최종모델로 선정되었을 모형이, 올바른 설계에서는 무작위 추측 수준이었다",
         16, True, DARK, first=True)


def s09_coupling(prs):
    s = blank(prs)
    head(s, "과제 B — 결합관계 이탈 탐지", "단변량 임계 규칙(점선)은 원리상 탐지하지 못한다")
    pic(s, "f15_coupling_detection.png", Inches(0.9), Inches(1.9), w=Inches(11.5))
    kpi(s, [
        ("M0 가압력 단변량", "AUC 0.498", RED),
        ("M1 전류 단변량", "AUC 0.615", RED),
        ("M3 Isolation Forest", "AUC 0.911", NAVY),
        ("오경보율", "1.20%", NAVY),
    ])
    tf = textbox(s, Inches(0.8), Inches(7.0), Inches(11.7), Inches(0.4), PP_ALIGN.CENTER)
    para(tf, "남은 약점: 약한 이탈(1σ) 탐지율 6.3% → 과제 C 에서 정면으로 다룬다", 12, True, RED, first=True)


def s09b_normal(prs):
    s = blank(prs)
    head(s, "과제 C — 공정 관계를 명시적으로 학습한다",
         "범용 이상탐지기는 공정 구조를 모른다. 이 설비에서는 알고 있다")
    pic(s, "f18_normal_model.png", Inches(0.8), Inches(1.9), w=Inches(11.7))
    kpi(s, [
        ("LODO R²", "0.509", NAVY),
        ("RMSE", "0.050 kA", NAVY),
        ("비교 모형", "9종", NAVY),
        ("일중위치 추가 시 최악일 R²", "−0.755", RED),
    ])
    tf = textbox(s, Inches(0.8), Inches(7.0), Inches(11.7), Inches(0.4), PP_ALIGN.CENTER)
    para(tf, "일중 드리프트는 9/9일 유의하지만 방향이 날마다 뒤바뀐다 — "
             "유의한 변수가 일반화에는 해로울 수 있다", 12, True, DARK, first=True)


def s09c_combine(prs):
    s = blank(prs)
    head(s, "평가를 넓히자 결론이 뒤집혔다",
         "한 가지 이상유형으로 고르면 그 유형에 특화된 모형이 뽑힌다")
    pic(s, "f22_benchmark_heatmap.png", Inches(0.9), Inches(1.85), w=Inches(7.4))

    tf = textbox(s, Inches(8.5), Inches(2.0), Inches(4.4), Inches(3.6))
    para(tf, "평가를 세 축으로 확장", 16, True, DARK, first=True)
    para(tf, "이상유형  1종 → 7종", 13, False, DARK, space_before=12)
    para(tf, "   현장 고장모드를 각각 모사", 11, False, GRAY, space_before=2)
    para(tf, "특징공간  1종 → 3종", 13, False, DARK, space_before=8)
    para(tf, "   물리 파생(동저항 R, 입열 E) 포함", 11, False, GRAY, space_before=2)
    para(tf, "모형      6종 → 11종", 13, False, DARK, space_before=8)
    para(tf, "결과", 15, True, RED, space_before=16)
    para(tf, "기존 공간 FS1 최악 유형 0.499", 12, False, RED, space_before=4)
    para(tf, "   전압·통전 고장은 원리상 탐지 불가", 11, False, GRAY, space_before=2)
    para(tf, "기존 최종모형 → kNN 거리로 교체", 12, True, RED, space_before=6)

    kpi(s, [
        ("kNN 최악 유형", "0.892", NAVY),
        ("기존 결합모형", "0.820", RED),
        ("단변량 기준선", "0.489", RED),
    ])


def s09d_final(prs):
    s = blank(prs)
    head(s, "최종 모형 — 합성 고장모드 7종 전부에서 안정",
         "kNN 거리 (k는 중첩 교차검증 선택), 물리 파생 특징공간 — 〔합성 이상〕 실제 불량 탐지율이 아니다")
    pic(s, "f25_severity_sweep.png", Inches(2.2), Inches(1.85), w=Inches(8.6))
    kpi(s, [
        ("합성 7유형 평균", "0.971", NAVY),
        ("최악 유형", "0.954", NAVY),
        ("약한 이상(1σ) 평균", "0.967", NAVY),
        ("안정일 오경보", "0.55%", NAVY),
    ])


def s09e_conformal(prs):
    s = blank(prs)
    head(s, "임계값 문제 — 보장의 전제를 검증했다",
         "컨포멀의 오경보율 보장은 교환가능성을 전제한다. 이 공정에는 드리프트가 있다")
    pic(s, "f26_conformal.png", Inches(0.7), Inches(1.9), w=Inches(7.0))

    tf = textbox(s, Inches(8.1), Inches(2.0), Inches(4.8), Inches(3.6))
    para(tf, "목표 1% 대비 실제 오경보율", 15, True, DARK, first=True)
    para(tf, "교차일 교정        1.47배", 13, False, RED, space_before=12)
    para(tf, "당일 순차 교정     8.71배", 13, True, RED, space_before=6)
    para(tf, "당일 무작위 교정   0.28배  충족", 13, False, DARK, space_before=6)
    para(tf, "무작위 분할은 정의상 교환가능 —", 11, False, GRAY, space_before=12)
    para(tf, "구현이 아니라 전제의 문제임을 확인", 11, False, GRAY, space_before=2)
    para(tf, "운영상의 함의", 15, True, RED, space_before=14)
    para(tf, "유효한 방식은 미래 점이 필요해", 12, False, DARK, space_before=4)
    para(tf, "실시간에 쓸 수 없다", 12, False, DARK, space_before=2)
    para(tf, "→ 팽창계수 보정으로 근사", 12, True, RED, space_before=6)
    para(tf, "'보장'이 아니라 '근사'로 보고한다", 12, True, DARK, space_before=8)


def s09f_link(prs):
    s = blank(prs)
    head(s, "합성 이상 0.971 — 그러나 실제 불량과의 연결은 검출되지 않았다",
         "숨기지 않고 함께 제시한다")
    tf = textbox(s, Inches(0.8), Inches(2.0), Inches(5.6), Inches(3.6))
    para(tf, "수행한 것", 16, True, DARK, first=True)
    para(tf, "최종 모형 출력을 작업일로 집계해", 13, False, DARK, space_before=10)
    para(tf, "실제 불량 39건과 대조 (순열검정 20,000회)", 13, False, DARK, space_before=2)
    para(tf, "복제 구간 3,200점 전부 제외", 12, False, GRAY, space_before=6)
    para(tf, "04-03 은 전체가 복제 → 표본 8일에서 7일로", 12, False, GRAY, space_before=2)
    para(tf, "결과", 16, True, RED, space_before=16)
    para(tf, "유의한 지표 0/5,  최소 순열 p = 0.651", 14, True, RED, space_before=6)
    para(tf, "불량건수 예측도 기준선 초과 없음", 13, False, DARK, space_before=4)

    tf2 = textbox(s, Inches(7.0), Inches(2.0), Inches(5.6), Inches(3.6))
    para(tf2, "유일하게 일관된 것 — 방향성", 16, True, NAVY, first=True)
    para(tf2, "이상점수 평균        양(+)", 13, False, DARK, space_before=12)
    para(tf2, "이상점수 p95         양(+)", 13, False, DARK, space_before=4)
    para(tf2, "플래그율 p01         양(+)", 13, False, DARK, space_before=4)
    para(tf2, "플래그율 p05         양(+)", 13, False, DARK, space_before=4)
    para(tf2, "4개 지표 전부 부호 일치", 13, True, NAVY, space_before=10)
    para(tf2, "\"이상이 많은 날에 불량이 많다\"", 12, False, GRAY, space_before=2)
    para(tf2, "방향은 일관되나 크기가 우연과 구분되지 않는다", 12, False, GRAY, space_before=6)
    para(tf2, "→ '연결이 없다'가 아니라 '검출할 표본이 없다'", 13, True, RED, space_before=12)


def s09g_tool(prs):
    s = blank(prs)
    head(s, "현장 활용 — 정교한 모형이 단순한 관리도에 졌다",
         "두 방법은 겨루는 관계가 아니라 다른 문제를 푼다")
    pic(s, "f29_day_control_chart.png", Inches(0.7), Inches(1.9), w=Inches(7.0))

    tf = textbox(s, Inches(8.1), Inches(1.95), Inches(4.9), Inches(3.9))
    para(tf, "공정 이동 3일 탐지", 16, True, DARK, first=True)
    para(tf, "kNN + 컨포멀 + CUSUM", 12.5, False, DARK, space_before=10)
    para(tf, "   0/3  (15개 작동점 전부)", 13.5, True, RED, space_before=2)
    para(tf, "가압력 중앙값 2σ 관리도", 12.5, False, DARK, space_before=8)
    para(tf, "   3/3,  오경보 0/5", 13.5, True, NAVY, space_before=2)
    para(tf, "   정확 순열검정 p = 0.0179", 13.5, True, NAVY, space_before=2)
    para(tf, "   (C(8,3)=56 배열 전수 열거)", 10.5, False, GRAY, space_before=1)
    para(tf, "평가 대상을 밝힌다", 14, True, DARK, space_before=12)
    para(tf, "03-25 · 03-31 · 03-27", 11.5, False, DARK, space_before=3)
    para(tf, "03-27 은 불량 라벨 없는 예측대상일", 10.5, False, RED, space_before=1)
    para(tf, "04-03 은 복제 100% → 평가 제외", 10.5, False, RED, space_before=1)
    para(tf, "→ 공정 이동 탐지이며,", 12, True, RED, space_before=8)
    para(tf, "   불량 발생 탐지 근거는 없다", 12, True, RED, space_before=1)

    kpi(s, [
        ("일 단위 경보", "가압력 2σ 관리도", NAVY),
        ("작업일 단위 검정", "p = 0.0179", NAVY),
        ("적용 단계", "시범 운용", RED),
        ("필요 참조일", "최소 25일", RED),
    ])


def s09h_whynot(prs):
    s = blank(prs)
    head(s, "용접점 단위 예측 — 왜 불가능한가. 다섯 겹의 증명",
         "불량 예측은 라벨이 있는 작업일 단위로 수행했다. 불가능한 것은 용접점 단위다")

    rows = [
        ("① 형식적으로 올바른 방법", "라벨 비율 학습(LLP) 4종 비교",
         "가방 5개 → 검정의 최소 p가 0.05. 판정 자체가 불가"),
        ("② 시뮬레이션", "관계를 직접 심고 같은 절차 적용",
         "불량률 0.347% 에서 가방 5→50 늘려도 AUC 0.40~0.52"),
        ("③ 외부 실제 라벨 데이터", "라벨 해상도만 낮춰서 비교",
         "제품 단위 0.958 → 가방 집계 0.627"),
        ("④ 실제 불량 39건과 대조", "6종 지표 × 순열검정 20,000회",
         "전부 비유의 (최소 p 0.651). 단 부호는 4개 모두 양(+)"),
        ("⑤ 전가 라벨 이진분류 강행", "작업일 그룹 분할로 올바르게 평가",
         "AUC 0.305 — 무작위보다 나쁨. 0.586 은 같은 작업일 누수"),
    ]
    y = Inches(1.92)
    for i, (t1, t2, t3) in enumerate(rows, 1):
        tf = textbox(s, Inches(0.8), y, Inches(4.3), Inches(0.8))
        para(tf, t1, 13.5, True, NAVY, first=True)
        tf2 = textbox(s, Inches(5.2), y, Inches(3.3), Inches(0.8))
        para(tf2, t2, 11.5, False, DARK, first=True)
        tf3 = textbox(s, Inches(8.6), y, Inches(4.2), Inches(0.8))
        para(tf3, t3, 11.5, True, RED, first=True)
        y = Emu(y + Inches(0.72))

    tf = textbox(s, Inches(0.8), Inches(5.65), Inches(11.9), Inches(1.4))
    para(tf, "단, \"관계가 없다\"고는 말하지 않는다", 15, True, DARK, first=True)
    para(tf, "동등성 검정(TOST)에서 기각 가능 최소 δ = 0.79~0.88 → 효과의 유무를 양방향 모두 판정 불가",
         13, False, GRAY, space_before=4)
    para(tf, "→ 그래서 제언 1순위는 '작업일을 더 모으기'가 아니라 '용접점 단위 검사 기록'이다",
         14, True, RED, space_before=8)


def s09i_external(prs):
    s = blank(prs)
    head(s, "외부 검증 — 합성 이상이 아닌 실제 라벨에서",
         "용접기 분석에서 확정한 설정을 튜닝 없이 그대로 적용 (부록 A)")
    pic(s, "f36_external_press.png", Inches(0.7), Inches(1.9), w=Inches(7.0))

    tf = textbox(s, Inches(8.1), Inches(2.0), Inches(4.8), Inches(3.6))
    para(tf, "KAMP 공개 데이터 (정상 20,000 / 이상 600)", 12, False, GRAY, first=True)
    para(tf, "실제 라벨 기준 ROC-AUC", 14, True, DARK, space_before=12)
    para(tf, "0.899", 26, True, NAVY, space_before=2)
    para(tf, "모형 순위도 재현", 14, True, DARK, space_before=12)
    para(tf, "kNN 1위 / 9개 모형", 13, True, NAVY, space_before=2)
    para(tf, "컨포멀 FPR 제어", 14, True, DARK, space_before=12)
    para(tf, "목표 1% → 실제 0.175%  충족", 13, True, NAVY, space_before=2)
    para(tf, "용접기에서 보장이 깨진 원인을", 11, False, GRAY, space_before=6)
    para(tf, "드리프트로 진단한 것이 확인됨", 11, False, GRAY, space_before=1)

    kpi(s, [
        ("실제 라벨 ROC-AUC", "0.899", NAVY),
        ("모형 순위", "1위 / 9", NAVY),
        ("튜닝", "없음", NAVY),
        ("한계", "α=1%에서 탐지율 43%", RED),
    ])


def s09j_window(prs):
    s = blank(prs)
    head(s, "지문 요구 ③ — 모델이 제안하는 안정 공정조건 범위",
         "서로 다른 두 산출 경로가 모순 없이 겹친다 (표본은 공유 — 독립 검증 아님)")
    pic(s, "f34_model_window.png", Inches(0.5), Inches(1.9), w=Inches(8.3))

    tf = textbox(s, Inches(9.0), Inches(2.1), Inches(3.9), Inches(3.6))
    para(tf, "2단 구조", 17, True, DARK, first=True)
    para(tf, "내측 — 운영 목표", 13, True, NAVY, space_before=12)
    para(tf, "가압력 2.323~2.356", 12, False, DARK, space_before=2)
    para(tf, "안정일 I-MR 관리한계", 11, False, GRAY, space_before=1)
    para(tf, "외측 — 이상 판정 경계", 13, True, RED, space_before=10)
    para(tf, "가압력 2.260~2.383", 12, False, DARK, space_before=2)
    para(tf, "이상탐지 모형 ∩ 지지영역", 11, False, GRAY, space_before=1)
    para(tf, "외측이 내측을 완전히 포함", 13, True, NAVY, space_before=12)
    para(tf, "자연변동과 모델 출력이", 11, False, GRAY, space_before=2)
    para(tf, "같은 영역을 지목", 11, False, GRAY, space_before=1)
    para(tf, "단, 같은 안정일 5일 표본을", 10.5, False, RED, space_before=6)
    para(tf, "공유 → 방법 간 일관성 확인", 10.5, False, RED, space_before=1)
    para(tf, "검증 안 된 LLP 구성요소는", 11, False, RED, space_before=10)
    para(tf, "최종 권고에서 제외", 11, True, RED, space_before=1)


def s_divider(prs):
    """본 발표 / 백업 구분 간지."""
    s = blank(prs)
    from pptx.enum.shapes import MSO_SHAPE
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.9), W, Inches(1.7))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background(); bar.shadow.inherit = False
    tf = textbox(s, Inches(0.9), Inches(3.25), Inches(11.5), Inches(1.0))
    para(tf, "이하 백업 슬라이드", 30, True, RGBColor(0xFF, 0xFF, 0xFF), first=True)
    tf2 = textbox(s, Inches(0.9), Inches(4.9), Inches(11.5), Inches(1.0))
    para(tf2, "질의응답 시 참조. 본 발표는 앞 장까지입니다.", 14, False, GRAY, first=True)


def s10_errors(prs):
    s = blank(prs)
    head(s, "오류분석 — 작동하는 조건과 실패하는 조건", "반례를 숨기지 않는다")

    rows = [
        ("과제 A 작동", "불량건수 중심 수준 추정", "95% 예측구간 포함률 100% (8/8일)", NAVY),
        ("과제 A 실패", "공정조건으로부터의 예측", "전 변수가 기준선 대비 +11.2% ~ +42.6% 악화", RED),
        ("과제 B 작동", "강한 결합 이탈 (k ≥ 3σ)", "탐지율 75.9%, AUC 0.911", NAVY),
        ("과제 B 실패", "약한 이탈 (k = 1σ)", "탐지율 6.3% — 초기 징후는 대부분 놓침", RED),
        ("반례", "2020-03-26", "공정조건 최상인데 불량률 최고 (7.0건/1000점)", RED),
    ]
    y = Inches(2.05)
    for tag, cond, ev, col in rows:
        tf = textbox(s, Inches(0.8), y, Inches(2.3), Inches(0.5))
        para(tf, tag, 14, True, col, first=True)
        tf2 = textbox(s, Inches(3.0), y, Inches(3.5), Inches(0.5))
        para(tf2, cond, 14, True, DARK, first=True)
        tf3 = textbox(s, Inches(6.4), y, Inches(6.2), Inches(0.5))
        para(tf3, ev, 13, False, GRAY, first=True)
        y = Emu(y + Inches(0.82))

    tf = textbox(s, Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.8))
    para(tf, "2020-03-26 은 모든 공정조건 지표가 가장 안정적인 날이면서 불량률이 가장 높다 — "
             "공정조건 기반 모델이 원리적으로 설명할 수 없는 사례", 14, True, DARK, first=True)


def s11_power(prs):
    s = blank(prs)
    head(s, "왜 실패하는가 — '관계 없음'과 '검출 불가'는 다르다",
         "몬테카를로 검정력 분석으로 둘을 구분했다")
    pic(s, "f17_power_analysis.png", Inches(1.0), Inches(1.9), w=Inches(11.3))
    kpi(s, [
        ("현재 표본", "8 작업일", RED),
        ("80% 검정력 검출 하한", "|ρ| ≥ 0.87", RED),
        ("관측된 최대 상관", "0.558", RED),
        ("ρ=0.5 검출 필요", "33 작업일", NAVY),
    ])


def s12_field(prs):
    s = blank(prs)
    head(s, "현장 활용방안", "모델 출력이 어떤 조치로 이어지는가")
    pic(s, "f9_stable_window.png", Inches(0.7), Inches(1.9), w=Inches(7.2))

    tf = textbox(s, Inches(8.2), Inches(2.0), Inches(4.5), Inches(4.0))
    para(tf, "① 공정 관리범위", 15, True, NAVY, first=True)
    para(tf, "가압력 2.323~2.356 bar", 12, False, DARK, space_before=4)
    para(tf, "전류 14.642~14.703 kA", 12, False, DARK, space_before=2)
    para(tf, "Cp 28.8~331.5 → 사양은 관리에 무의미", 11, False, GRAY, space_before=4)

    para(tf, "② 혼입 모니터링", 15, True, NAVY, space_before=14)
    para(tf, "가압력 3bar 초과 비율, 관리한계 1%", 12, False, DARK, space_before=4)
    para(tf, "적재 경로 건전성 점검도 겸한다", 11, False, GRAY, space_before=2)

    para(tf, "③ 재검사 기준", 15, True, NAVY, space_before=14)
    para(tf, "임계값 0.95 → 일평균 16점, 최대 37점", 12, False, DARK, space_before=4)
    para(tf, "검출률은 제시하지 않는다 (정답 부재)", 11, False, GRAY, space_before=2)

    para(tf, "④ 데이터 수집 개선", 15, True, RED, space_before=14)
    para(tf, "용접점 단위 라벨 + 33 작업일", 12, True, RED, space_before=4)


def s13_diff(prs):
    s = blank(prs)
    head(s, "차별성 — 알고리즘이 아니라 검증 설계", "각 항목이 어느 결과에 기여했는가")

    rows = [
        ("라벨 해상도에 맞춘 문제 재정의",
         "강행했다면 올바른 검증에서 AUC 0.305 — 무작위보다 나쁜 모형을 제출"),
        ("평가설계 결함의 자체 발견과 수정",
         "단변량 기준선 AUC 100% → 0.498. 평가설계가 모델 선택을 뒤집음"),
        ("공정 구조를 명시적으로 학습",
         "결합관계를 회귀로 학습(R² 0.509). AUC 0.498 → 0.930"),
        ("실패조건이 다른 두 모형을 결합",
         "평균 동률(0.914)이나 약점이 상보적 → 결합 0.930"),
        ("가설을 세우고 기각한 과정 기록",
         "EWMA 조기탐지 가설. 1.5σ만 유리 → 전 구간 제시 후 기각"),
        ("귀무 결과의 정량적 해석",
         "순열검정 + 검정력 분석 → 결론이 '중단'이 아니라 '33일 추가 수집'으로"),
        ("데이터 무결성 검증의 분석 편입",
         "혼입률-불량 무관 결과를 설명. 현업 적재 점검 규칙으로 환원"),
        ("검증 가능/불가능의 명시적 분리",
         "제시하지 않은 것마다 근거를 명기 (검출률·비용기반 임계값 등)"),
    ]
    y = Inches(2.0)
    for i, (t, d) in enumerate(rows, 1):
        tf = textbox(s, Inches(0.8), y, Inches(5.2), Inches(0.5))
        para(tf, f"{i}. {t}", 14, True, NAVY, first=True)
        tf2 = textbox(s, Inches(6.1), y, Inches(6.5), Inches(0.5))
        para(tf2, d, 12.5, False, GRAY, first=True)
        y = Emu(y + Inches(0.75))


def s14_close(prs):
    s = blank(prs)
    from pptx.enum.shapes import MSO_SHAPE

    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), W, Inches(1.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    tf = textbox(s, Inches(0.9), Inches(0.42), Inches(11.5), Inches(0.8))
    para(tf, "결론", 30, True, RGBColor(0xFF, 0xFF, 0xFF), first=True)

    tf2 = textbox(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(3.6))
    para(tf2, "제공 데이터는 용접점 단위 지도학습을 허용하지 않는다.", 19, True, DARK, first=True)
    para(tf2, "예측 단위를 라벨 단위에 맞추어 과제를 재정의하고, 두 층 각각에서 "
              "베이스라인을 포함한 복수 모델을 동일 조건에서 비교했다.", 15, False, GRAY, space_before=10)
    para(tf2, "공정조건으로 일별 불량건수를 설명할 수 없다는 결과는, 검정력 분석을 통해 "
              "'관계 없음'이 아니라 '현재 표본으로는 검출 불가'임을 확인했다.", 15, False, GRAY, space_before=12)
    para(tf2, "제시한 수치는 전부 재현 가능하며, 제시하지 않은 것에는 각각 근거가 있다.",
         17, True, RED, space_before=16)

    kpi(s, [
        ("과제 A 최종", "LOO-MAE 1.536", NAVY),
        ("과제 B 최종", "합성 AUC 0.971", NAVY),
        ("03-27 예측", "4.88건 [1, 10]", NAVY),
        ("추가 수집 제언", "용접점 단위 검사기록", RED),
    ], y=Inches(5.9), h=Inches(1.2))


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    # 본 발표 (14장) — 발견, 재정의, 모델, 한계 증명, 현장 활용, 결론
    main_deck = [s01_title, s02_problem, s03_label, s04_integrity, s05_redefine,
                 s09c_combine, s09d_final, s09h_whynot, s09i_external,
                 s09j_window, s09g_tool, s10_errors, s13_diff, s14_close]
    # 백업 (질의응답용) — 세부 근거
    backup = [s06_taskA, s07_scaling, s08_flaw, s09_coupling, s09b_normal,
              s09e_conformal, s09f_link, s11_power, s12_field]
    for fn in main_deck + [s_divider] + backup:
        fn(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"  saved -> report/발표자료.pptx  ({len(prs.slides.__iter__.__self__._sldIdLst)} 슬라이드)")
    print("  PDF 변환: PowerPoint 또는 한컴에서 '다른 이름으로 저장 → PDF'")


if __name__ == "__main__":
    main()
