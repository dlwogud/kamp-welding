#!/usr/bin/env bash
# 전체 파이프라인 일괄 실행: 전처리 -> 학습 -> 추론 -> 결과 생성
# 사용법: bash run_all.sh
set -euo pipefail

cd "$(dirname "$0")"
PY="./.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "가상환경이 없습니다. 먼저 아래를 실행하세요:"
  echo "  python3.11 -m venv .venv && ./.venv/bin/pip install -r requirements.txt"
  exit 1
fi
PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"

mkdir -p outputs/tables outputs/figures

# 이 실행의 시작 시각. 25 가 이 값으로 '이번 실행이 만든 산출물' 과
# '이전 실행에서 남은 산출물' 을 구분한다(진위 경계).
date +%s > outputs/.run_start

run () {
  echo ""
  echo "======================================================================"
  echo "  $1"
  echo "======================================================================"
  ( cd src && "$PY" "$1" )
}

run 01_eda.py                 # 데이터 구조 진단            -> 제1장
run 06_data_integrity.py      # 무결성 검증                 -> 제1.5절
run 02_risk_score.py          # 용접점 위험스코어·외부검증   -> 제1·3장
run 03_daily_model.py         # 작업일 모델·가설검정·예측    -> 제2장
run 07_model_comparison.py    # 모델 비교 (1차 설계)        -> 제2.2.1절
run 08_coupling_detection.py  # 결합관계 이탈 탐지          -> 제2.2절
run 09_power_analysis.py      # 검정력 분석                 -> 제3.5절
run 11_normal_model.py        # 정상운전 모델·잔차·결합      -> 제2.3절
run 12_benchmark.py           # 다중 이상유형 종합 벤치마크   -> 제2.4절
run 15_robustness.py          # 강건성·중첩CV로 k 선택        -> 제2.5절
run 14_final_model.py         # 최종 모형·컨포멀 임계·스코어  -> 제2.6절
run 13_inference.py           # 예측구간·베이지안·동등성검정  -> 제2.7·3.5절
run 05_stable_window.py       # 관리범위·재검사 기준         -> 제4.3절
run 20_model_window.py        # 모델 기반 공정윈도우 (요구③)  -> 제4.2절
run 21_defect_types.py        # 불량유형별 사전지정 가설검정   -> 제3장
run 16_defect_link.py         # 최종 모형-실제 불량 연결      -> 제2.8절
run 17_operational_value.py   # 운영 효익과 한계·손익분기     -> 제4장
run 24_day_level_validity.py  # 일 단위 경보 타당성·순환성 검증 -> 제4.2절
run 26_cost_model.py          # 금액 단위 손익 (물량 실측·단가 모수) -> 제4.5절
run 18_llp.py                 # 라벨 비율 학습(LLP)          -> 제2.9절
run 19_simulation.py          # 시뮬레이션 검증              -> 제2.10절
run 22_external_press.py      # 외부 검증 (1) 이상탐지       -> 부록 A.1
run 23_external_mold.py       # 외부 검증 (2) LLP 주장       -> 부록 A.2
run 04_naive_check.py         # 전가 라벨 대조 분석          -> 제2.11절 (가장 오래 걸림)

# 종료 조건 (1) — 본문 표에 옮겨 적은 숫자가 출처 csv 의 값과 같은지 대조한다.
run 27_verify_report.py       # 보고서 수치 대조              -> 제6.5절

# 종료 조건 (2) — 보고서가 인용한 모든 산출물이 이번 실행에서 생성되었는지 검사한다.
# 하나라도 누락/과거파일이면 0 이 아닌 코드로 끝난다.
run 25_manifest.py            # 산출물 진위 검사·매니페스트    -> 제6.5절

echo ""
echo "======================================================================"
echo "  완료"
echo "======================================================================"
echo "  표   : outputs/tables/   ($(ls outputs/tables/*.csv 2>/dev/null | wc -l | tr -d ' ') 개)"
echo "  그림 : outputs/figures/  ($(ls outputs/figures/*.png 2>/dev/null | wc -l | tr -d ' ') 개)"
echo "  예측 : outputs/prediction_20200327.csv"
echo "  스코어: outputs/weld_anomaly_scores.csv (최종), outputs/weld_risk_scores.csv (중간)"
echo "  검사 : outputs/run_manifest.json  (산출물 진위·해시)"
