# 검증 기록

2026-11-17 로컬 macOS ARM 환경에서 확인한 내용이다.

## 통과

- Python 전체 소스·스크립트·테스트 문법 검사: `python3 -m compileall -q src scripts tests`.
- Bash 실행 스크립트 문법 검사: `bash -n scripts/reproduce_final.sh scripts/smoke.sh`.
- 두 YAML 설정의 Ruby Psych 파싱 검사.
- 표준 라이브러리 및 보조 런타임의 NumPy를 사용한 unittest: **12개 통과**.
- 검증 범위: 예산 계산·거절, 고정 시드 유지, 문자 byte 인코딩, IOB1→IOB2 span 보존, entity F1 수작업 예제, CoNLL 경계, archive 경로 이탈 거절, corpus 선택 재현성, JSON 저장, 합성 데이터 준비·배열/BOS/EOS 정렬·체크섬·재사용.

## 실행하지 못한 검증

**PyTorch 관련 13개 테스트는 건너뛰었다.** 로컬 가상환경에 PyTorch가 없고 네트워크 제한으로 pip 다운로드가 실패했다. 통과한 것으로 간주하지 않는다.

미실행 항목은 projected LSTM 학습·방향별 정보 누출·padding, CRF 완전탐색 대조, NER 과적합, 공개 가중치 수식·HDF5 매핑, checkpoint 재개 동등성, 특징 캐시 및 전체 학습 파이프라인이다.

실제 원본 데이터·SENNA·공개 ELMo 다운로드, `conlleval`과의 실행 결과 대조, 공개 가중치와 공식 구현의 수치 동등성, CUDA/5090/4090 검증, 24시간 실행 가능성 및 downstream 점수도 아직 확인하지 않았다.

## 서버에서 실행할 검사

README의 설치 절차를 마친 후 다음을 실행한다.

```bash
python scripts/check_environment.py
python -m pytest -q
bash scripts/smoke.sh
```

그다음 production `prepare`와 `pilot`을 실행한다. Pilot은 Perl 평가기를 먼저 점검하고, 처리량과 메모리를 측정한 뒤 실행 가능 여부를 저장한다. 모델 테스트·smoke·pilot이 성공하기 전에는 구현 검증 완료나 24시간 완료를 주장하지 않는다.

## 문헌 비교 전용 기본 설정 변경 검증

2026-11-18: 현재 저장소에 수정분을 반영하고 재검증했다. 기본 비교군을 `static`, `depth1`, `depth2`로 제한했다.

- unittest: **14개 통과, 14개 건너뜀**. 보조 런타임 NumPy를 사용했다.
- 로컬 파일 URL로 production 데이터 다운로드·압축 해제·준비 경로를 검사했다. ELMo URL 없이 준비가 완료되고, 명시적 opt-in에서만 ELMo artifact를 받는 것을 확인했다.
- 보고서의 문헌 수치가 측정 결과표·CSV·summary에 섞이지 않는 것을 확인했다.
- 새 production pilot 비교군 테스트를 포함한 PyTorch 관련 14개는 로컬 의존성 부재로 실행하지 못했다. 서버에서 `python -m pytest -q`와 `bash scripts/smoke.sh`로 확인해야 한다.
- 기본 실험에는 공개 ELMo 다운로드 및 원본 참조 구현과의 출력 대조가 필요하지 않다.
