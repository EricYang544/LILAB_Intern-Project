# ELMo depth ablation on CoNLL-2003

PyTorch로 문자 기반 양방향 언어모델을 직접 학습하고, CoNLL-2003 NER에서 정적 임베딩과 비교하는 연구용 프로젝트다. 목표는 RTX 5090 또는 4090 한 대에서 **데이터 준비 이후 전체 실험 약 24시간**이다. 실행 시간은 서버 pilot으로 측정하며, 24시간 내 원문 점수 재현을 보장하지 않는다.

## 실험 구성

| Arm | NER 입력 | biLM |
|---|---|---|
| `static` | SENNA 50d + 문자 CNN | 없음 |
| `depth1` | static + 문맥 표현 | 직접 학습한 1층 |
| `depth2` | static + 문맥 표현 | 직접 학습한 2층 |

`include_public_elmo: false`가 기본값이다. 공개 ELMo는 다운로드·실행하지 않고 원문 점수는 문헌 참고값으로만 제시한다. 기본 실행은 3개 조건 × 5개 시드, 총 15회 NER 학습이다.

NER는 모든 조건에서 2층 BiLSTM(200/100 hidden units per direction) + CRF를 사용한다. 변경하는 depth는 **사전학습 biLM의 깊이**다. 직접 학습 모델은 방향별 hidden 1024, projection 128, 문자 CNN + highway로 구성한다. 두 방향은 별도 LSTM 스택을 사용하고 문자 인코더와 출력 softmax를 공유한다. NER 학습에서는 biLM을 고정하고 scalar mix·SENNA·NER 모델을 학습한다.

`configs/ner_24h.yaml`이 실험 설정의 기준이다. 시드는 13/21/42/87/100이며 출력은 entity-level exact-match micro F1이다. 모델 선택은 dev F1로만 하고, 전체 조건의 학습이 끝난 후 test를 평가한다. SST-5는 이번 범위에 포함하지 않았다.

## 설치

Python 3.10 이상을 사용한다. 로컬 macOS에서는 CPU 검증을 수행하고, 본실험은 Linux CUDA 서버에서 실행한다. 환경 간 `.venv`를 복사하지 않는다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

서버의 RTX 5090/4090에는 다음처럼 Blackwell 지원 CUDA wheel을 먼저 설치한다. 설치 조합은 서버 드라이버와 맞아야 하며, 아래 명령이 현재 서버에서 검증되었다는 의미는 아니다.

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[test]'
python scripts/check_environment.py
```

[PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)와 [Blackwell 지원 안내](https://pytorch.org/blog/pytorch-2-7/)를 참고한다. CUDA toolkit을 별도로 컴파일하지 않는 wheel 기반 구성이며, `check_environment.py`가 실제 projected-LSTM forward/backward를 검사한다. 서버에는 공식 CoNLL 평가기 실행용 `perl`도 필요하다.

로컬 CPU 환경:

```bash
python -m pip install -e '.[test]'
python scripts/check_environment.py --cpu
python -m pytest -q
bash scripts/smoke.sh
```

smoke는 합성 데이터·작은 모델·3개 arm·2개 시드만 사용한다. 공개 가중치를 다운로드하지 않으며 **smoke 점수는 벤치마크 결과가 아니다**. CPU 테스트에는 방향별 정보 누출, CRF 완전탐색 비교, 작은 LM 학습, checkpoint 재개 동등성, 캐시 무결성, 전체 파이프라인이 포함된다.

## 서버로 옮기기

프로젝트의 `src/`, `configs/`, `scripts/`, `tests/`, `pyproject.toml`, 문서를 복사한다. `.venv`, `__pycache__`, 로컬 smoke 데이터와 `runs/`는 제외한다. 서버에서 가상환경을 새로 만들고 editable install을 실행한다. 설정의 경로는 프로젝트 루트 기준이므로 디렉터리 위치가 달라도 된다.

원본 데이터·SENNA 정적 임베딩·CoNLL 평가기를 서버의 `data/raw/`에 다운로드한다. 압축 해제·준비 데이터·특징 캐시·체크포인트를 위해 여유 디스크를 확보한다. 30GB 이상의 여유를 권장하되 실제 사용량은 데이터와 checkpoint 크기에 따라 달라진다.

## 실행 순서

```bash
python -m elmo_project.prepare --config configs/ner_24h.yaml
python -m elmo_project.pilot --config configs/ner_24h.yaml
python -m elmo_project.run --plan runs/pilot/resolved_plan.json
python -m elmo_project.report --run-dir runs/ner_24h
```

처음부터 실행하려면 `bash scripts/reproduce_final.sh`를 사용한다. **재개할 때는 pilot을 다시 실행하지 말고 `run` 명령만 반복한다.** Pilot은 실행할 때마다 새 실측 계획 ID를 만든다.

- `prepare`: 원본 다운로드·추출, CoNLL 크기 검사, IOB2 변환, 고정 LM 부분집합, 메모리 매핑 배열, SENNA 행렬을 만든다. manifest와 모든 준비 파일의 SHA-256을 저장한다. 실패한 다운로드는 `.part`에 남고 재실행 시 다시 받는다. URL이 변경되었거나 직접 받은 파일이 있다면 `data/README.md`를 따른다.
- `pilot`: 실제 데이터 길이 분포에서 두 LM의 optimizer step, exact-softmax 검증, 두 종류의 특징 추출, 세 NER 조건의 비용을 측정한다. 점수에 따른 튜닝을 하지 않는다. GPU 정보와 추정 시간을 `runs/pilot/measurements.json`에 저장한다.
- `run`: 고정 계획의 두 LM을 같은 batch 순서·step 한도·토큰 상한으로 학습한다. best LM은 검증 NLL, best NER는 dev F1로 선택한다. 모든 NER 학습 완료 후 test를 평가하고 `conlleval`과 결과를 대조한다.
- `report`: 완료된 모든 arm/seed가 있을 때만 평균·표준편차·개선량을 생성한다. 최상 test 시드를 선택하거나 미완료 실험을 평균에 포함하지 않는다.

Pilot은 LM 학습 14시간, 특징 추출 2시간, NER 5시간, pilot 1시간, 여유 2시간을 기본 배정으로 사용한다. 안전계수 1.3을 적용하고, 공통 LM step 수와 공통 NER 최대 epoch를 결정한다. NER는 최소 10 epoch를 확보하지 못하면 거절한다. 실제 조기 종료는 patience 8이다. 예산이 안 맞으면 `ready: false`와 이유를 저장하며 학습은 시작하지 않는다. 모델 폭이나 시드 수를 몰래 줄이지 않는다.

24시간은 pilot과 `run`의 누적 실행 시간을 포함한다. 제한에 도달하면 step 경계에서 checkpoint를 저장하고 중단한다. 데이터 검증·마지막 저장 등으로 약간 초과할 수 있다. 실제 시간은 예측과 다를 수 있다.

```bash
# 일반 재개: 같은 24시간 누적 한도
python -m elmo_project.run --plan runs/pilot/resolved_plan.json
# 사용자가 명시적으로 총 한도를 28시간으로 늘릴 때만
python -m elmo_project.run --plan runs/pilot/resolved_plan.json --extend-hours 4
```

첫 Ctrl-C는 다음 안전 경계에서 중단한다. 강제 종료 후에는 마지막 완료 checkpoint부터 재개한다. 코드·데이터·GPU가 달라지면 기존 계획을 거절한다. 조건을 바꾼 새 실험은 새 `run_dir`/`pilot_dir`로 pilot부터 실행한다. 운영체제나 PyTorch 버전도 유지하고, 실험별 환경 파일을 보관한다.

## 데이터와 원문 비교 조건

- LM: [1 Billion Word Benchmark](https://www.statmt.org/lm-benchmark/)의 이미 섞인 train shard를 seed 42로 정렬한 순서에서 최대 5천만 토큰을 선택한다. 문장을 유지해 선택하고 마지막 몇 토큰은 목표보다 작을 수 있다. 실제 수와 출처 행은 manifest에 기록한다.
- 출력 어휘: train에서 빈도순 상위 50,000단어 + 4개 특수 토큰. 입력은 출력 OOV도 UTF-8 byte 문자 CNN으로 표현한다. held-out의 단어는 입력 문자 인코딩에만 사용한다.
- LM 학습 문장은 최대 64단어 조각으로 나누고 BOS/EOS를 추가한다. 각 조각의 상태를 초기화한다. 원문의 stateful truncated BPTT와 다르다. NER 문장은 자르지 않는다.
- CoNLL: train/dev/test 14,041/3,250/3,453문장. 문서 구분행은 제외하며 원래 token·문장 경계를 유지한다. POS/chunk 라벨과 gazetteer는 사용하지 않는다. LM train에서 NER dev/test와 정확히 같은 문장은 제외한다.
- SENNA: 원형 일치, 소문자, 숫자를 `0`으로 정규화한 순서로 조회한다. train 단어와 원래 SENNA lexicon만 lookup에 등록한다. NER train에서 못 본 단어도 SENNA가 알고 있다면 공개 벡터를 사용한다.
- 원문 점수는 문헌 참고값이다. 공개 ELMo 모델을 실행한 결과나 본 구현의 재현 점수로 표시하지 않는다.
- 직접 학습 모델은 PyTorch projected LSTM을 사용하므로 원문 cell/projection clipping과도 차이가 있다. 원문과 같은 크기·학습량·출력 어휘가 아니므로 perplexity나 F1의 일치가 구현 정확성의 필수 조건은 아니다.

선택적 공개 가중치 어댑터와 참조 출력 검사 스크립트는 보관하되 기본 실험에서는 사용하지 않는다. 나중에 공개 모델 비교를 추가하려면 `include_public_elmo: true`로 설정하고 새 `data_dir`/`run_dir`/`pilot_dir`에서 준비·pilot부터 실행한다. 이 비교군은 NER LM 적응 학습을 생략하므로 완전한 원문 재현이 아니다.

## 결과와 해석

`runs/ner_24h/report/`에 다음을 생성한다.

- `results.csv`: seed별 F1·precision·recall·유형별 F1·dev 점수·시간·메모리·파라미터 수.
- `summary.json`, `report.md`: 평균 ± 표본 표준편차, 같은 seed의 static 대비 차이, depth 2−1 차이.
- `learning_curves.png`: LM NLL과 고정 seed의 NER dev 곡선.
- `lm_resources.json`: 실제 처리 토큰·throughput·학습률·시간·VRAM.
- `errors_<arm>.json`: 고정된 첫 seed의 오류 20문장과 gold/predicted span. 언어적 오류 해석은 사람이 채울 수 있도록 비워 둔다.

외부 기준은 [ELMo 논문](https://aclanthology.org/N18-1202.pdf)의 baseline 90.15 F1 / ELMo 92.22 ± 0.10 F1, [Lample et al.](https://aclanthology.org/N16-1030.pdf)의 90.94 F1이다. 점수 옆에 설정 차이를 기록한다. 먼저 직접 측정한 static 대비 개선과 두 depth의 차이를 점검하고, 문헌 점수와의 차이는 학습량·모델 규모·학습 방식 차이를 함께 고려해 해석한다. static 점수가 원문과 1 F1 이상 차이 나면 전처리·평가·학습 상태를 우선 확인하되 이를 보편적인 통과 기준으로 주장하지 않는다.

각 depth의 사전학습 seed는 하나다. NER 5회 표준편차는 사전학습 변동성을 포함하지 않는다. depth는 파라미터 수와 계산량도 바꾸므로 깊이만의 인과 효과로 해석하지 않는다. static 대비 개선에는 추가 모델 용량과 사전학습 효과도 포함된다.

## 구조

`src/elmo_project/`에는 데이터 준비·모델·캐시·학습·평가·예산 계산 모듈이 있다. `configs/`는 실험 설정, `scripts/`는 실행 및 검사 도구, `tests/`는 정확성·재개·통합 검증을 담는다. `data/`와 `runs/`의 생성 파일은 버전 관리에서 제외한다. 이 하위 폴더의 실행 절차는 본 README를 따른다.
