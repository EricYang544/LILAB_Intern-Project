# Data and artifacts

`python -m elmo_project.prepare --config configs/ner_24h.yaml`이 다음 파일을 `data/raw/`에 준비한다. 인터넷 다운로드는 서버에서 실행한다. 이미 받은 파일이 있으면 같은 이름으로 배치할 수 있으며 archive를 임의로 재구성하지 않는다.

| 파일 | 출처 | 용도 |
|---|---|---|
| `lm1b.tar.gz` | [1B Word Benchmark](https://www.statmt.org/lm-benchmark/) | 원문과 같은 코퍼스의 축소 사전학습 |
| `conll2003.zip` | [CoNLL-2003 데이터 카드](https://huggingface.co/datasets/eriktks/conll2003)의 원본 loader가 사용하는 DeepAI 배포본 | 영어 NER 표준 분할 |
| `senna.tgz` | [SENNA](https://ronan.collobert.com/senna/) | 공개 50d 정적 임베딩 |
| `elmo_options.json`, `elmo_weights.hdf5` | [bilm-tf / AllenNLP](https://github.com/allenai/bilm-tf)의 original 2x4096/512 모델 | 선택적 공개 비교군 (`include_public_elmo: true`일 때만) |
| `conlleval.pl` | [Lample의 평가기 사본](https://github.com/glample/tagger/blob/master/evaluation/conlleval) | 독립적인 CoNLL entity F1 확인 |

기본 설정은 `include_public_elmo: false`이므로 ELMo 설정·가중치를 받지 않는다. SENNA 정적 임베딩 다운로드는 유지하며, 두 depth의 biLM은 무작위 초기화부터 직접 학습한다.

SENNA는 [연구·평가 목적의 비상업 라이선스](https://ronan.collobert.com/senna/download.html)를 사용한다. 데이터와 외부 가중치는 원래 배포처의 조건을 따르며 프로젝트 소스 배포물에 포함하지 않는다. CoNLL의 Reuters 원문과 배포 스크립트의 라이선스를 혼동하지 않는다.

LM archive에는 알려진 SHA-256을 검사한다. 다른 artifact는 첫 다운로드의 SHA-256을 manifest에 저장하고 이후 실행에서 재사용한다. 고정된 신뢰 checksum을 알고 있다면 YAML `sources`에 `<key>_sha256`을 추가할 수 있다. 배포 URL이 바뀌면 직접 원본을 같은 파일명으로 배치하거나 URL을 수정하고 **새 prepared/run 디렉터리**를 사용한다. 이름이 같은 다른 ELMo 모델로 대체하지 않는다.

압축 해제 시 경로 이탈과 symlink를 거절한다. CoNLL은 `train.txt`, `valid.txt`, `test.txt`, SENNA는 `words.lst`, `embeddings.txt`를 찾아서 행 수·차원을 검사한다. 데이터 준비 과정은 원래 CoNLL 토큰과 문장 경계를 유지하고 entity span을 보존해 IOB2로 바꾼다.

준비 파일은 `data/prepared/manifest.json`이 식별한다. 입력·출력 LM 어휘, 선택 코퍼스, 메모리 매핑 token 배열, CoNLL JSONL과 SENNA 행렬의 해시를 포함한다. 어휘와 라벨 설정을 바꾼 뒤 기존 manifest를 삭제하여 재사용하지 말고 새 디렉터리로 준비한다.
