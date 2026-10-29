# tell-scope

해석 가능한 AI 텍스트 탐지기 [DetectorScope](https://github.com/google-deepmind/detectorscope)로 글에서 "AI 티"가 나는 지점을 찾고, 그 지점만 고쳐 쓴 뒤 다시 채점하는 에이전트 스킬입니다. Claude Code, Codex처럼 `SKILL.md`를 읽는 에이전트에서 씁니다.

- **영어**: DetectorScope의 공개 오픈 도메인 탐지기를 그대로 씁니다.
- **한국어**: 공개 탐지기는 한국어에서 우연 수준(AUROC 0.39~0.53)이라, 같은 SAE 특징 위에 한국어 헤드(`heads/ko.json`)를 새로 학습했습니다. 실험적인 모델입니다.

```
$ python scripts/score.py draft.txt
[{"lang": "ko", "label": "ai", "p_ai": 0.969, "gap": 3.952,
  "ai_features": [{"importance": 0.773, "description": "합쇼체 종결 '-니다.'(생각합니다/바랍니다)",
                   "passages": ["...선될 것으로 기대[[됩니다.]] 또한 서버의 스..."],
                   "hint": "일부 문장을 해라체(-다)로 바꾸거나 …", "confidence": "high"}, ...],
  "human_features": [...]}]
```

에이전트는 이 출력에서 AI 쪽 특징이 켜진 구간(`[[...]]`)을 보고 그 부분만 고친 뒤 다시 채점합니다. 절차는 [`SKILL.md`](SKILL.md)에 있습니다.

## 설치

NVIDIA GPU(6GB 이상 권장)가 있으면 빠릅니다. CPU로도 돌지만 글 하나에 1분 이상 걸립니다.

```bash
git clone https://github.com/yuyeol3/tell-scope && cd tell-scope
python -m venv .venv
# Windows: .venv\Scripts\activate   macOS/Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU. CPU면 생략
pip install -r requirements.txt
python -m pytest -q tests                                               # 모델 없이 도는 단위 테스트
```

처음 채점할 때 Hugging Face에서 탐지기(약 11GB, Gemma-2-2b 포함)를 받습니다.

### 스킬로 등록

레포 폴더를 각 에이전트의 스킬 폴더에 연결합니다.

```bash
# macOS/Linux
ln -s "$PWD" ~/.claude/skills/tell-scope
ln -s "$PWD" ~/.codex/skills/tell-scope
```

```bat
:: Windows (관리자 권한 없이 되는 디렉터리 정션)
mklink /J %USERPROFILE%\.claude\skills\tell-scope %CD%
mklink /J %USERPROFILE%\.codex\skills\tell-scope %CD%
```

그다음 에이전트에게 "이 글 AI 티 빼줘"처럼 요청하면 됩니다.

## 채점기만 쓰기

```bash
python scripts/score.py a.txt b.txt          # 여러 파일을 한 번의 모델 로딩으로
python scripts/score.py draft.txt --lang ko  # 언어 지정 (기본은 자동 판별)
```

출력 필드는 `SKILL.md`의 표를 보세요. `gap`은 logit에서 임계값을 뺀 값이고, 0 아래로 내려가야 사람으로 판정됩니다.

## 속도

RTX 4050 Laptop(6GB)에서 잰 결과입니다.

| | 900토큰 한국어 글 1편 |
|---|---|
| CPU (fp32) | 72초 |
| GPU, 원본 그대로 | 16초 (VRAM이 넘쳐 시스템 메모리로 넘침) |
| GPU, 쓰지 않는 마지막 블록과 lm_head 제거 (`score.load`) | **3.7초**, 점수 동일 |

모델 로딩에는 약 1분이 걸립니다. 그래서 스킬은 후보 여러 개를 한 번에 채점합니다.

## 한국어 헤드

### 만든 과정

1. **공개 탐지기는 한국어에서 동작하지 않습니다.** 나무위키 문단과 AI 블로그 글 40쌍에서 AUROC 0.53, 한국어 PR 설명 39쌍에서 0.39였습니다. 사람 글은 ChatGPT 이전(나무위키 2021년 덤프, 2019~2022년 GitHub PR)에서 가져왔고, 같은 내용을 Claude가 다시 쓰게 해서 AI 글을 만들었습니다.
2. **SAE 특징에는 한국어 신호가 남아 있습니다.** 처음 측정한 AUROC 1.0은 아티팩트였습니다. AI 글에만 `#` 제목이 있었고, GitHub 사람 글에만 CRLF 줄바꿈이 있었습니다. 그래서 양쪽 모두 마크다운과 줄바꿈을 정규화했고(`score.plain`), 채점할 때도 같은 정규화를 거칩니다.
3. **헤드 학습**: [KatFish](https://github.com/Shinwoo-Park/katfishnet)(ACL 2025)의 논설문과 논문 초록 1,149편(사람 글, 그리고 GPT-4o, Solar, Qwen2, Llama3.1이 쓴 글)으로 L1 로지스틱 회귀를 학습했습니다(`scripts/train_ko_head.py`). 특징은 87개입니다. 시는 넣으면 산문으로 옮겨 가는 성능이 떨어져서 뺐습니다.
4. **특징 설명**: 원래 설명은 영어 글로 만들고 검증한 것이라 한국어에서는 맞지 않았습니다(예: 한국어 글에서 "침대와 매트리스"). 그래서 각 특징이 가장 강하게 켜진 한국어 구간을 모으고(`scripts/ko_snippets.py`), Claude가 그 구간을 읽고 한국어 설명을 다시 썼습니다. 설명마다 `confidence`가 붙어 있지만, DetectorScope처럼 설명을 따로 검증하지는 않았습니다.
5. **임계값**: KatFish의 AI 글(2024년 모델)은 요즘 모델의 글보다 티가 훨씬 강합니다(logit 중앙값 +3.6 대 0 근처). KatFish 기준 임계값(+0.80)을 쓰면 Claude가 쓴 블로그 글의 75%가 사람으로 통과했습니다. 그래서 블로그 쌍과 PR 쌍에서 따로 정한 값(−0.52, −0.59)이 비슷했던 것을 근거로 −0.5로 정했습니다.

### 성능

학습에 쓰지 않은 한국어 쌍(AI 글은 모두 Claude가 썼습니다)으로 잰 결과입니다.

| | AUROC | 임계값 −0.5에서 AI 적중률 / 사람 오판률 |
|---|---|---|
| 나무위키 vs AI 블로그 글 (40쌍) | 0.95 | 0.95 / 0.17 * |
| PR 설명 (39쌍) | 0.85 | 0.67 / 0.08 * |

\* 임계값을 다른 장르에서 정했을 때의 값입니다.

비교를 위해 [KatFishNet](https://aclanthology.org/2025.acl-long.1030/)(쉼표 특징 5개, 공개된 구현)을 같은 데이터로 재 보니, KatFish로 학습한 경우 0.32와 0.57이었습니다. 쉼표 습관이 장르마다 반대로 나타나기 때문이었습니다.

### 예시: 한국어 PR 설명

Claude가 쓴 PR 설명(합쇼체, "이를 해결하기 위해", "~것으로 기대됩니다", 템플릿 섹션)을 스킬 절차대로 3라운드 고쳐 쓴 결과입니다.

| 라운드 | 바꾼 것 | gap | 판정 (p_ai) |
|---|---|---|---|
| 0 | 원문 | +3.95 | AI (0.97) |
| 1 | 문두 연결어와 "기대됩니다" 맺음 제거, 목록을 명사형으로, 해요체 | +0.48 | AI (0.50) |
| 2 | "개요"와 "기대 효과" 섹션을 첫 문단에 합침 | +0.10 | AI (0.40) |
| 3 | 문장 끝을 개조식("~수정", "~덜 씀")으로 | −0.86 | 사람 (0.20) |

### 한계

- **존댓말 편향**: 이 헤드는 합쇼체('-습니다')와 해요체 문장 끝 자체를 AI 쪽으로 봅니다. 위 예시에서도 마지막 한 걸음은 개조식으로 바꿔서 넘었습니다. 학습 데이터에서 AI 글이 실제로 합쇼체를 훨씬 많이 쓰기 때문입니다(KatFish 논설문: 사람 45%, AI 88%. 초록: 사람 0%, AI 35%). 근거가 있는 상관관계지만, 합쇼체로 쓴 사람 글도 AI 쪽으로 밀립니다. 자기소개서처럼 합쇼체가 필요한 글에서는 종결 어미 특징이 남을 수밖에 없고, 스킬은 어조를 유지한 채 다른 특징만 고치도록 되어 있습니다.
- **같은 탐지기로만 확인**: 고친 글을 독립된 다른 탐지기로 다시 재 보지는 않았습니다. 점수가 내려간 것이 글이 사람다워졌다는 뜻인지, 이 헤드의 허점을 파고든 것인지는 구분하지 못했습니다.

- 평가 데이터가 작고(79쌍), AI 글을 쓴 모델도 Claude 하나뿐입니다.
- 나무위키 평가셋은 어조 차이로 부풀려졌을 수 있습니다. 사람 쪽은 백과사전체(합쇼체 2%)이고 AI 쪽은 블로그체(합쇼체 78%)입니다. 어조 차이가 작은 PR 평가셋(37% 대 59%)의 0.85가 더 현실적인 수치입니다. 요즘 GPT 등 다른 모델의 글에서는 확인하지 않았습니다.
- 자기소개서는 공개된 사람 글이 없어서 검증하지 못했습니다.
- KatFish의 사람 논설문은 초·중·고 학생 글이라, 사람 쪽 특징 일부는 "어린 학생 문체"(띄어쓰기 실수, 긴 문장 등)를 담고 있습니다. 스킬은 사람 쪽 특징을 늘리려 하지 않고, AI 쪽 특징을 줄이는 데만 씁니다.
- 1024 토큰(한국어 약 1,500자)을 넘는 부분은 잘립니다. 긴 글은 나눠서 채점하세요.

## 주의

- 이 결과는 **탐지기 하나**의 판정입니다. 이 탐지기를 통과했다고 해서 다른 탐지기나 사람 독자도 통과한다는 뜻은 아닙니다.
- 탐지기는 사람이 쓴 글도 AI로 판정하는 등 틀립니다. 원 DetectorScope와 마찬가지로, 개인에 대한 판단(학업 부정행위, 채용 등)에 쓰지 마세요.
- 이 도구의 목적은 본인 글의 문체를 다듬는 것입니다. 글의 내용과 사실은 바꾸지 않도록 설계했습니다.

## 라이선스와 출처

- 이 레포의 코드: Apache-2.0
- [DetectorScope](https://github.com/google-deepmind/detectorscope) 코드는 Apache-2.0이고, 탐지기 모델은 CC-BY-4.0입니다(Dugan et al., EMNLP 2026). 함께 받는 Gemma-2-2b에는 [Gemma 이용 약관](https://ai.google.dev/gemma/terms)이 적용됩니다.
- `heads/ko.json`은 [KatFish](https://github.com/Shinwoo-Park/katfishnet)(Park et al., ACL 2025)로 학습했습니다. KatFish 레포에는 라이선스가 명시돼 있지 않아서, 이 레포에는 데이터를 넣지 않았고 학습된 가중치와 설명만 넣었습니다.
- 스킬의 재작성 규칙은 DetectorScope의 feature-guided paraphrase 프롬프트(`scripts/prompts/adversarial/feat.txt`)를 참고했습니다.
