# STEP5 융합 LLM 3종 비교 — CLIP2 (입력 고정)

## 자동 지표

| 모델 | 회차 | 후보 | 융합성공 | utt접지율 | 시도 | 시간(s) | peakVRAM(GiB) |
|---|---|---|---|---|---|---|---|
| qwen3 | 1 | 5 | 5/5 | 1.0 | 1 | 256.6 | 14.36 |
| qwen3 | 2 | 5 | 5/5 | 1.0 | 1 | 265.6 | 14.36 |
| nemo | 1 | 4 | 1/4 | 0.25 | 1 | 170.7 | 14.36 |
| nemo | 2 | 5 | 1/5 | 0.2 | 1 | 162.1 | 14.36 |

융합성공 = 발화+행동 윈도우의 후보가 두 증거를 스텝에 담고 reasoning_origin=utterance 인 비율(분모=해당 윈도우 후보 수).

## 윈도우별 3모델 대조 (판단 재료 — 유창성·일관성은 사람이 평가)

### W01 (fusion) 12~32s

**입력**
- 행동 @16s: 손가락으로 컴퓨터 케이스 우측 면의 원형 전원 버튼을 누르는 듯한 움직임을 한다
- 행동 @18s: 컴퓨터 케이스 후면 팬 격자망이 보이고, 좌측에는 USB 포트 및 스티커가 붙은 케이스 앞면이 보인다
- 행동 @24s: 케이스 상단 덮개를 아래쪽 방향으로 당겨 열면서 내려놓는다
- (행동 1건 더)
- 발화 @24s: “황색 4번에 백색 5번이면은 램 불량이네요 원인을 좁히기 전에 일단 연결 상태부터 전체 확인을 해야합니다”

**qwen3** (Qwen/Qwen3-14B)
- run1 [W01] origin=utterance
  - insight: 전원 버튼 LED의 깜빡임 패턴을 확인하여 문제 원인을 추정한다.
  - reasoning: LED의 깜빡임 패턴은 하드웨어 문제를 나타내는 중요한 지표이므로, 이를 확인하여 원인을 좁히는 것이 중요하다.
- run2 [W01] origin=utterance
  - insight: 전원 버튼을 누르고 LED 반응을 확인하여 전원 상태를 진단한다.
  - reasoning: 전원 버튼을 누르고 LED 반응을 관찰함으로써 전원 공급 장치나 메모리/프로세서 문제를 확인할 수 있다.

**nemo** (mistralai/Mistral-Nemo-Instruct-2407)
- run1 [W01] origin=model_inferred
  - insight: 전원 버튼을 눌러 전원 상태를 확인한다.
  - reasoning: 전원 버튼을 눌러 전원 상태를 확인하지 않으면, 전원이 꺼진 상태에서 작업을 시작할 수 있어 작업이 불가능하다.
- run2 [W01] origin=model_inferred
  - insight: LED 깜빡임 패턴을 통해 램 불량을 진단한다.
  - reasoning: LED가 황색 1회 + 백색 3회로 깜빡이면 램 불량으로 진단한다. 진단 코드표를 통해 해석한다.

### W02 (fusion) 26~42s

**입력**
- 행동 @30s: GPU 카드의 금속 클립을 왼손으로 잡고, 오른손으로 GPU를 앞으로 당기는 동작을 한다
- 행동 @34s: GPU 카드를 왼손으로 잡고 오른손으로 GPU를 앞으로 당김
- 행동 @38s: 모니터 화면이 꺼져 있다
- 발화 @24s: “황색 4번에 백색 5번이면은 램 불량이네요 원인을 좁히기 전에 일단 연결 상태부터 전체 확인을 해야합니다”

**qwen3** (Qwen/Qwen3-14B)
- run1 [W02] origin=utterance
  - insight: GPU를 제거하기 전에 쇼크포트와 클립의 상태를 확인한다.
  - reasoning: GPU를 제거할 때 쇼크포트나 클립이 손상되지 않도록 확인하는 것이 중요하다.
- run2 [W02] origin=utterance
  - insight: GPU 카드의 쇼크포트를 확인하여 제거 시 손상 방지한다.
  - reasoning: GPU 카드를 제거할 때 쇼크포트를 직접 확인하고 당기는 것이 칩이나 쇼크포트 손상을 방지한다.

**nemo** (mistralai/Mistral-Nemo-Instruct-2407)
- run1 [W02] origin=model_inferred
  - insight: GPU를 뽑기 위해 GPU의 금속 클립을 잡고, GPU를 앞으로 당기는 동작을 한다.
  - reasoning: GPU를 뽑기 위해 GPU의 금속 클립을 잡고, GPU를 앞으로 당기는 동작을 하지 않으면, GPU를 뽑을 수 없다.
- run2 [W02] origin=model_inferred
  - insight: GPU 카드를 교체한다.
  - reasoning: GPU 카드가 고장났기 때문에 교체한다.

### W03 (fusion) 48~62s

**입력**
- 행동 @52s: GPU를 잡고 치우려는 자세로 오른손 검지와 엄지를 사용하여 GPU의 좌측 가장자리를 누르고 왼손으로 GPU 아래쪽 쇼크포트를 잡아 당긴다
- 행동 @54s: GPU를 완전히 뽑아내면서 오른손 검지와 엄지로 GPU 상단 쇼크포트를 잡고 왼손으로 GPU 하단 쇼크포트를 잡으며 동시에 올리는 동작을 수행함
- 행동 @58s: GPU를 완전히 뽑아내고 왼손으로 GPU를 잡아 들며 오른손 검지와 엄지로 GPU의 쇼크포트를 잡아 당김
- 발화 @45s: “이렇게 손으로 직접 당겨 보면은 빠져 있는 것은 바로 알 수가 있습니다 연결 상태가 전부 괜찮아서 이제는 패턴대로 램을 봐야 합니다”

**qwen3** (Qwen/Qwen3-14B)
- run1 [W03] origin=utterance
  - insight: GPU를 제거할 때 쇼크포트와 클립을 동시에 잡아 당겨야 한다.
  - reasoning: GPU를 제거할 때 쇼크포트와 클립을 동시에 잡지 않으면 GPU나 메인보드에 손상이 발생할 수 있다.
- run2 [W03] origin=utterance
  - insight: GPU 카드를 제거할 때 쇼크포트를 직접 확인하고 당기는 것이 칩이나 쇼크포트 손상을 방지한다.
  - reasoning: GPU 카드를 제거할 때 쇼크포트를 직접 확인하고 당기는 것이 칩이나 쇼크포트 손상을 방지한다.

**nemo** (mistralai/Mistral-Nemo-Instruct-2407)
- run1 [W03+W04] origin=utterance
  - insight: RAM을 뽑고, 치수 확인 후 재삽입한다.
  - reasoning: RAM을 뽑고, 치수 확인 후 재삽입하지 않으면, RAM이 제대로 장착되지 않아 오류가 발생할 수 있다.
- run2 [W03] origin=utterance
  - insight: GPU를 손으로 직접 당겨 확인한다.
  - reasoning: GPU를 직접 당겨 확인하면 빠진 것을 바로 알 수 있다.

### W04 (fusion) 79~99s

**입력**
- 행동 @83s: RAM을 클립형 스타ンド에 고정되어 있는 상태에서 좌우로 가볍게 누르면서 끼워넣는다
- 행동 @87s: 검지와 엄지를 사용하여 RAM의 상단 끝 부분을 잡아 올려 뚫리는 방향으로 당긴다
- 행동 @91s: RAM을 완전히 뽑아내고, 치수 확인 후 재삽입 준비를 위한 자세를 취한다
- (행동 1건 더)
- 발화 @45s: “이렇게 손으로 직접 당겨 보면은 빠져 있는 것은 바로 알 수가 있습니다 연결 상태가 전부 괜찮아서 이제는 패턴대로 램을 봐야 합니다”
- 발화 @80s: “램은 슬롯이 총 6개로 구분되어 있는데 색상으로 쉽게 구분을 할 수가 있습니다”
- 발화 @90s: “램을 뺄때는 양쪽을 동시에 눌러 줘야 합니다”
- 발화 @95s: “안그러면은 슬롯이 파손될 수가 있습니다”

**qwen3** (Qwen/Qwen3-14B)
- run1 [W04] origin=utterance
  - insight: RAM을 제거할 때 양쪽을 동시에 눌러야 슬롯이 손상되지 않는다.
  - reasoning: RAM을 제거할 때 양쪽을 동시에 누르지 않으면 메인보드의 슬롯이 손상될 수 있다.
- run2 [W04] origin=utterance
  - insight: RAM을 제거할 때 양쪽을 동시에 눌러야 슬롯이 손상되지 않는다.
  - reasoning: RAM을 제거할 때 양쪽을 동시에 눌러야 슬롯이 손상되지 않으며, 이는 메인보드의 안정성을 보장한다.

**nemo** (mistralai/Mistral-Nemo-Instruct-2407)
- run1 [W03+W04] origin=utterance
  - insight: RAM을 뽑고, 치수 확인 후 재삽입한다.
  - reasoning: RAM을 뽑고, 치수 확인 후 재삽입하지 않으면, RAM이 제대로 장착되지 않아 오류가 발생할 수 있다.
- run2 [W04] origin=model_inferred
  - insight: RAM을 확인하고 뽑는다.
  - reasoning: RAM을 확인하고 뽑는 것은 RAM을 교체하기 전에 하는 일반적인 절차다.

### W05 (fusion) 95~107s

**입력**
- 행동 @99s: RAM을 살짝 내리고, 볼트 형태의 클립을 이용해 고정시키는 동작을 수행한다
- 행동 @103s: 모니터 화면이 꺼져 있다
- 발화 @90s: “램을 뺄때는 양쪽을 동시에 눌러 줘야 합니다”
- 발화 @95s: “안그러면은 슬롯이 파손될 수가 있습니다”

**qwen3** (Qwen/Qwen3-14B)
- run1 [W05] origin=utterance
  - insight: RAM을 고정할 때 방향을 맞추고 클립으로 고정해야 한다.
  - reasoning: RAM을 잘못 삽입하면 메인보드 슬롯에 손상이 발생할 수 있으므로, 방향을 확인한 후 고정해야 한다.
- run2 [W05] origin=utterance
  - insight: RAM을 삽입한 후 볼트 클립으로 고정하여 제자리에 제대로 삽입되었는지 확인한다.
  - reasoning: RAM을 고정하지 않으면 메인보드에 제대로 삽입되지 않아 시스템이 작동하지 않을 수 있다.

**nemo** (mistralai/Mistral-Nemo-Instruct-2407)
- run1 [W05] origin=model_inferred
  - insight: RAM을 고정한다.
  - reasoning: RAM을 고정하지 않으면, RAM이 흔들려 오류가 발생할 수 있다.
- run2 [W05] origin=model_inferred
  - insight: RAM을 삽입하고 고정한다.
  - reasoning: RAM을 삽입하고 고정하는 것은 RAM을 교체한 후 하는 절차다.