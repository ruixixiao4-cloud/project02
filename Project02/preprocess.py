"""
preprocess.py - 포커 프로젝트 공용 전처리 모듈
================================================================

[이 파일이 하는 일]
  카드 3~4장  ->  모델이 이해하기 쉬운 숫자 11개(특징)로 바꿔 준다.
  예) [(1,10), (2,10), (3,5)]  ->  "같은 숫자 2장, 한 쌍 1개, ..." 같은 숫자들

[누가 쓰는가]
  train.py : build_dataset()  로 학습 데이터를 만든다.
  poker.py : hand_to_array()  로 예측용 입력 1건을 만든다.

[가장 중요한 규칙]
  학습할 때와 예측할 때 반드시 같은 hand_features()를 쓴다.
  (둘이 다르면 모델이 학습 때 본 적 없는 입력을 받아 성능이 무너진다)

[이 특징들의 성질]
  - 카드 순서에 영향받지 않는다.     (앞 k장은 '집합'이라 순서가 의미 없음)
  - 무늬 번호(1~4)에 영향받지 않는다. (무늬 번호는 이름표일 뿐)
  - 에이스(A)는 1과 14 둘 다로 취급한다. (스트레이트/로열 플러시 판정용)
"""
from itertools import combinations
from functools import lru_cache

import numpy as np
import pandas as pd


# ================================================================
# 1. 상수: CSV 컬럼 이름과 특징 이름
# ================================================================
SUIT_COLS = ['Suit of Card {}'.format(i) for i in range(1, 6)]   # 무늬 컬럼 5개
RANK_COLS = ['Rank of Card {}'.format(i) for i in range(1, 6)]   # 숫자 컬럼 5개
LABEL_COL = 'Poker Hand'                                          # 정답(족보 점수 0~9)

# 특징 이름. 아래 hand_features()가 돌려주는 값과 "같은 순서"여야 한다.
# (특징을 추가할 땐 이름과 값을 같은 위치에 함께 추가할 것)
FEATURE_NAMES = [
    'n_distinct_ranks',   # 서로 다른 숫자가 몇 종류인가
    'max_same_rank',      # 같은 숫자가 최대 몇 장인가 (2=페어, 3=트리플, 4=포카드)
    'n_pairs',            # 정확히 2장인 숫자가 몇 개인가
    'n_trips',            # 정확히 3장인 숫자가 몇 개인가
    'n_quads',            # 4장인 숫자가 몇 개인가
    'max_same_suit',      # 같은 무늬가 최대 몇 장인가 (플러시 진행도)
    'n_distinct_suits',   # 서로 다른 무늬가 몇 종류인가
    'straight_draw',      # 연속 5칸 구간에 들어 있는 최대 카드 수 (스트레이트 진행도)
    'royal_cnt',          # {A,10,J,Q,K} 중 몇 종류를 갖고 있는가
    'sf_draw',            # 같은 무늬끼리만 본 straight_draw (스트레이트 플러시 진행도)
    'royal_flush_draw',   # 같은 무늬로 {A,10,J,Q,K} 중 몇 장을 갖고 있는가
]


# ================================================================
# 2. 특징 계산  (아래 두 함수는 표준 라이브러리만 사용)
#    -> poker.py 의 FILL HERE 영역에 그대로 복사해서 쓸 수 있다.
# ================================================================
_ROYAL = (1, 10, 11, 12, 13)   # 로열 플러시를 이루는 숫자 (A, 10, J, Q, K)


def _best_window(ranks):
    """
    [하는 일] 스트레이트가 얼마나 완성에 가까운지 센다.
              연속된 5칸 구간(A-5, 2-6, ..., 10-A) 중에서
              카드가 가장 많이 들어 있는 구간의 카드 수를 돌려준다.
    [예]  {10, J, Q}  -> 3   (10~14 구간에 3장)
          {5, 10}     -> 1   (두 카드가 한 구간에 같이 못 들어감)
          {A, 2, K}   -> 2   (K-A 는 이어지지만 A-2 로 '돌아서' 이어지진 않음)
    """
    vals = set(ranks)
    if 1 in vals:
        vals.add(14)          # 에이스는 1(낮은 쪽)도, 14(높은 쪽)도 된다
    # lo ~ lo+4 가 한 구간. lo = 1(A-5) ... 10(10-A)
    return max(sum(1 for v in vals if lo <= v <= lo + 4) for lo in range(1, 11))


def hand_features(hand):
    """
    [하는 일] 카드 3~4장을 특징 11개로 바꾼다. (FEATURE_NAMES 와 같은 순서)
    [입력]   hand = [(suit, rank), ...]   (길이 3 또는 4)
    [출력]   길이 11짜리 리스트
    """
    # (1) 먼저 세어 둔다. 아래 특징은 전부 이 세 가지로부터 계산한다.
    rank_cnt = {}        # 숫자별 장수      예) {10: 2, 5: 1}
    suit_cnt = {}        # 무늬별 장수      예) {1: 1, 2: 1, 3: 1}
    ranks_by_suit = {}   # 무늬별 숫자 묶음  예) {1: {10}, 2: {10}, 3: {5}}
    for s, r in hand:
        rank_cnt[r] = rank_cnt.get(r, 0) + 1
        suit_cnt[s] = suit_cnt.get(s, 0) + 1
        ranks_by_suit.setdefault(s, set()).add(r)

    counts = list(rank_cnt.values())   # 각 숫자가 몇 장씩인지 (예: [2, 1])

    # (2) 특징 11개 (순서는 FEATURE_NAMES 와 동일)
    return [
        len(rank_cnt),                                   # n_distinct_ranks
        max(counts),                                     # max_same_rank
        sum(1 for c in counts if c == 2),                # n_pairs
        sum(1 for c in counts if c == 3),                # n_trips
        sum(1 for c in counts if c == 4),                # n_quads
        max(suit_cnt.values()),                          # max_same_suit
        len(suit_cnt),                                   # n_distinct_suits
        _best_window(rank_cnt.keys()),                   # straight_draw
        sum(1 for r in _ROYAL if r in rank_cnt),         # royal_cnt
        # 무늬마다 따로 본 뒤 가장 큰 값 (무늬 번호 자체는 쓰지 않는다)
        max(_best_window(rs) for rs in ranks_by_suit.values()),                    # sf_draw
        max(sum(1 for r in _ROYAL if r in rs) for rs in ranks_by_suit.values()),   # royal_flush_draw
    ]
# ================================================================


# ================================================================
# 3. 예측용: poker.py 의 predict_call() 에서 사용
# ================================================================
@lru_cache(maxsize=None)
def _cached_features(sorted_hand):
    """
    [하는 일] 같은 패는 한 번만 계산하고 결과를 기억(캐시)해 둔다.
    [왜]      3장/4장 조합은 가짓수가 유한(약 2.2만 / 27만)해서
              캐시를 쓰면 학습 데이터 변환과 예측이 훨씬 빨라진다.
    [주의]    인자는 정렬된 튜플이어야 한다. (리스트는 캐시 키가 될 수 없음)
    """
    return tuple(hand_features(sorted_hand))


def hand_to_array(hand):
    """
    [하는 일] 예측용 입력 1건을 만든다.
    [입력]   hand = [(suit, rank), ...]   (길이 3 또는 4)
    [출력]   shape (1, 11) 의 numpy 배열 -> model.predict_proba() 에 바로 넣는다.
    [참고]   정렬하는 이유: 같은 패가 다른 순서로 와도 캐시를 공유하기 위해서.
    """
    return np.array([_cached_features(tuple(sorted(hand)))], dtype=float)


# ================================================================
# 4. 학습용: train.py 에서 사용
# ================================================================
def load_data(train_path='poker-hand-training.csv',
              test_path='poker-hand-testing.csv'):
    """학습/테스트 CSV를 읽어서 DataFrame 두 개로 돌려준다."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def build_dataset(df, n_cards, augment=True, dedupe=True):
    """
    [하는 일] 5장짜리 원본 데이터를 학습용 (X, y, w)로 바꾼다.

    [입력]
      df      : 원본 DataFrame (컬럼 11개: 무늬/숫자 x 5장 + 족보 점수)
      n_cards : 3 또는 4   (3 -> model3 용, 4 -> model4 용)
      augment : True  -> 5장 중 가능한 모든 n_cards장 조합을 학습 샘플로 쓴다.
                         (3장 모델 10배, 4장 모델 5배로 데이터가 늘어남)
                         카드 순서가 무작위임을 확인했으므로(원페어 위치가
                         10곳에 각각 약 10%) 어떤 부분집합도 '앞 n장'과
                         같은 분포라 이렇게 늘려도 된다.
                False -> 앞 n_cards장만 쓴다.
      dedupe  : True  -> 같은 (특징, 정답) 행을 하나로 합치고, 합친 개수를
                         가중치 w 로 돌려준다. (정보 손실 없이 크기만 줄임)
                         예) 3장 모델: 약 275만 행 -> 198행
                False -> 합치지 않는다. (이때 w 는 None)

    [출력]
      X : 특징 행렬  (numpy, 열 11개)
      y : 정답 점수 0~9
      w : 가중치 (dedupe=True 일 때만. model.fit(X, y, sample_weight=w) 에 사용)
    """
    assert n_cards in (3, 4)

    suits = df[SUIT_COLS].to_numpy()
    ranks = df[RANK_COLS].to_numpy()
    labels = df[LABEL_COL].to_numpy()

    # 어떤 카드 위치들을 샘플로 쓸지 정한다.
    #   augment=True : (0,1,2), (0,1,3), ... 처럼 5장 중 n장 고르는 모든 조합
    #   augment=False: (0,1,2) 또는 (0,1,2,3) 한 가지뿐
    idx_sets = list(combinations(range(5), n_cards)) if augment \
        else [tuple(range(n_cards))]

    # 한 행(5장)마다, 위치 조합별로 특징을 계산해서 모은다.
    feats, ys = [], []
    for srow, rrow, y in zip(suits, ranks, labels):
        cards = [(int(s), int(r)) for s, r in zip(srow, rrow)]   # [(무늬, 숫자) x 5]
        for idx in idx_sets:
            # 고른 카드를 정렬해서 캐시 키로 쓴다. (정답 y 는 5장 전체의 족보라 그대로)
            key = tuple(sorted(cards[i] for i in idx))
            feats.append(_cached_features(key))
            ys.append(y)

    # 특징 + 정답을 한 표로 묶는다. (컬럼 이름 = FEATURE_NAMES)
    X = pd.DataFrame(feats, columns=FEATURE_NAMES)
    X[LABEL_COL] = ys

    if dedupe:
        # 같은 (특징 11개 + 정답) 행끼리 묶고, 몇 개였는지를 w 로 저장한다.
        g = X.groupby(FEATURE_NAMES + [LABEL_COL]).size().reset_index(name='w')
        return (g[FEATURE_NAMES].to_numpy(dtype=float),
                g[LABEL_COL].to_numpy(),
                g['w'].to_numpy(dtype=float))

    return (X[FEATURE_NAMES].to_numpy(dtype=float),
            X[LABEL_COL].to_numpy(),
            None)


# ================================================================
# 5. 직접 실행하면 동작 확인용 출력만 한다:  python preprocess.py
# ================================================================
if __name__ == '__main__':
    train, test = load_data()
    for k in (3, 4):
        X, y, w = build_dataset(train, k)
        print('model{}: X={}, y={}, 가중치 합계={:.0f}'.format(
            k, X.shape, y.shape, w.sum()))
    print('예시:', dict(zip(FEATURE_NAMES,
                            hand_features([(1, 10), (1, 11), (1, 13)]))))
