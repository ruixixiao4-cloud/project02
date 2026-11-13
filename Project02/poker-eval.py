## ---------------- FILL HERE ---------------------------------
# user-defined part
import pickle
from functools import lru_cache

# 공용 전처리 모듈 (preprocess.py 는 이 파일과 같은 폴더에 있어야 한다)
from preprocess import hand_to_array

# ---------- 조정 가능한 매개변수 ----------
# 결정 규칙
#   'score'  : 기대 족보 점수 + SCORE_MARGIN >= 상대 점수 이면 call
#   'payoff' : call 했을 때의 기대 손익이 fold 보다
#              PAYOFF_MARGIN 넘게 좋으면 call
DECISION_RULE = 'payoff'

# 키 = 공개된 카드 수 (3 또는 4)
# 양수 = 더 보수적으로, 음수 = 더 공격적으로
SCORE_MARGIN = {3: 0.0, 4: 0.0}
PAYOFF_MARGIN = {3: 0.0, 4: 0.0}

# 손익 구조 (poker-eval.py 의 계산 방식과 동일)
WIN_GAIN = 3                 # 끝까지 가서 승리
LOSE_LOSS = -3               # 끝까지 가서 패배 (무승부는 0)
FOLD_COST = {3: -1, 4: -2}   # 3장 / 4장 공개 상태에서 fold
# ------------------------------------------

# 저장된 모델 불러오기
filename = 'poker-model3.sav'
model3 = pickle.load(open(filename, 'rb'))

filename = 'poker-model4.sav'
model4 = pickle.load(open(filename, 'rb'))

MODELS = {3: model3, 4: model4}


@lru_cache(maxsize=None)
def _score_proba(n_cards, feats):
    # 같은 특징 벡터는 한 번만 예측한다 (특징 조합은 수백 가지뿐이다)
    model = MODELS[n_cards]
    p = model.predict_proba([list(feats)])[0]
    # classes_ 기준으로 0~9 점수 10칸에 맞춘다 (모델에 없는 클래스가 있어도 어긋나지 않게)
    proba = [0.0] * 10
    for cls, pk in zip(model.classes_, p):
        proba[int(cls)] = float(pk)
    return tuple(proba)


def score_proba(hand):
    # 최종 5장 패의 점수가 0~9 일 확률을 돌려준다
    feats = tuple(hand_to_array(hand)[0])
    return _score_proba(len(hand), feats)


# 입력:
#   hand: (무늬, 숫자) 튜플의 리스트; 길이는 3 또는 4
#   opposite_score: 상대 플레이어의 점수
#
# 반환:
#   call 하려면 True,
#   fold 하려면 False
def predict_call(hand, opposite_score):
    n = len(hand)
    opposite_score = int(opposite_score)
    proba = score_proba(hand)

    if DECISION_RULE == 'score':
        exp_score = sum(k * p for k, p in enumerate(proba))
        return exp_score + SCORE_MARGIN[n] >= opposite_score

    # 'payoff': 끝까지 call 했을 때의 기대 손익을 지금 fold 하는 경우와 비교한다
    p_win = sum(proba[opposite_score + 1:])
    p_lose = sum(proba[:opposite_score])
    call_ev = WIN_GAIN * p_win + LOSE_LOSS * p_lose
    return call_ev - FOLD_COST[n] > PAYOFF_MARGIN[n]
## -------------------------------------------------

import pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import random

test = pd.read_csv('poker-hand-testing.csv')

money = [1000] * 10
for i in range(10):
    samp = np.random.randint(0, len(test), size=1000)
    X_test = test.iloc[samp, :-1].to_numpy()
    y_test = test.iloc[samp, -1].to_numpy()

    for test_hand, test_score in zip(X_test, y_test):
        # 가진 돈을 모두 잃으면 게임을 즉시 중단
        if money[i] <= 0:
            break

        test_input = [ (suit, rank) for suit, rank in zip(test_hand[::2], test_hand[1::2]) ]
        oppo_score = y_test[random.randrange(len(y_test))]

        if not predict_call(test_input[:3], oppo_score):
            money[i] -= 1
            continue

        if not predict_call(test_input[:4], oppo_score):
            money[i] -= 2
            continue

        if test_score > oppo_score:
            money[i] += 3

        elif test_score < oppo_score:
            money[i] -= 3
        # if tie, your money stays the same

print('average of money= {}'.format(round(np.mean(money), 3)))