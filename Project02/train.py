# train.py - poker.py 에서 사용하는 두 모델을 학습하고 테스트한다.
#
# 실행 방법 (preprocess.py, CSV 파일 두 개와 같은 폴더에서 실행)
#   python train.py
# 결과물: poker-model3.sav (앞 3장용), poker-model4.sav (앞 4장용)
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

from preprocess import load_data, build_dataset

# -------------------------------------------------------------------------------
# 1. 데이터 로드
# -------------------------------------------------------------------------------
train, test = load_data()

# -------------------------------------------------------------------------------
# 2. 공개된 카드 수(3장, 4장)별로 모델을 학습, 테스트, 저장
#    build_dataset() 은 카드를 특징 11개로 바꾸고, 5장 중 가능한 모든
#    3장(4장) 조합을 학습 샘플로 쓰며, 같은 행은 합쳐서 가중치로 넘겨준다.
# -------------------------------------------------------------------------------
for n_cards in (3, 4):
    X, y, w = build_dataset(train, n_cards)
    model = HistGradientBoostingClassifier(max_iter=200).fit(X, y, sample_weight=w)

    # 테스트: 테스트 데이터 각 패의 앞 n장으로 평가 (게임에서 받는 입력과 동일)
    X_test, y_test, _ = build_dataset(test, n_cards, augment=False, dedupe=False)
    proba = model.predict_proba(X_test)
    print('model{}: train rows = {}, test accuracy = {:.4f}, test log loss = {:.4f}'.format(
        n_cards, len(y), accuracy_score(y_test, model.classes_[np.argmax(proba, axis=1)]),
        log_loss(y_test, proba, labels=model.classes_)))

    # 학습된 모델을 파일로 저장
    filename = 'poker-model{}.sav'.format(n_cards)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print('saved:', filename)
