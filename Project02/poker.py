from pydoc import cli
import pygame, sys
pygame.init()

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

import pandas as pd
cand_hands = pd.read_csv('poker-hand-testing.csv')

def pick_opposite_score():
    return cand_hands['Poker Hand'].sample().values[0]

def pick_five_cards():
    sam = cand_hands.sample().values[0]
    score = sam[-1]
    lis = sam[:-1]
    return (
        [ (a, b) for a, b in zip(lis[::2], lis[1::2]) ],
        score)


# game money: initially, $1000
money = 1000

# game count: 최대 1,000번의 게임 (남은 게임 수)
game_count = 1000

# initially, pick five cards
full_hand, hand_score = pick_five_cards()
print(full_hand, hand_score)

# currently, three flipped cards
flipped = 3
current_cards = full_hand[:flipped]

# current game's opposite hand score
opposite_hand_score = pick_opposite_score()

game_count = game_count - 1
money = money - 1

font = pygame.font.SysFont("comicsans",20)
SCREEN_WIDTH, SCREEN_HEIGHT = 512, 512

# create window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("POKER")

CARD_WIDTH, CARD_HEIGHT = 60, 100
SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT = 32, 64

# create a surface object, image is drawn on it.
imp = pygame.image.load("cards.jpg")
imp = pygame.transform.scale(imp, (SUIT_ICON_WIDTH*4, SUIT_ICON_HEIGHT))

# list of suit icons (0: hear, 1: spade, 2: diamond, 3: club)
suits = list()

# 0: heart
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*3, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 1: spade
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (0, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 2: diamond
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# 3: club
surf = pygame.Surface((SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT))
surf.blit(imp, (0, 0), (SUIT_ICON_WIDTH*2, 0, SUIT_ICON_WIDTH, SUIT_ICON_HEIGHT) )
suits.append(surf)

# colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (128, 128, 128)

def draw_card(suit, rank, pos):
    # white card
    rect = pygame.Rect( pos, (CARD_WIDTH, CARD_HEIGHT) )
    pygame.draw.rect(screen, white, rect)

    # suit icon
    screen.blit(
        suits[suit], 
        (pos[0] + CARD_WIDTH/2 - SUIT_ICON_WIDTH/2, 
            pos[1] + CARD_HEIGHT*2/3 - SUIT_ICON_HEIGHT/2))

    # rank
    rank_text = font.render("{}".format(rank), True, black)
    rank_rect = rank_text.get_rect()
    rank_rect.center = (pos[0] + CARD_WIDTH/2, pos[1] + CARD_HEIGHT/4)
    screen.blit(rank_text, rank_rect)

def gamemoney_text():
    text = font.render("Your money: ${}".format(money), True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6)
    screen.blit(text, rank_rect)

def opposite_hand_text():
    text = font.render(
        "The opposite's hand score: {}".format(opposite_hand_score), 
        True, white)
    rank_rect = text.get_rect()
    rank_rect.center = (SCREEN_WIDTH/2, SCREEN_HEIGHT/6 + rank_rect.height)
    screen.blit(text, rank_rect)

BUTTON_WIDTH, BUTTON_HEIGHT = 120, 50

def draw_button(center_pos, button_text, button_color):
    # rect button
    #rect = pygame.Rect(
    #    (center_pos[0] - BUTTON_WIDTH/2, center_pos[1] - BUTTON_HEIGHT/2),
    #    (BUTTON_WIDTH, BUTTON_HEIGHT) )
    rect = pygame.Rect((0, 0, BUTTON_WIDTH, BUTTON_HEIGHT))
    rect.center = center_pos
    pygame.draw.rect(screen, button_color, rect) 

    # button text
    text = font.render(button_text, True, black)
    text_rect = text.get_rect()
    text_rect.center = rect.center
    screen.blit(text, text_rect)

msg = None
MESSAGE_HEIGHT = 40
def display_message(_msg):
    if _msg is not None:
        # white message board at the top of screen
        rect = pygame.Rect( (0, 0), (SCREEN_WIDTH, MESSAGE_HEIGHT) )
        pygame.draw.rect(screen, white, rect)
        
        # display message text
        text = font.render(_msg, True, black)
        rect = text.get_rect()
        rect.center = (SCREEN_WIDTH/2, MESSAGE_HEIGHT/2)
        screen.blit(text, rect)

# bet and die buttons at the bottom of screen
BET_BUTTON_CENTER = (SCREEN_WIDTH/4, SCREEN_HEIGHT*4/5)
bet_button_color = white

def draw_bet_button():
    draw_button(BET_BUTTON_CENTER, "CALL", bet_button_color)

# answer if pos is on the button
def on_bet_button(pos):
    return (abs(pos[0] - BET_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - BET_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

DIE_BUTTON_CENTER = (SCREEN_WIDTH/2, SCREEN_HEIGHT*4/5)
die_button_color = white

def draw_die_button():
    draw_button(DIE_BUTTON_CENTER, "FOLD", die_button_color)

def on_die_button(pos):
    return (abs(pos[0] - DIE_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - DIE_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# auto play with classifier
CLS_BUTTON_CENTER = (SCREEN_WIDTH*3/4, SCREEN_HEIGHT*4/5)
cls_button_color = white

def draw_cls_button():
    draw_button(CLS_BUTTON_CENTER, "AUTO", cls_button_color)

def on_cls_button(pos):
    return (abs(pos[0] - CLS_BUTTON_CENTER[0]) <= BUTTON_WIDTH/2) \
        and (abs(pos[1] - CLS_BUTTON_CENTER[1]) <= BUTTON_HEIGHT/2)

# play sound when cards are revealed
flip_sound = pygame.mixer.Sound("flipcard.wav")
def play_cardflip_sound(rep):
    pygame.mixer.Sound.play(flip_sound, loops=rep-1)

def draw_current_cards():
    for i, (suit, rank) in enumerate(current_cards):
        draw_card(
            suit-1, rank, # suit and rank 
            ((i+1)*SCREEN_WIDTH/7, SCREEN_HEIGHT*2/5))

# game state for updating screen
# 0: beginning of game - three cards are flipped
# 1: bet - one more card is flipped
# 2: bet & game over - the last card is flipped and
#    win/tie/lose messages are shown
#    then, go to the next game after 5 secs
# 3: die & game over - message shown then,
#    go to the next game after 5 secs
# 9: wait for the next action
update_state = 0

def draw_items_on_screen():
    screen.fill(black)
    opposite_hand_text()
    gamemoney_text()
    draw_bet_button()
    draw_die_button()
    draw_cls_button()
    draw_current_cards()

    if msg is not None:
        display_message(msg)

    pygame.display.flip()
    pygame.time.delay(200)

# drawing cards and playing sound
def update_cards_and_sounds():
    global flipped, update_state, full_hand, hand_score,\
        money, msg, current_cards, opposite_hand_score,\
        game_count, game_stop

    if update_state == 0:
        # at the beginning of game, three cards are flipped
        play_cardflip_sound(3)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 1:
        # check button clicked and now four cards revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # then, the screen of waiting the next action
        update_state = 9

    elif update_state == 2 or update_state == 3:
        # the last cards are revealed
        play_cardflip_sound(1)
        draw_items_on_screen()

        # delay another 1 sec for showing message
        pygame.time.delay(1000)

        # 종료 조건: 1,000번의 게임을 모두 마쳤거나 가진 돈을 모두 잃은 경우
        if game_count <= 0 or money <= 0:
            game_stop = True
            msg = "Game over! Final money: ${}".format(money)
            draw_items_on_screen()
            update_state = 9
            return

        msg = None
        draw_items_on_screen()

        # a new poker hand is drawn
        full_hand, hand_score = pick_five_cards()
        print(full_hand, hand_score)

        # three flipped cards
        flipped = 3
        current_cards = full_hand[:flipped]

        # opposite's hand score
        opposite_hand_score = pick_opposite_score()
        game_count = game_count - 1
        money = money - 1
        update_state = 0


game_stop = False
running = True
while running:
    # 게임 도중이라도 가진 돈을 모두 잃으면 즉시 중단
    if money <= 0 and not game_stop and update_state == 9:
        game_stop = True
        msg = "Game over! Final money: ${}".format(money)
        draw_items_on_screen()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN and not game_stop:
            click_pos = pygame.mouse.get_pos()

            auto_bet = False
            auto_die = False
            if on_cls_button(click_pos):
                if predict_call(current_cards, opposite_hand_score):
                    print("auto call!")
                    auto_bet = True
                    display_message("AI calls")
                    pygame.display.update()
                    pygame.time.delay(800)
                else:
                    print("auto fold!")
                    auto_die = True
                    display_message("AI folds")
                    pygame.display.update()
                    pygame.time.delay(800)

            if on_bet_button(click_pos) or auto_bet:
                flipped = flipped + 1
                current_cards = full_hand[:flipped]
                money = money - 1

                # if game over,
                if flipped == 5:
                    # case 1: win
                    if hand_score > opposite_hand_score:
                        money = money + 6
                        msg = "Win"
                        update_state = 2
                    # case 2: tie
                    elif hand_score == opposite_hand_score:
                        money = money + 3
                        msg = "Tie"
                        update_state = 2
                    # case 3: lose
                    else:
                        msg = "Lose"
                        update_state = 2
                # if cards can be flipped more,
                else:
                    update_state = 1
                        
            elif on_die_button(click_pos) or auto_die:
                msg = "Fold, go on to the next"
                update_state = 3

    # show cards and info message including sound
    update_cards_and_sounds()

    pygame.time.delay(200)