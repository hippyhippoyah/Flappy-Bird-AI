import tensorflow as tf
from settings import *
from objects import Player, Tube, TubeController
import pygame

# Initialize pygame
pygame.init()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
background = pygame.image.load(BACKGROUND_IMG)

# Title
pygame.display.set_caption("Flappy Bird")

# Choose AI
choice = input('Enter file, ex: 2024-01-04 12:39:42.626259.data-00000-of-00001\n')
if choice == '':
    ai = False
else:
    ai = True

if ai:
    # Create model
    ai_player = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(5,)),
            tf.keras.layers.Dense(4,activation='tanh'),
            tf.keras.layers.Dense(4,activation='tanh'),
            tf.keras.layers.Dense(1, activation='tanh')
    ])

    # Load weights
    ai_player.load_weights(f'./weights/{choice}')

# Play
playerImg = pygame.transform.scale(pygame.image.load(PLAYER_IMG), (PLAYER_WIDTH, PLAYER_HEIGHT))
player = Player(playerImg)

# Tubes
tubes = TubeController()

# Score
score_value = [0]
font = pygame.font.Font('freesansbold.ttf', 32)
textX = 10
textY = 10

def show_score(x, y):
    score = font.render("Score: " + str(score_value[0]), True, (255, 255, 255))
    screen.blit(score, (x, y))

def reset_game():
    global score_value, player, tubes
    score_value[0] = 0
    player = Player(playerImg)
    tubes = TubeController()

# Game loop
game_mode_ai = ai
up_pressed = False
running = True
while running:
    # delay
    if not game_mode_ai:
        pygame.time.delay(DELAY)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # toggle game_mode_ai
    if ai:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            if not up_pressed:
                up_pressed = True
                game_mode_ai = not game_mode_ai
                print(f'AI: {game_mode_ai}')
        else:
            up_pressed = False

    # background
    screen.fill((255, 0, 0))
    screen.blit(background, (0, 0))

    # tubes
    tubes.update(screen)

    # player
    if game_mode_ai:
        if not player.update(screen, tubes.tubes, score_value=score_value, ai=ai_player, tube_controller=tubes):
            reset_game()
    else:
        if not player.update(screen, tubes.tubes, score_value=score_value):
            reset_game()

    # score
    show_score(10, 10)

    pygame.display.update()
    
print('done')
