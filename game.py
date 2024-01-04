import pygame
import math
from settings import *
from objects import Player, Tube, TubeController


# Initialize pygame
pygame.init()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
background = pygame.image.load(BACKGROUND_IMG)

# Title
pygame.display.set_caption("Flappy Bird")

# Player
playerImg = pygame.transform.scale(pygame.image.load(PLAYER_IMG), (PLAYER_WIDTH, PLAYER_HEIGHT))
player = Player(playerImg)

# Tubes
tubes = TubeController()
tubes.create_pair()

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
running = True
while running:
    pygame.time.delay(DELAY)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # background
    screen.fill((255, 0, 0))
    screen.blit(background, (0, 0))

    # score
    show_score(10, 10)

    # tubes
    tubes.update(screen)

    # player
    if not player.update(screen, tubes.tubes, score_value):
        reset_game()

    pygame.display.update()