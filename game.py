import pygame
import random
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
score_value = 0
font = pygame.font.Font('freesansbold.ttf', 32)
textX = 10
textY = 10


def show_score(x, y):
    score = font.render("Score: " + str(score_value), True, (255, 255, 255))
    screen.blit(score, (x, y))


# Game loop
running = True
while running:
    pygame.time.delay(10)
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
    player.update(screen, tubes.tubes)

    pygame.display.update()