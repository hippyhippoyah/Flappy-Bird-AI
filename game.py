import pygame
import random
import math

# Initialize pygame
pygame.init()

# Create screen
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
background = pygame.image.load('background.png')


# Title
pygame.display.set_caption("Flappy Bird")

# Player
player = pygame.image.load('bird.png')
playerX = WIDTH / 4
playerY = HEIGHT / 2
playerWidth = 100
playerHeight = 100
playerYVel = 0
player = pygame.transform.scale(player,(120,120))

# Tubes
tubesBottom = []
tubesTop = []
tubeX_change = 2
tube_height = 300
tube_space = 100

# Score
score_value = 0
font = pygame.font.Font('freesansbold.ttf', 32)
textX = 10
textY = 10

def show_score(x, y):
    score = font.render("Score: " + str(score_value), True, (255, 255, 255))
    screen.blit(score, (x, y))

def create_tubes():
    offset = 0
    # make it random'
    tube_up = pygame.image.load('tube-up.png')
    tube_down = pygame.image.load('tube-down.png')
    tube_up = pygame.transform.scale(tube_up, (200,200))
    tube_down = pygame.transform.scale(tube_down, (200,200))
    tubesBottom.append([tube_up, WIDTH, tube_height + tube_space])
    tubesTop.append([tube_down, WIDTH, 0])


create_tubes()

# Game loop
running = True
while running:
    pygame.time.delay(10)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        playerYVel = -4

    playerYVel += 0.1
    playerY += playerYVel

    # background
    screen.fill((255, 0, 0))
    screen.blit(background, (0, 0))

    # score
    show_score(10, 10)

    # player
    screen.blit(player, (playerX, playerY))

    # tubes
    i = 0
    while i < len(tubesBottom):
        tubeBottom = tubesBottom[i][0]
        tubeTop = tubesTop[i][0]
        tubesBottom[i][1] -= tubeX_change
        tubesTop[i][1] -= tubeX_change
        screen.blit(tubeBottom, (tubesBottom[i][1], tubesBottom[i][2]))
        screen.blit(tubeTop, (tubesTop[i][1], tubesTop[i][2]))
        if tubesBottom[i][1] == WIDTH / 2:
            create_tubes()
        elif tubesBottom[i][1] < -100:
            del tubesBottom[0]
            del tubesTop[0]
            i -= 1
        elif abs(tubesBottom[i][1] - playerX) <= tubeX_change:
            score_value += 1
        i += 1





    pygame.display.update()