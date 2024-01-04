import tensorflow as tf
from ai_settings import *
from settings import *
from objects import Player, Tube, TubeController
import pygame
import random


def weight_perturbation(model):
    for layer in model.layers:
        trainable_weights = layer.trainable_variables
        for weight in trainable_weights:
            random_weights = tf.random.uniform(tf.shape(weight), -0.1, 0.1, dtype=tf.float32)
            weight.assign_add(random_weights)


# Initialize pygame
pygame.init()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
background = pygame.image.load(BACKGROUND_IMG)

# Title
pygame.display.set_caption("Flappy Bird")


# input
# The Birds Y position
# The Birds velocity
# The Birds distance from the pipe
# The Y position of the top pipe
# The Y position of the bottom pipe


best_player = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(5,)),
        tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
        tf.keras.layers.Dense(4, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
])


for i in range(GENERATIONS):
    print(f'Generation: {i}')
    # make 10 ais
    players = []
    for i in range(NUM_BIRDS):
        # copy the best_player
        new_player = tf.keras.models.clone_model(best_player)

        # mutate it randomly a bit
        weight_perturbation(new_player)

        #here 
    
        players.append(new_player)

    # have them all play
    birds = []
    playerImg = pygame.transform.scale(pygame.image.load(PLAYER_IMG), (PLAYER_WIDTH, PLAYER_HEIGHT))
    for i in range(NUM_BIRDS):
        birds.append(Player(playerImg))

    # Tubes
    tubes = TubeController()
    tubes.create_pair()

    # play the game
    alive = NUM_BIRDS
    while alive > 1:
        # pygame.time.delay(DELAY)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # background
        screen.fill((255, 0, 0))
        screen.blit(background, (0, 0))

        # tubes
        tubes.update(screen)

        # player
        for i in range(NUM_BIRDS):
            bird = birds[i]
            if bird.alive:
                if not bird.update(screen, tubes.tubes, ai=players[i], tube_controller=tubes):
                    alive -= 1


        pygame.display.update()
    
    for i in range(NUM_BIRDS):
        if birds[i].alive:
            break
    # i is the best
    best_player = players[i]
    


print('done')

