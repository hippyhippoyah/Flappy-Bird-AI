import tensorflow as tf
from ai_settings import *
from settings import *
from objects import Player, Tube, TubeController
import pygame
import datetime


NUM_BIRDS = NUM_BIRDS_HIGH + NUM_BIRDS_LOW + 1  

def weight_perturbation(model, amount):
    for layer in model.layers:
        trainable_weights = layer.trainable_variables
        for weight in trainable_weights:
            random_weights = tf.random.uniform(tf.shape(weight), -amount, amount, dtype=tf.float32)
            weight.assign_add(random_weights)


# Initialize pygame
pygame.init()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Background
background = pygame.image.load(BACKGROUND_IMG)

# Title
pygame.display.set_caption("Flappy Bird")

# Create starting player
best_player = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(5,)),
        tf.keras.layers.Dense(4,activation='tanh'),
        tf.keras.layers.Dense(4,activation='tanh'),
        tf.keras.layers.Dense(1, activation='tanh')
])

# Do GENERATIONS generations
for generation in range(1, GENERATIONS + 1):
    print(f'Generation: {generation}')

    # create players
    players = [best_player]
    for i in range(NUM_BIRDS_HIGH):
        # copy the best_player
        new_player = tf.keras.models.clone_model(best_player)
        new_player.set_weights(best_player.get_weights())
        # "mutate"
        weight_perturbation(new_player, AMOUNT_HIGH * 0.95 ** min(generation, 40))
        players.append(new_player)

    for i in range(NUM_BIRDS_LOW):
        # copy the best_player
        new_player = tf.keras.models.clone_model(best_player)
        new_player.set_weights(best_player.get_weights())
        # "mutate"
        weight_perturbation(new_player, AMOUNT_LOW * 0.95 ** min(generation, 40))
        players.append(new_player)

    # have them all play
    birds = []
    playerImg = pygame.transform.scale(pygame.image.load(PLAYER_IMG), (PLAYER_WIDTH, PLAYER_HEIGHT))
    for i in range(NUM_BIRDS):
        birds.append(Player(playerImg))

    # Tubes
    tubes = TubeController()

    # play the game
    alive = NUM_BIRDS
    save_weights = False
    while alive > 1:
        # pygame.time.delay(DELAY)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # option to save weights
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            save_weights = True
            print('Will save weights')

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
    
    # set best_player to the remaining one
    for i in range(NUM_BIRDS):
        if birds[i].alive:
            break
     # i is the index of the best
    best_player = players[i]

    if save_weights or generation % 10 == 0:
        save_file = f'./weights/{datetime.datetime.now()}'
        best_player.save_weights(save_file)
        print(f'Weights saved to {save_file}')

save_file = f'./weights/{datetime.datetime.now()}'
best_player.save_weights(save_file)
print(f'Weights saved to {save_file}')
print('done')
