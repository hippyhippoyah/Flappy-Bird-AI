import tensorflow as tf
import numpy as np
import math
import tkinter as tk
import threading

import model
import game


ai = model.Model('s')


def create_ai():
    if input('y to confirm new model (DELETES WEIGHTS): ') == 'y':
        global ai
        ai = model.Model(input('s for small model, b for big model: '))


def print_ai():
    print(ai)


def load():
    if input('y to load: ') == 'y':
        checkpoint_num = input('load from checkpoint number: ')
        try:
            ai.load_weights('./checkpoints/checkpoint' + checkpoint_num)
            print(f'loaded "./checkpoints/checkpoint{checkpoint_num}"')
        except Exception as e:
            print(e)

    else:
        print('not loaded!!!')


def play():
    ai.play()


def train():
    print(ai.get_policy(game.board_start))
    print(ai.get_value(game.board_start))
    checkpoint_num = input('save to checkpoint number: ')
    print(f'will save to "./checkpoints/checkpoint{checkpoint_num}"')

    def stop(parent, thread):
        if input('s to stop training: ') == 's':
            if input('"save" to train on incomplete training data: ') == 'save':
                ai.save_incomplete = True
            else:
                ai.save_incomplete = False
            print('Stopping....')
            ai.stop_training = True
            thread.join()
            print('Training stopped')
            parent.destroy()

    class Train(tk.Frame):
        def __init__(self, parent, thread):
            tk.Frame.__init__(self, parent)
            self.parent = parent
            self.thread = thread

            stop_button = tk.Button(self.parent, text='Stop', width=25, command=lambda: stop(self.parent, self.thread))
            stop_button.pack()

    training = tk.Toplevel()
    training.title('Training')
    training.geometry('300x200')

    training_thread = threading.Thread(target=start_training, args=(str('./checkpoints/checkpoint' + checkpoint_num),))
    training_thread.start()

    training_frame = Train(training, training_thread)
    training_frame.pack()

    training.mainloop()


def start_training(checkpoint):
    ai.train(checkpoint)
    print(f'Policy on starting position: {ai.get_policy(game.board_start)}')
    print(f'Value on starting position: {ai.get_value(game.board_start)}')


def toggle_show_progress():
    ai.toggle_show_progress()


def show_save_log():
    save_log = open('save_log.txt', 'r')
    for line in save_log:
        print(line, end='')
    save_log.close()


class Main(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        load_button = tk.Button(self.parent, text='Load', width=25, command=load)
        train_button = tk.Button(self.parent, text='Train', width=25, command=train)
        play_button = tk.Button(self.parent, text='Play', width=25, command=play)
        create_ai_button = tk.Button(self.parent, text='Create New AI', width=25, command=create_ai)
        show_ai_info_button = tk.Button(self.parent, text='Show AI Info', width=25, command=lambda: print(ai))
        toggle_show_progress_button = tk.Button(self.parent, text='Toggle Progress', width=25, command=toggle_show_progress)
        show_save_log_button = tk.Button(self.parent, text='Show Save Log', width=25, command=show_save_log)

        load_button.pack()
        train_button.pack()
        play_button.pack()
        create_ai_button.pack()
        show_ai_info_button.pack()
        toggle_show_progress_button.pack()
        show_save_log_button.pack()


if __name__ == '__main__':
    root = tk.Tk()
    root.title('Connect4 AI')
    root.geometry('300x200')
    GUI = Main(root)
    GUI.pack()
    root.mainloop()



import tensorflow as tf
import numpy as np
import math
import game
import random
import datetime


class Model:
    architecture = 'b'
    load_from = ''
    save_to = ''

    def __init__(self, architecture):
        # Training thread stuff
        self.save_incomplete = False
        self.stop_training = False

        # Hyperparams
        self.cpuct = 3
        self.show_progress = True
        self.alpha = 0.5
        self.epsilon = 0.25
        self.games_per = 100
        self.batch_size = 32  # tensorflow default... change?
        # see https://arxiv.org/pdf/1902.10565.pdf section 3.1
        # if fast search (75% of training turns), then:
        #   turn off dirichlet noise
        #   retain subtree
        #   perform only mcts_num_simulations_fast simulations
        self.mcts_num_simulations_full = 25
        self.mcts_num_simulations_fast = 15
        self.p_full_search = 0.25
        self.mcts_num_simulations_play = 200

        if architecture == 's':
            # Create actual model
            board_input = tf.keras.Input(shape=(7, 6, 1))
            x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(board_input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # 3 residual blocks
            for i in range(3):
                x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                x = tf.keras.layers.add([board_input, x])
                x = tf.keras.layers.ReLU()(x)

                # policy head
                policy_layer = tf.keras.layers.Conv2D(2, (1, 1), padding='same')(x)
                policy_layer = tf.keras.layers.BatchNormalization()(policy_layer)
                policy_layer = tf.keras.layers.ReLU()(policy_layer)
                policy_layer = tf.keras.layers.Flatten()(policy_layer)
                policy_pred = tf.keras.layers.Dense(7, activation='softmax', name='policy')(policy_layer)

                # value head
                value_layer = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(x)
                value_layer = tf.keras.layers.BatchNormalization()(value_layer)
                value_layer = tf.keras.layers.ReLU()(value_layer)
                value_layer = tf.keras.layers.Flatten()(value_layer)
                value_layer = tf.keras.layers.Dense(256)(value_layer)
                value_layer = tf.keras.layers.ReLU()(value_layer)
                value_pred = tf.keras.layers.Dense(1, activation='sigmoid', name='value')(value_layer)

                self.model = tf.keras.Model(
                    inputs=board_input,
                    outputs=[policy_pred, value_pred]
                )

                # compile
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.005) # 0.001 default
                self.model.compile(
                    optimizer=optimizer,
                    loss={
                        "policy": tf.keras.losses.CategoricalCrossentropy(),
                        "value": tf.keras.losses.MeanSquaredError(),
                    }
                )

            print('Network created!: Small model')
            self.architecture = 's'
        else:
            # Create actual model
            board_input = tf.keras.Input(shape=(7, 6, 1))
            x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(board_input)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            # 19 residual blocks
            for i in range(19):
                x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                x = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
                x = tf.keras.layers.add([board_input, x])
                x = tf.keras.layers.ReLU()(x)

                # policy head
                policy_layer = tf.keras.layers.Conv2D(2, (1, 1), padding='same')(x)
                policy_layer = tf.keras.layers.BatchNormalization()(policy_layer)
                policy_layer = tf.keras.layers.ReLU()(policy_layer)
                policy_layer = tf.keras.layers.Flatten()(policy_layer)
                policy_pred = tf.keras.layers.Dense(7, activation='softmax', name='policy')(policy_layer)

                # value head
                value_layer = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(x)
                value_layer = tf.keras.layers.BatchNormalization()(value_layer)
                value_layer = tf.keras.layers.ReLU()(value_layer)
                value_layer = tf.keras.layers.Flatten()(value_layer)
                value_layer = tf.keras.layers.Dense(256)(value_layer)
                value_layer = tf.keras.layers.ReLU()(value_layer)
                value_pred = tf.keras.layers.Dense(1, activation='sigmoid', name='value')(value_layer)

                self.model = tf.keras.Model(
                    inputs=board_input,
                    outputs=[policy_pred, value_pred]
                )

                # compile
                optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
                self.model.compile(
                    optimizer=optimizer,
                    loss={
                        "policy": tf.keras.losses.CategoricalCrossentropy(),
                        "value": tf.keras.losses.MeanSquaredError(),
                    }
                )

            print('Network created!: Big model')
            self.architecture = 'b'

        # graph model
        # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
        # from IPython.display import Image, display
        # display(Image('model.png'))


        # Training or playing things
        # states[hash][ each action has a [expected reward, visit count]]
        self.states = {}
        self.states_train = []
        self.policy_train = []
        self.value_train = []

    def __str__(self):
        return f'Architecture: {self.architecture}\nLoaded from: {self.load_from}\nSaved to: {self.save_to}'

    def toggle_show_progress(self):
        self.show_progress = not self.show_progress
        print(f'Showing Progress: {self.show_progress}')

    def get_policy(self, state):
        return self.model.predict(state, verbose=0)[0][0]

    def get_value(self, state):
        return self.model.predict(state, verbose=0)[1][0]

    def load_weights(self, checkpoint):
        self.model.load_weights(checkpoint)
        self.load_from = checkpoint

    def save_weights(self, checkpoint):
        self.model.save_weights(checkpoint)
        self.save_to = checkpoint

        # TODO: Add loaded from to log. Also, maybe training iteration number
        save_log = open('save_log.txt', 'a')
        save_log.write(f'{checkpoint}: {datetime.datetime.now()}\n')
        save_log.close()


    # Train
    def train(self, checkpoint):
        # Training thread stuff
        self.save_incomplete = False
        self.stop_training = False
        for i in range(self.games_per):
            if self.stop_training:
                if self.save_incomplete:
                    # fit model
                    self.model.fit(np.array(self.states_train),
                                   {"policy": np.array(self.policy_train), "value": np.array(self.value_train)},
                                   batch_size=self.batch_size)
                    self.save_weights(checkpoint)
                    print(f'Saved weights to {checkpoint}.')
                else:
                    print('Weights were not saved')
                return
            print(f'\n\nGame #{i}')
            self.states = {}
            self.move(game.board_start)
        # fit model
        self.model.fit(np.array(self.states_train), {"policy": np.array(self.policy_train), "value": np.array(self.value_train)},
                       batch_size=self.batch_size)
        self.save_weights(checkpoint)
        print(f'Saved weights to {checkpoint}.')

    # Play
    def play(self):
        turn_number = 0.5
        state = game.board_start
        while True:
            turn_number += 0.5
            if abs(turn_number - int(turn_number)) < 0.1:
                print(f'\nTurn: {int(turn_number)} (x)')
                game.print_state(state)
            else:
                print(f'\nTurn: {turn_number} (o)')
                game.print_state_inverted(state)

            if game.check_end(state)[0]:
                print('GG')
                break

            while True:
                self.states = {}
                choice = input("Get policy (p) or input move (1-7): ")
                if choice == 'p':
                    print(self.get_policy(state))
                    print(self.get_value(state))
                    policy = self.run_mcts_simulations_and_get_policy(state, play=True)
                    print('\n' + str(policy))
                elif choice.isdigit() and 1 <= int(choice) <= 7:
                    state = game.update_state(state, int(choice) - 1)
                    break
                elif choice.isdigit() and 7 < int(choice) < 9999:
                    print(self.get_policy(state))
                    print(self.get_value(state))
                    temp = self.mcts_num_simulations_play
                    self.mcts_num_simulations_play = int(choice)
                    policy = self.run_mcts_simulations_and_get_policy(state, play=True)
                    self.mcts_num_simulations_play = temp
                    print('\n' + str(policy))

    # Make a move (takes turn) in a game
    # Returns game result (value)
    def move(self, state):
        # check if thread should stop (so returns None)
        if self.stop_training:
            return

        # print initial policy and value
        if self.show_progress:
            print(self.get_policy(state))
            print(self.get_value(state))

        # print board
        game.print_state(state)

        # reset states
        self.states = {}

        # check if the current state is already the end of game
        end, score = game.check_end(state)
        if end != 0:
            return -score

        # get improved policy
        improved_policy = self.run_mcts_simulations_and_get_policy(state)
        # print improved policy
        if self.show_progress:
            print("\n" + str(improved_policy))
        print()

        # pick action (action w/ the highest probability in improved policy)
        # action = improved_policy.index(max(improved_policy))

        # sample from improved policy to get action
        action = np.random.choice(list(range(7)), p=improved_policy)

        # move and recurse to continue game
        result = self.move(game.update_state(state, action))

        # check if thread should stop (so returns None)
        if self.stop_training:
            return

        # add to training data
        self.states_train.append(state[0])
        self.policy_train.append(improved_policy)
        self.value_train.append((result + 1) / 2)

        # return -result because of turn switch
        return -result

    # runs mcts_num_simulations simulations from a state
    # returns the improved policy based off visits
    def run_mcts_simulations_and_get_policy(self, state, play=False):
        # determine if full search of fast search
        if not play:
            full_search = random.random() < self.p_full_search

            # set mcts_num_simulations
            if full_search:
                mcts_num_simulations = self.mcts_num_simulations_full
            else:
                mcts_num_simulations = self.mcts_num_simulations_fast
        else:
            full_search = False
            mcts_num_simulations = self.mcts_num_simulations_play

        if self.show_progress:
            print('/', mcts_num_simulations)

        # run mcts_num_simulations simulations
        for i in range(mcts_num_simulations):
            if self.show_progress and (i % 5 == 0):
                print(i + 1, end=',')
            # call mcts with this state as root
            self.mcts(state, noise=full_search)

        # get the improved policy based off number of visits
        improved_policy = [0, 0, 0, 0, 0, 0, 0]
        for action in range(7):
            improved_policy[action] = self.states[str(state)][action][1] / mcts_num_simulations
        return improved_policy

    # Monte Carlo Tree Search
    # Returns game result (value)
    def mcts(self, state, noise=False):
        end, score = game.check_end(state)
        if end:
            return -score

        policy = self.get_policy(state)

        # create new element in states if element for current state is not in it
        state_hash = str(state)
        if state_hash not in self.states:
            value = self.get_value(state)
            self.states[state_hash] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

        # add dirichlet noise if this is the root (current move of game, not simulation)
        if noise:
            # create noise using alpha
            dirichlet_noise = np.random.dirichlet([self.alpha, self.alpha, self.alpha, self.alpha, self.alpha, self.alpha, self.alpha])
            for i in range(7):
                # insert noise using epsilon
                policy[i] = (1 - self.epsilon) * policy[i] + self.epsilon * dirichlet_noise[i]

        # give illegal moves -inf prob
        legal_policy = [(policy[i] if state[0][i][5][0] == 0 else -math.inf) for i in range(7)]
        policy = legal_policy

        # choose next action based on policy, [expected reward and visit counts] -- the elements of states, and cpuct
        max_upper = -math.inf
        best_action = 0
        for action in range(7):
            upper = self.states[state_hash][action][0] + self.cpuct * policy[action] * math.sqrt(
                sum(a[1] for a in self.states[state_hash])) / (1 + self.states[state_hash][action][1])
            if upper > max_upper:
                max_upper = upper
                best_action = action

        # recursively call mcts to get a result (value) of this move at this state
        value = self.mcts(game.update_state(state, best_action))

        # update Q (expected value)
        self.states[state_hash][best_action][0] = (self.states[state_hash][best_action][1] * self.states[state_hash][best_action][
            0] + value) / (self.states[state_hash][best_action][1] + 1)
        # update N (visit count)
        self.states[state_hash][best_action][1] += 1

        # return -value because turn switches
        return -value












# import pygame
# import random
# import math
# from pygame import mixer

# # Initialize pygame
# pygame.init()

# # Create screen
# screen = pygame.display.set_mode((800, 600))

# # Background
# background = pygame.image.load('space.jpg')

# # Background Sounds
# mixer.music.load('background.wav')
# mixer.music.play(-1)

# # Title and Icon
# pygame.display.set_caption("Space Invaders")
# icon = pygame.image.load('space-ship.png')
# pygame.display.set_icon(icon)

# # Player
# playerImg = pygame.image.load('alien.png')
# PlayerX = 370
# PlayerY = 480
# PlayerX_change = 0

# # Enemy
# enemyImg = []
# EnemyX = []
# EnemyY = []
# EnemyX_change = []
# EnemyY_change = []
# num_of_enemies = 6

# for i in range(num_of_enemies):
#     enemyImg.append(pygame.image.load('enemy.png'))
#     EnemyX.append(random.randint(0, 735))
#     EnemyY.append(random.randint(50, 150))
#     EnemyX_change.append(7)
#     EnemyY_change.append(50)

# # Bullet

# # Ready - Can't see bullet
# # Fire - bullet is currently moving
# BulletImg = pygame.image.load('bullet.png')
# BulletX = 0
# BulletY = 480
# BulletY_change = 25
# Bullet_state = "ready"

# # Score
# score_value = 0
# font = pygame.font.Font('freesansbold.ttf', 32)
# textX = 10
# textY = 10

# # Game over text
# over_font = pygame.font.Font('freesansbold.ttf', 64)

# def show_score(x, y):
#     score = font.render("Score: " + str(score_value), True, (255, 255, 255))
#     screen.blit(score, (x, y))

# def game_over_text():
#     over_text = over_font.render("GAME OVER!!!", True, (255, 255, 255))
#     screen.blit(over_text, (180, 250))
#     play_again_text = font.render('Press "p" to play again', True, (255, 255, 255))
#     screen.blit(play_again_text, (240, 325))

# def player(x, y):
#     screen.blit(playerImg, (x, y))

# def enemy(x, y, i):
#     screen.blit(enemyImg[i], (x, y))

# def fire_bullet(x, y):
#     global Bullet_state
#     Bullet_state = "fire"
#     screen.blit(BulletImg, (x + 16, y + 10))

# def iscollision(EnemyX, EnemyY, BulletX, BulletY):
#     distance = math.sqrt((math.pow(EnemyX - BulletX, 2)) + (math.pow(EnemyY - BulletY, 2)))
#     if distance < 29:
#         return True
#     else:
#         return False

# # Play again
# def play_again():
#     for i in range(num_of_enemies):
#         EnemyX[i] = random.randint(0, 800)
#         EnemyY[i] = random.randint(50, 150)
#         enemy(EnemyX[i], EnemyY[i], i)

# # Game loop
# running = True
# while running:

#     # RGB - Red, Green, Blue
#     screen.fill((0, 0, 0))
#     # Background image
#     screen.blit(background, (0, 0))

#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     # Check key pressed
#         if event.type == pygame.KEYDOWN:
#             if event.key == pygame.K_a:
#                 PlayerX_change = -15
#             if event.key == pygame.K_d:
#                 PlayerX_change = 15
#             if event.key == pygame.K_SPACE:
#                 if Bullet_state is "ready":
#                     # Get current x position of player
#                     BulletX = PlayerX
#                     fire_bullet(BulletX, BulletY)

#         if event.type == pygame.KEYUP:
#             if event.key == pygame.K_a or event.key == pygame.K_d:
#                 PlayerX_change = 0

#     # Ensure player stays on the screen
#     PlayerX += PlayerX_change

#     if PlayerX < 0:
#         PlayerX = 0
#     elif PlayerX > 736:
#         PlayerX = 736

#     # Enemy movement
#     for i in range(num_of_enemies):

#         # Game Over
#         if EnemyY[i] > 440:
#             for j in range(num_of_enemies):
#                 EnemyY[j] = 2000

#             game_over_text()
#             if event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_p:
#                    play_again()

#         EnemyX[i] += EnemyX_change[i]

#         if EnemyX[i] < 0:
#             EnemyX_change[i] = 7
#             EnemyY[i] += EnemyY_change[i]
#         elif EnemyX[i] >= 736:
#             EnemyX_change[i] = -7
#             EnemyY[i] += EnemyY_change[i]

#         # Collision
#         collision = iscollision(EnemyX[i], EnemyY[i], BulletX, BulletY)
#         if collision:
#             BulletY = 480
#             Bullet_state = "ready"
#             score_value += 1
#             EnemyX[i] = random.randint(0, 800)
#             EnemyY[i] = random.randint(50, 150)

#         enemy(EnemyX[i], EnemyY[i], i)

#     # Bullet movement
#     if BulletY <= 0:
#         BulletY = 480
#         Bullet_state = "ready"

#     if Bullet_state is "fire":
#         fire_bullet(BulletX, BulletY)
#         BulletY -= BulletY_change



#     player(PlayerX, PlayerY)
#     show_score(textX, textY)
#     pygame.display.update()