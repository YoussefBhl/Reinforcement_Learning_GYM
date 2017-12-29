import gym
import numpy as np
import random,tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter
import pickle
import tensorflow as tf

learning_rate = 0.001

env = gym.make('CartPole-v0')

#this is our neurl net model (using tfLearn)
def neural_network_model(input_size):

    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    y = [i[1] for i in training_data]
    if not model:
        model = neural_network_model(input_size=len(X[0]))

        model.fit({'input': X}, {'targets': y}, n_epoch=5,
                  snapshot_step=500, show_metric=True,run_id='CartPole-v2')
    return model

'''Qlearning is the most important functoin
    args: epsilon is float between 0 and 1 we choose a random float ([0,1])
    if this float >= epsilon then we will explore (choose a random action) else
    we exploit our model (predect the next action)
    we let the algorithm play many times (game_replay)
    for each game we calculate the game's score 
    if the score >= requirement_score then we add the game stats (obsravtions + actions) to the training_data
    the training_data is the data that we will use to make our model learn
    
    max_requ_score: is used to make the alogithme stop 
    when the 70% of the played games has a score >= requirement_score
    '''
def Qlearning(epsilon,requirement_score,max_requ_score,model=None,game_replay=1000):
    scores = []
    choices = []
    training_data = []
    for _ in range(game_replay):
        if (not _ % 100):  print('{} precessed'.format(_))
        env.reset()
        #each game we have to reset the variabls
        score = 0
        game = []
        prev_observation = []
        prev_action = 0
        #now the game begin
        for _ in range(50000):
            #Exploration
            if(np.random.rand()>=epsilon or len(prev_observation) == 0):
                action = random.randrange(0, 2)
            #exploitation
            else:
                action = np.argmax(model.predict
                (np.append(prev_observation,prev_action).reshape(-1, len(prev_observation)+1, 1))[0])
            #take the action and calculate the score
            observation, reward, done, info = env.step(action)
            score += reward
            #if the game start we save
            # the game stat(previous obseravtion, previous action and the chosen action)
            if (len(prev_observation)):
                # prev_observation = np.append(prev_observation,prev_action)
                game.append([np.append(prev_observation,prev_action), action])
            #update the vars
            prev_observation = observation
            prev_action = action
            #if the game is finished we move to next game
            if done: break
        #testing if the score is >= requirement score
        #we prepare the date to be feed to our model
        #we label are the action (CartPole-v0 have 2 actions)
        #we add score to the data cuz we need it later to separete the data by score
        if (score >= requirement_score):
            scores.append(score)
            # Prepare the date to neural nets
            for data in game:
                if data[1] == 1:
                    label = [0, 1]
                else:
                    label = [1, 0]
                training_data.append([data[0], label, score])
    #the training_data must be != []
    # at least the is a game that have score < requirment score
    if(len(training_data)):
        counter = Counter(scores)
        game_status = False
        print('Average accepted score:', mean(scores))
        print('Median score for accepted scores:', median(scores))
        print(counter)
        #if the numbre of games that have score >= max requirement score
        #is greater then the 70% of the played games then we stop the training
        if(counter[max_requ_score] >= game_replay*0.7):
            game_status = True
        #we return the game status (stop learning when it'true)
        #counter each step we save the statistics and data
        return game_status, counter, training_data
    return False,[],training_data

'''this function used to test a model
    the alogrithm plyas game_replay times 
    it take actions based on model
    return all the games counter (informations of the games and scores)'''
def play(model,game_replay=10):
    scores = []
    choices = []
    for _ in range(game_replay):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        prev_action = 0
        for _ in range(5000):
            #env.render()
            if len(prev_obs) == 0:
                action = random.randrange(0, 2)
            else:
                action = np.argmax(model.predict(np.append(prev_obs,prev_action).reshape(-1, len(prev_obs)+1, 1))[0])

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            prev_action = action
            game_memory.append([new_observation, action])
            score += reward
            if done: break
        scores.append(score)
    av = sum(scores) / len(scores)
    print('Average Score:', av)
    print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
    return Counter(scores)


'''this is the training function
    the first step the algorithm start to choose random actions 
    based on the function Qlearning we return the data that have score >= score_init.
    
    fore each step in training_step we increase requirement_score (+ score_inc)
    and change epsilon (it depends how of how much we've done steps)
    we get the data and start training our model 
    
    
    we return the the trained model (that can play and get the max score)'''
def training(score_init, score_inc,max_requ_score,training_step,game_replay):
    #initialze vars
    requirement_score = score_init
    game_replay = game_replay
    data = []
    training_state = False
    step_counter = []
    epsilon=0.3
    for _ in range(training_step):
        train_data = []
        # stop training
        if (training_state): break
        #change epsilon value
        if(_<5): epsilon = 0.1
        elif(_<8): epsilon = 0.4
        elif(_<11): epsilon = 0.6
        elif (_ < 13): epsilon = 0.8
        else: epsilon = 1
        # first we need our algorithm to play randomly
        if(_ == 0):
            training_state,step_counter,data =  Qlearning(0, requirement_score, max_requ_score, game_replay=game_replay)
        else:
            #increse the requ scor
            requirement_score += score_inc
            # we have to make sure that the requirement score will never be grater the max
            if (requirement_score >= max_requ_score):
                requirement_score = max_requ_score
            #start Play
            training_state,step_counter, new_data = Qlearning(epsilon,requirement_score,max_requ_score,model, game_replay)
            #combaine the prevouis and the new data
            #we need the old data cuz maybe it containe an imporant score
            data = data+new_data
        print("-" * 100 + " \n this {} ".format(_, ))
        #we get the training data (the data that have score >=  requirement score)
        train_data = precessing_data(data, requirement_score)
        tf.reset_default_graph()
        if(len_data):
            #we save the steps in pickle files so we can monitor later the progress of our algothim
            pickle.dump(step_counter, open("scores_"+str(_)+".p", "wb"))
            # start train our new model
            model = train_model(train_data)
        #if the trining data is [] we need to minimize the requirement score
        else:
            requirement_score = requirement_score/2
    return model

#from the all data we return the data that have score >= req_score
def precessing_data(data, req_sc):
    processed_data = []
    for d in data:
        if(d[2]>=req_sc):
            processed_data.append([d[0],d[1]])
    return processed_data

#start train our model and save it
model = training(50,10,200,16,1000)
model.save("CartPole-v0.model")
# we test out model
scores=play(model,game_replay=100)
print(scores)
pickle.dump(scores, open("final_scores.p", "wb"))