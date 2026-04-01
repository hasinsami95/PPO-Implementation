import gym
import numpy as np
from PPO_func import Agent
from utils import plot_curve

if __name__=='__main__':
    env=gym.make('CartPole-v0')
    N=20
    batch_size=5
    num_epoch=4
    alpha=0.0003

    agent=Agent(num_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, num_epoch=num_epoch,
                in_dim=env.observation_space.shape)

    num_games=150 #nuber of episodes 
    figure = 'plot/performance_curve.png'

    best_score=env.reward_range[0]
    score_history=[]
    train_iters=0
    avg_score=0
    num_steps=0

    for i in range(num_games):
        observation,info=env.reset()
        end=False
        score=0
        while not end:
            #print(observation)
            #exit()
            action, prob, val=agent.select_action(observation)
            #print(env.step(action))
            observation_, reward, end, info, empty=env.step(action)
            num_steps+=1
            score+=reward ## reward is preset in the environment for each action
            agent.remember(observation, action, prob, val, reward, end)

            if num_steps%N==0:
                agent.train()
                train_iters+=1
            observation=observation_    ## Set current state to the new state
        score_history.append(score)
        avg_score=np.mean(score_history[-100:]) ## average of previous 100 games

        if avg_score>best_score:
            best_score=avg_score
            agent.save_models()

        print('episode', i, 'score', score, 'avg_score', avg_score, 'time_steps', num_steps, 'train_steps', train_iters)

        x=[i+1 for i in range(len(score_history))]
        plot_curve(x, score_history, figure)


