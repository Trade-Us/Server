import pickle
import time
import numpy as np
import argparse
import re

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--episode', type=int, default=10000,
                      help='number of episode to run')
  parser.add_argument('-b', '--batch_size', type=int, default=32,
                      help='batch size for experience replay')
  parser.add_argument('-i', '--initial_invest', type=int, default=2000000,
                      help='initial investment amount')
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  parser.add_argument('-w', '--weights', type=str, help='a trained model weights')
  args = parser.parse_args()

  maybe_make_dir('weights')
  maybe_make_dir('portfolio_val')

  timestamp = time.strftime('%Y%m%d%H%M')

  data = np.around(get_data())
  train_data = data[:, :]
  test_data = data[:, :]

  env = TradingEnv(train_data, args.initial_invest)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  ## Limit Var
  OBSERVE = 1000
  TRAIN_INTERVAL = 4
  TARGET_UPDATE_INTERVAL = 100
  time_step = 0

  ## Action Collection
  actions = np.zeros((args.episode, env.n_step))

  ## Portfolio
  portfolio_value = []

  if args.mode == 'test':
    # remake the env with test data
    env = TradingEnv(test_data, args.initial_invest)
    # load trained weights
    agent.load(args.weights)
    # when test, the timestamp is same as time when weights was trained
    timestamp = re.findall(r'\d{12}', args.weights)[0]

  for e in range(args.episode):
    state = env.reset()
    state = scaler.transform([state])
    for time in range(env.n_step):
      # Collecting Actions
      actions[e, time] = action

      # count step time
      time_step += 1

      # go step
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      next_state = scaler.transform([next_state])

      # remember steps
      if args.mode == 'train':
        agent.remember(state, action, reward, next_state, done)
      state = next_state

      # episode done
      if done:
        print(actions[e, : ], end="")
        print(" ", end="")
        print("episode: {}/{}, episode end value: {}".format(
          e + 1, args.episode, info['cur_val']))
        portfolio_value.append(info['cur_val']) # append episode end portfolio value
        break

      # train Network
      # train 에 대한 Observe 및 주기 설정
      if args.mode == 'train' and len(agent.memory) > args.batch_size and time_step > OBSERVE:
        agent.replay(args.batch_size)

      # 입실론 감쇄에 대한 Observe 설정
      if args.mode == 'train' and e > OBSERVE:
        agent.deprecate_epsilon

      # target Network 업데이트에 대한 주기 설정
      if args.mode == 'train' and  time_step % TARGET_UPDATE_INTERVAL == 0 and time_step > OBSERVE:
        agent.update_target_model

    # Save Weights    
    if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
      agent.save('weights/{}-dqn.h5'.format(timestamp))

  # save portfolio value history to disk
  with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
    pickle.dump(portfolio_value, fp)