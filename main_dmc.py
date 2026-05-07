import numpy as np
import torch

import argparse
import os

import tqdm.auto

import utils
import os   
# os.add_dll_directory(r"C:\Users\Administrator\.mujoco\mujoco210\bin")
# import gymnasium as gym
# import pybullet
# import pybullet_envs_gymnasium
# #pybullet.connect(pybullet.DIRECT)
import mujoco
import dm_control
from tqdm import trange
import numpy as np
import torch
import argparse

import gym_dmc as gym
import utils
from algos import ALH , TD3,  SAC
torch.backends.cudnn.benchmark = True
torch.set_num_threads(8)

 
def eval_policy(policy, eid,args, seed, eval_episodes=10):
    eval_env = gym.make(
        eid=eid,
        domain_name=args.domain_name,
        task_name=args.task_name,
        
        visualize_reward=False,
        from_pixels=False,
        frame_skip=args.action_repeat,
        
    )
    
    eval_env.seed(seed + 100)
    options = {}
   
    avg_reward = 0.

    if hasattr(policy, 'forget'):
        policy.forget()
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
       
        while not done:
            action = policy.select_action(np.array(state),deterministic=True)
            p_state = np.array(state)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            if hasattr(policy, 'watch'):  # for ALH
                # adaptive rollout
                policy.watch(p_state, action, reward)

    avg_reward /= eval_episodes
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()  
    parser.add_argument("--policy", default='CWAC')
    parser.add_argument('--domain_name', default="hopper")
    parser.add_argument('--task_name', default="hop")
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--mu", default=0.8, type=float)   # Stochastic pessimistic value sampling
    parser.add_argument("--beta_omega", default=1.0, type=float)  #Uncertainty coefficient 
    parser.add_argument("--beta_xi", default=2.0, type=float)      #TD-error coefficient
    parser.add_argument('--action_repeat', default=2, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=int)
    parser.add_argument("--max_timesteps", default=5e5, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mini_batch_size", default=None, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    
    parser.add_argument("--policy_noise", default=0.2)  # for TD3, ALH
    parser.add_argument("--noise_clip", default=0.5)  
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    parser.add_argument("--device", default='cuda:0', type=str)
    args = parser.parse_args()
   

    # learning rate
    if args.domain_name in ["cheetah"]:
        args.actor_lr = 2e-4
        args.critic_lr = 2e-4
        args.encoder_lr = 2e-4
    else:
        args.actor_lr = 1e-3
        args.critic_lr = 1e-3
        args.encoder_lr = 1e-3

    eid = f"{args.domain_name.capitalize()}-{args.task_name}-v1"

    device = args.device
    start_timesteps = args.start_timesteps
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    file_name = f'{args.policy}_{args.domain_name}_{args.task_name}_{args.seed}'

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.domain_name}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("result_dmc/"):
        os.makedirs("result_dmc/")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make(
        eid=eid,
        domain_name=args.domain_name,
        task_name=args.task_name,
        
        visualize_reward=False,
        from_pixels=False,
        frame_skip=args.action_repeat,
        
    )
    options = {}
    # Set seeds
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {"state_dim": state_dim, "action_dim": action_dim, "max_action": max_action, "discount": args.discount,
              "tau": args.tau, "device": device}
   
 
 
    # Initialize policy
    if args.policy == "SAC":  # no policy Noise
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
 
        policy = SAC.SAC(**kwargs)
    elif args.policy == "CWAC":  # no policy Noise
        import algos.CWAC as CWAC
        kwargs["mu"] = args.mu
        kwargs["beta_omega"] = args.beta_omega
        kwargs["beta_xi"] = args.beta_xi
        policy = CWAC.CWAC(**kwargs)
    
    
    elif args.policy == "TD3":  # gaussian noise
        
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
   
    elif "ALH" in args.policy:
       
        kwargs["hypo_dim"]=64
        kwargs["mini_batch_size"] = args.mini_batch_size
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = ALH.memTD3(**kwargs)
     
 
    
    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim,batch_size=args.batch_size,device=device)

    # Evaluate untrained policy
    evaluations = [eval_policy(policy, eid,args, args.seed)]
    state, done = env.reset(), False
    # state = state[0]
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    if args.save_model:
        policy.save(filename=f'./models/{file_name}')

    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

  
    for t in tqdm.auto.tqdm(range(int(args.max_timesteps)), f"Training {file_name}..."):

        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < start_timesteps:
            action = env.action_space.sample()
        else:
            if   args.policy in ['SAC','CWAC']  :
                action = policy.select_action(np.array(state))                 
            else:
                action = (policy.select_action(np.array(state)) + np.random.normal(0, max_action * args.expl_noise,
                                                                               size=action_dim)).clip(-max_action,
                                                                                                      max_action)


        # Perform action
        next_state, reward, done, _ = env.step(action)
        if args.policy == 'ALH':
            # if is ALH
            policy.watch(state, action, reward)
        done_bool = float(done)
        if done_bool>1:
            done_bool=1
        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= start_timesteps:
            policy.train(replay_buffer, args.batch_size)
        
        if done:
            # Reset environment
            state, done = env.reset(), False
           
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
            if args.policy == 'ALH':
                # if is ALH
                policy.forget()

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:            
            evaluations.append(eval_policy(policy, eid,args, args.seed))
            np.save("result_dmc/"+file_name, evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}_{str(t + 1)}")
       
