import os
import gym
import tqdm
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import tqdm_config, MovAvg
from tianshou.env import VectorEnv
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import test_episode, gather_info
from tianshou.policy.dist import DiagGaussian

if __name__ == '__main__':
    from net import Net,ActorProb, Critic
else:  # pytest
    from test.continuous.net import Net,ActorProb, Critic


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--seed', type=int, default=1636)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=100)
    parser.add_argument('--collect-per-step', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(args=get_args()):
    torch.set_num_threads(1)  # for poor CPU
    args.num_of_users = 4
    env = gym.make(args.task)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    args.min_action = env.action_space.low[0]
    # train_envs = [gym.make(args.task) for i in range(args.num_of_users)]
    # you can also use tianshou.env.SubprocVectorEnv
    train_envs = [VectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.training_num)]) for i in range(args.num_of_users)]
    # test_envs = gym.make(args.task) # 给global model
    test_envs = VectorEnv(
         [lambda: gym.make(args.task) for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for i in range(args.num_of_users):
        train_envs[i].seed(args.seed)
    test_envs.seed(args.seed)
    # global model
    # global_net = Net(args.layer_num, args.state_shape, device=args.device)
    # global_actor = ActorProb(global_net, args.action_shape,args.max_action).to(args.device)
    # global_critic = Critic(global_net).to(args.device)
    global_actor = ActorProb(
        args.layer_num, args.state_shape, args.action_shape,
        args.max_action, args.device
    ).to(args.device)
    global_critic = Critic(
        args.layer_num, args.state_shape, device=args.device
    ).to(args.device)
    # orthogonal initialization
    for m in list(global_actor.modules()) + list(global_critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    global_optim = torch.optim.Adam(list(
        global_actor.parameters()) + list(global_critic.parameters()), lr=args.lr)
    global_dist = DiagGaussian
    global_policy = PPOPolicy(
        global_actor, global_critic, global_optim, global_dist, args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        action_range=[args.min_action,args.max_action],
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip)
    # local model
    # local_nets = [Net(args.layer_num, args.state_shape, device=args.device) for _ in range(args.num_of_users)]
    local_actors = [ActorProb(
        args.layer_num, args.state_shape, args.action_shape,
        args.max_action, args.device
    ).to(args.device) for i in range(args.num_of_users)]
    local_critics = [Critic(
        args.layer_num, args.state_shape, device=args.device
    ).to(args.device) for i in range(args.num_of_users)]
    # orthogonal initialization
    for i in range(args.num_of_users):
        for m in list(local_actors[i].modules()) + list(local_critics[i].modules()):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.zeros_(m.bias)
    local_optims = [torch.optim.Adam(list(
        local_actors[i].parameters()) + list(local_critics[i].parameters()), lr=args.lr) for i in range(args.num_of_users)]
    local_dists = [DiagGaussian for i in range(args.num_of_users)]
    local_policies = [PPOPolicy(
        local_actors[i], local_critics[i], local_optims[i], local_dists[i], args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        action_range=None,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip) for i in range(args.num_of_users)]
    # collector
    train_collectors = [Collector(
        local_policies[i], train_envs[i], ReplayBuffer(args.buffer_size)) for i in range(args.num_of_users)]
    test_collector = Collector(global_policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ppo')
    writer = SummaryWriter(log_path)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(x):
        return x >= env.spec.reward_threshold

    all_server_costs = []
    for epoch in range(1, 1 + args.epoch):
        # 每个epoch收集N*T数据，然后用B训练M次
        server_costs = []
        with tqdm.tqdm(total=args.step_per_epoch, desc=f'Epoch #{epoch}',
                       **tqdm_config) as t:
            while t.n < t.total:
                # 分发模型
                for i in range(args.num_of_users):
                    for global_param,local_param in zip(global_actor.parameters(),local_actors[i].parameters()):
                        local_param.data.copy_(global_param.data)
                for i in range(args.num_of_users):
                    for global_param,local_param in zip(global_critic.parameters(),local_critics[i].parameters()):
                        local_param.data.copy_(global_param.data)
                # 收集数据,不用梯度
                for i in range(args.num_of_users):
                    result = train_collectors[i].collect(n_episode=args.collect_per_step)
                    losses = local_policies[i].learn(
                        train_collectors[i].sample(0), args.batch_size, args.repeat_per_collect)
                    train_collectors[i].reset_buffer()
                t.update(1)
                # temp_global_param = 0
                for global_param, local_param1, local_param2, local_param3, local_param4 in zip(global_actor.parameters(),
                                                                                  local_actors[0].parameters(),
                                                                                  local_actors[1].parameters(),
                                                                                  local_actors[2].parameters(),local_actors[3].parameters()):
                    global_param.data.copy_((local_param1.data + local_param2.data + local_param3.data+local_param4.data) / args.num_of_users)
                for global_param, local_param1, local_param2, local_param3, local_param4 in zip(global_critic.parameters(),
                                                                                  local_critics[0].parameters(),
                                                                                  local_critics[1].parameters(),
                                                                                  local_critics[2].parameters(),local_critics[3].parameters()):
                    global_param.data.copy_((local_param1.data + local_param2.data + local_param3.data+local_param4.data) / args.num_of_users)
                '''
                for i in range(args.num_of_users):
                    for local_param in zip(local_nets[i].parameters()):
                        temp_global_param += local_param.data
                for global_param in zip(global_net.parameters()):
                    global_param.data.copy_(temp_global_param / args.num_of_users)
                '''
                test_result = test_episode(
                    global_policy, test_collector, test_fn=None,
                    epoch=epoch, n_episode=args.collect_per_step)
                print("test_result:",test_result)
                # 聚合模型
                # print("losses:",losses)
            if t.n <= t.total:
                t.update()
    # trainer
    '''
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.collect_per_step, args.repeat_per_collect,
        args.test_num, args.batch_size, save_fn=save_fn,
        writer=writer)
    '''
    # assert stop_fn(result['best_reward'])
    train_collector.close()
    test_collector.close()
    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        print(f'Final reward: {result["rew"]}, length: {result["len"]}')
        collector.close()


if __name__ == '__main__':
    test_ppo()