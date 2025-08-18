import os
import gymnasium as gym
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from distutils.util import strtobool
import argparse
from torch.utils.tensorboard import SummaryWriter

from nmt_cmr_parallels.data.sequence_data import create_pretrained_semantic_embedding
from nmt_cmr_parallels.rl.gym_env import SequenceEnv
from nmt_cmr_parallels.rl.ppo_agent import Agent
from nmt_cmr_parallels.models.encdec_recall_model import EncoderDecoderRecallmodel
from nmt_cmr_parallels.utils.checkpoint_utils import save_rl_agent, load_rl_agent, load_recall_model

DEFAULT_DATA_PATH = os.path.expanduser(os.path.join('~', '.seq_nlp_data'))

def make_env(seed, sequence_length=16, seq_tokens=False, timeout=300):
    def thunk():
        env = SequenceEnv(vocab_source='peers_vocab.json', sequence_length=sequence_length, seq_tokens=seq_tokens,timeout=timeout)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def train_agent(seq_tokens=True,
                sequence_length=16,
                timeout=300,
                use_attention=False,
                attention_type="luong",
                disable_semantic_embedding=False,
                vocab_size = 500,
                embedding_dim=300,
                hidden_dim=50,
                data_dir=DEFAULT_DATA_PATH,
                checkpoint_load_path=None,
                actor_load_path=None,
                rnn_mode="LSTM",
                model_type="recurrent_layer",
                save_freq=2000,
                total_timesteps=500000, 
                learning_rate=2.5e-4, 
                num_envs=4, 
                num_steps=128,
                gamma=0.99, 
                gae_lambda=0.95, 
                minibatch_size=64,
                batch_size=64,
                num_minibatches=4, 
                update_epochs=4, 
                norm_adv=True, 
                clip_coef=0.2, 
                clip_vloss=True, 
                ent_coef=0.01, 
                exp_name='run_0',
                vf_coef=0.5, 
                max_grad_norm=0.5, 
                target_kl=None, 
                seed=1234,
                cuda=True, **kwargs):

    writer = SummaryWriter(f"{exp_name}")
    checkpoint_path = os.path.join(exp_name, "agent_checkpoint.pt")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(seed+i, sequence_length=sequence_length, seq_tokens=seq_tokens,timeout=timeout) for i in range(num_envs)])
    os.makedirs(data_dir,exist_ok=True)
    vocab = envs.envs[0].vocab
    vocab_size = len(vocab)
    
    # Extract pretrained semantic embedding from Word2Vec
    pretrained_embedding = None
    if not disable_semantic_embedding:
        pretrained_embedding = create_pretrained_semantic_embedding(data_dir,embedding_dim)

    # Make sequence model
    # Initialize the model and optimizer
    use_attention = False if attention_type=='none' else True
    if actor_load_path is not None: 
        sequence_model = load_recall_model(actor_load_path,device=device)
        logging.info("Pre-Trained Actor Loaded...")
    else:
        if model_type == 'encoderdecoder':
            sequence_model = EncoderDecoderRecallmodel(vocab_size, embedding_dim, hidden_dim, 
                                            use_attention=use_attention, 
                                            pretrained_embedding=pretrained_embedding,
                                            rnn_mode=rnn_mode,
                                            attention_type=attention_type, 
                                            vocab=vocab,
                                            device=device)

    agent = Agent(envs,  sequence_model,  sequence_length+2, embedding_dim, model_type, device=device)
    if checkpoint_load_path is not None:
        agent = load_rl_agent(checkpoint_path=checkpoint_load_path, agent=agent)
    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    if rnn_mode == "LSTM":
        hidden_state, cell_state = agent.actor.init_hidden(num_envs, cell_state=True)
        hidden_states = torch.zeros((num_steps, num_envs)+hidden_state.squeeze(0).squeeze(0).shape).to(device)
        cell_states = torch.zeros((num_steps, num_envs)+cell_state.squeeze(0).squeeze(0).shape).to(device)
    else:
        hidden_state = agent.actor.init_hidden(num_envs)
        hidden_states = torch.zeros((num_steps, num_envs)+hidden_state.squeeze(0).squeeze(0).shape).to(device)

    encoder_outputs = torch.zeros((num_steps, num_envs, sequence_length+2)+(hidden_state.shape[-1],)).to(device)
    source_seqs = torch.zeros((num_steps, num_envs, sequence_length+2)).to(device)

    global_step = 0
    start_time = time.time()
    new_sequence = True
    next_obs, _ = envs.reset()
    source_sequence = next_obs
    logging.debug(f"Initial observation: {next_obs}")
    next_obs = torch.Tensor(next_obs).to(dtype=torch.int64).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    for update in range(1, num_updates + 1):

        for step in range(num_steps):

            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic2
            with torch.no_grad():
                action, logprob, _, value, hidden_state, encoder_output = agent.get_action_and_value(next_obs, encode=new_sequence, return_states=True)
                values[step] = value.flatten()
            action = action.squeeze(1)
            logprob = logprob.squeeze(1)

            actions[step] = action
            logprobs[step] = logprob
            # if rnn_mode == "LSTM":
            #     hidden_states[step] = hidden_state[0]
            #     cell_states[step] = hidden_state[1]
            # else:
            hidden_states[step] = hidden_state
            encoder_outputs[step] = encoder_output
            source_seqs[step] = torch.tensor(source_sequence)

            if new_sequence:
                new_sequence = False

            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())

            if any(done):
                next_obs, _ = envs.reset()
                source_sequence = next_obs
                new_sequence = True

            logging.debug(f"Action: {action}")
            logging.debug(f"Next observation: {next_obs}")
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(dtype=torch.int64).to(device), torch.Tensor(done).to(device)

            if "episode" in info.keys():
                logging.debug(f"global_step={global_step}, episode_length={info['episode'][0]['l']}, episodic_return={info['episode'][0]['r']}")
                writer.add_scalar("charts/episodic_return", info['episode'][0]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info['episode'][0]["l"], global_step)

            if step % save_freq == 0:
                save_rl_agent(checkpoint_path=checkpoint_path, agent=agent)
                

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(encoder_output, next_obs, source_sequence).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # if rnn_mode == "LSTM":
        #     b_cell_states = cell_states.reshape((-1,) + (1,cell_states.shape[-1]))
        b_hidden_states = hidden_states.reshape((-1,) + (1,hidden_states.shape[-1]))
        b_encoder_outputs = encoder_outputs.reshape((-1,) + encoder_outputs.shape[-2:])
        b_source_seqs = source_seqs.reshape((-1, sequence_length+2))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                # if rnn_mode == "LSTM":
                #     hidden_state_batch = (b_hidden_states[mb_inds], b_cell_states[mb_inds])
                # else:
                hidden_state_batch = b_hidden_states[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds], 
                                                                            prev_hidden_state=hidden_state_batch, 
                                                                            prev_encoder_outputs=b_encoder_outputs[mb_inds],
                                                                            prev_sequences = b_source_seqs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl 
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        logging.info(f"SPS: {sps}")
        logging.info(f"Policy Loss: {pg_loss.item()}, Entropy Loss: {entropy_loss.item()}, Value Loss: {v_loss.item()}")
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, 
                        default=os.path.basename(__file__).rstrip(".py"),
                        help="The name of this experiment.")

    parser.add_argument("--seed", type=int, 
                        default=1,
                        help="Seed of the experiment.")

    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), 
                        default=True, nargs="?", const=True,
                        help="If toggled, CUDA will be enabled by default.")

    parser.add_argument("--save_freq", type=int, 
                        default=2000, 
                        help="Save agent every n steps.")

    parser.add_argument("--checkpoint_load_path", type=str, 
                        default=None,
                        help="Path to a saved RL agent checkpoint.")

    parser.add_argument("--actor_load_path", type=str, 
                        default=None,
                        help="Path to a pre-trained EncoderDecoder actor model.")

    parser.add_argument("--rnn_mode", type=str, 
                        default="LSTM",
                        choices=["LSTM", "GRU"], 
                        help="Select type of recurrent network cell to be used in recall model.")

    parser.add_argument("--model_type", type=str, 
                        default="encoderdecoder",
                        choices=["encoderdecoder"], 
                        help="Select recall model configuration.")

    parser.add_argument("--attention_type", type=str, 
                        default="luong",
                        choices=["bahdanau", "luong", "none"], 
                        help="Select attention mechanism.")

    parser.add_argument("--sequence_length", type=int, 
                        default=16, 
                        help="Word sequence length.")

    parser.add_argument("--seq_tokens", action="store_true", 
                        default=False, 
                        help="Use sequence tokens to denote beginning and end of sequence.")

    parser.add_argument("--timeout", type=int, 
                        default=300, 
                        help="Maximum number of environment steps before episode is ended.")

    parser.add_argument("--disable_semantic_embedding", action="store_true", 
                        default=False, 
                        help="Disable pretrained embedding layer for semantic association (i.e. no GloVe embeddings.).")

    parser.add_argument("--hidden_dim", type=int, 
                        default=50, 
                        help="Dimension of hidden states.")

    parser.add_argument("--embedding_dim", type=int, 
                        default=50, 
                        choices=[50, 100, 200, 300],
                        help="Dimension of embeddings (Available embedding sizes coincide with GloVe embedding sizes).")

    parser.add_argument("--vocab_size", type=int, 
                        default=5000, 
                        help="Size of testing vocabulary for generated dataset.")
    
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, 
                        default=500000,
                        help="Total timesteps of the experiments.")

    parser.add_argument("--learning-rate", type=float, 
                        default=2.5e-4,
                        help="The learning rate of the optimizer.")

    parser.add_argument("--num-envs", type=int, 
                        default=4,
                        help="The number of parallel game environments.")

    parser.add_argument("--num-steps", type=int, 
                        default=128,
                        help="The number of steps to run in each environment per policy rollout.")

    parser.add_argument("--gamma", type=float, 
                        default=0.99,
                        help="The discount factor gamma.")

    parser.add_argument("--gae-lambda", type=float, 
                        default=0.95,
                        help="The lambda for the general advantage estimation.")

    parser.add_argument("--num-minibatches", type=int, 
                        default=4,
                        help="The number of mini-batches.")

    parser.add_argument("--update-epochs", type=int, 
                        default=4,
                        help="The K epochs to update the policy.")

    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), 
                        default=True, nargs="?", const=True,
                        help="Toggles advantages normalization.")

    parser.add_argument("--clip-coef", type=float, 
                        default=0.2,
                        help="The surrogate clipping coefficient.")

    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), 
                        default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")

    parser.add_argument("--ent-coef", type=float, 
                        default=0.01,
                        help="Coefficient of the entropy.")

    parser.add_argument("--vf-coef", type=float, 
                        default=0.5,
                        help="Coefficient of the value function.")

    parser.add_argument("--max-grad-norm", type=float, 
                        default=0.5,
                        help="The maximum norm for the gradient clipping.")

    parser.add_argument("--target-kl", type=float, 
                        default=None,
                        help="The target KL divergence threshold.")

    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Enable verbose logging mode.")

    parser.add_argument("-d", "--debug", action="store_true", 
                        help="Enable debugging logging mode.")
        
    args = parser.parse_args()
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)

    if args.verbose:
        logging_level = logging.INFO
    elif args.debug:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.WARNING
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
    args_dict = vars(args)

    train_agent(minibatch_size=minibatch_size, batch_size=batch_size, **args_dict)

