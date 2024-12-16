import os
import pickle
import imageio
import jax
import jax.numpy as jnp

import neat
import gym
import slimevolleygym
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

# Set training parameters
POPULATION_SIZE = 128
MAX_ITER = 500000
TEST_INTERVAL = 5

def neat_activation_code(act_fn):
    if callable(act_fn):
        name = act_fn.__name__
        if 'sigmoid' in name:
            return 0
        elif 'tanh' in name:
            return 1
        elif 'relu' in name:
            return 2
        elif 'identity' in name or 'linear' in name:
            return 3
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")
    else:
        if act_fn == 'sigmoid':
            return 0
        elif act_fn == 'tanh':
            return 1
        elif act_fn == 'relu':
            return 2
        elif act_fn == 'identity':
            return 3
        else:
            raise ValueError(f"Unsupported activation function: {act_fn}")

def apply_activation(x, code):
    def do_sigmoid(x):
        return jax.nn.sigmoid(x)
    def do_tanh(x):
        return jax.nn.tanh(x)
    def do_relu(x):
        return jax.nn.relu(x)
    def do_identity(x):
        return x

    return jax.lax.switch(
        code,
        [do_sigmoid, do_tanh, do_relu, do_identity],
        x
    )

def genome_to_jax_network(genome, config):
    ff_net = neat.nn.FeedForwardNetwork.create(genome, config)
    node_evals = ff_net.node_evals
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys

    idx_map = {}
    current_idx = 0
    for k in input_keys:
        idx_map[k] = current_idx
        current_idx += 1

    for o in output_keys:
        if o not in idx_map:
            idx_map[o] = current_idx
            current_idx += 1

    for node_id, _, _, _, _, _ in node_evals:
        if node_id not in idx_map:
            idx_map[node_id] = current_idx
            current_idx += 1

    for o in output_keys:
        if o not in idx_map:
            raise KeyError(f"Output key {o} not found.")

    num_evals = len(node_evals)
    b_list = []
    r_list = []
    a_list = []
    ni_list = []
    max_in_links = max(len(n[5]) for n in node_evals)
    idx_list = []
    w_list = []

    for i, (node_id, act_fn, agg_fn, bias, response, links) in enumerate(node_evals):
        b_list.append(bias)
        r_list.append(response)
        a_list.append(neat_activation_code(act_fn))
        ni_list.append(idx_map[node_id])

        link_ids = [idx_map[in_id] for (in_id, w) in links]
        link_ws = [w for (in_id, w) in links]

        padded_ids = link_ids + [-1]*(max_in_links - len(link_ids))
        padded_ws = link_ws + [0.0]*(max_in_links - len(link_ws))
        idx_list.append(padded_ids)
        w_list.append(padded_ws)

    biases_arr = jnp.array(b_list, dtype=jnp.float32)
    responses_arr = jnp.array(r_list, dtype=jnp.float32)
    activation_codes = jnp.array(a_list, dtype=jnp.int32)
    node_indices = jnp.array(ni_list, dtype=jnp.int32)
    agg_indices_arr = jnp.array(idx_list, dtype=jnp.int32)
    agg_weights_arr = jnp.array(w_list, dtype=jnp.float32)
    output_indices = jnp.array([idx_map[o] for o in output_keys], dtype=jnp.int32)
    num_inputs = len(input_keys)

    @jax.jit
    def forward_fn(inputs):
        out_values = jnp.zeros(current_idx, dtype=jnp.float32)
        out_values = out_values.at[0:num_inputs].set(inputs)

        def compute_node(i, out_vals):
            node_idx = node_indices[i]
            valid_mask = (agg_indices_arr[i] >= 0)
            in_ids = jnp.where(valid_mask, agg_indices_arr[i], 0)
            wts = agg_weights_arr[i] * valid_mask.astype(jnp.float32)
            z = jnp.sum(out_vals[in_ids] * wts)
            z = z * responses_arr[i] + biases_arr[i]
            activated = apply_activation(z, activation_codes[i])
            out_vals = out_vals.at[node_idx].set(activated)
            return out_vals

        out_values = jax.lax.fori_loop(0, num_evals, compute_node, out_values)
        return out_values[output_indices]

    return forward_fn

def eval_genomes(genomes, config):
    env = gym.make("SlimeVolley-v0")
    for genome_id, genome in genomes:
        net = genome_to_jax_network(genome, config)
        total_reward = 0.0
        for episode in range(5):
            observation = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                obs_j = jnp.array(observation, dtype=jnp.float32)
                action_values = net(obs_j)
                move_left = np.clip(action_values[0], 0, 1)
                move_right = np.clip(action_values[1], 0, 1)
                jump = np.clip(action_values[2], 0, 1)
                action = [move_left, move_right, jump]

                observation, reward, done, info = env.step(action)

                ball_x, ball_y = observation[2], observation[3]
                agent_x, agent_y = observation[0], observation[1]
                following_reward = 0.5 * (1 - (agent_x - ball_x))
                vertical_proximity_reward = 10.0 * (1 - abs(agent_y - ball_y)) if ball_y > 0.5 else 0
                hit_ball_reward = 1000.0 if reward > 0 else 0.0
                lose_point_penalty = -10.0 if reward < 0 else 0.0
                jump_reward = 5.0 if jump > 0.5 and ball_y > 0.5 else 0.0
                right_side_preference = 1.0 if agent_x > 0.5 else 0.0

                combined_reward = (
                    reward * 2.0 +
                    following_reward +
                    vertical_proximity_reward +
                    hit_ball_reward +
                    lose_point_penalty +
                    jump_reward +
                    right_side_preference
                )
                episode_reward += combined_reward
            total_reward += episode_reward

        genome.fitness = total_reward / 5.0
    env.close()

def render_best_genome(winner, config, num_episodes=1, gif_filename="best_genome.gif"):
    """
    Render the actual environment simulation into a GIF using env.render(mode='rgb_array').
    Requires a virtual display (e.g., via xvfb-run) when headless.
    """
    env = gym.make("SlimeVolley-v0")
    net = genome_to_jax_network(winner, config)

    frames = []
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0.0
        episode_frames = []

        # Capture at least one frame at the start
        frame = env.render(mode='rgb_array')
        episode_frames.append(frame)

        while not done:
            obs_j = jnp.array(observation, dtype=jnp.float32)
            action_values = net(obs_j)
            move_left = np.clip(action_values[0], 0, 1)
            move_right = np.clip(action_values[1], 0, 1)
            jump = np.clip(action_values[2], 0, 1)
            action = [move_left, move_right, jump]

            observation, reward, done, info = env.step(action)
            total_reward += reward

            # Capture the new frame after the step
            frame = env.render(mode='rgb_array')
            episode_frames.append(frame)

        print(f"Episode {episode + 1} finished with reward: {total_reward}")
        frames.extend(episode_frames)

    env.close()

    if frames:
        imageio.mimsave(gif_filename, frames, fps=30)
        print(f"Saved visualization to {gif_filename}")
    else:
        print("No frames captured. GIF not created.")

class VisualizeBestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, config, save_interval=10):
        self.config = config
        self.generation = 0
        self.save_interval = save_interval

    def post_evaluate(self, config, population, species, best_genome):
        self.generation += 1
        if self.generation % TEST_INTERVAL == 0:
            print(f"\nRendering the best genome at generation {self.generation}...")
            # Render a matplotlib-based GIF, no pyglet
            render_best_genome(best_genome, self.config, num_episodes=1, gif_filename=f"best_genome_gen_{self.generation}.gif")

        if self.generation % self.save_interval == 0:
            with open(f"best_genome_gen_{self.generation}.pkl", "wb") as f:
                pickle.dump(best_genome, f)
            print(f"Saved best genome at generation {self.generation}")

    def complete_extinction(self):
        print("Population extinct.")

    def species_stagnant(self, sid, species):
        print(f"Species {sid} is stagnant with {len(species.members)} members.")

def run_neat(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(VisualizeBestGenomeReporter(config, save_interval=10))

    winner = p.run(eval_genomes, MAX_ITER)

    with open("best_genome_final.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Final best genome saved to 'best_genome_final.pkl'.")

    print("Rendering the final best genome...")
    render_best_genome(winner, config, num_episodes=2, gif_filename="best_genome_final.gif")

if __name__ == '__main__':
    config_path = 'neat-config-slimevolley.ini'
    run_neat(config_path)
