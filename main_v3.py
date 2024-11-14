import neat
import gym
import slimevolleygym
import numpy as np
import pygame
import pickle
import os
from PIL import Image 

# Set training parameters
POPULATION_SIZE = 128    # Population size
MAX_ITER = 500000            # Max number of generations
TEST_INTERVAL = 5        # Interval for testing/rendering the best genome

# Initialize pygame to capture keyboard input
pygame.init()

# Evaluation function for genomes in the population
def eval_genomes(genomes, config):
    env = gym.make("SlimeVolley-v0")

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0

        for episode in range(5):
            observation = env.reset()
            done = False
            episode_reward = 0

            while not done:
                action_values = net.activate(observation)

                move_left = np.clip(action_values[0], 0, 1)
                move_right = np.clip(action_values[1], 0, 1)
                jump = np.clip(action_values[2], 0, 1)
                action = [move_left, move_right, jump]

                observation, reward, done, info = env.step(action)

                ball_x, ball_y = observation[2], observation[3]
                agent_x, agent_y = observation[0], observation[1]

                # Primary reward: following the ball
                following_reward = 0.5 * (1 - (agent_x - ball_x))

                # Reward for being close to the ball vertically when it's high
                vertical_proximity_reward = 10.0 * (1 - abs(agent_y - ball_y)) if ball_y > 0.5 else 0

                # Reward for hitting the ball
                hit_ball_reward = 1000.0 if reward > 0 else 0.0

                # Small penalty for losing a point to encourage defensive play
                lose_point_penalty = -10.0 if reward < 0 else 0.0

                # Encourage jumping when the ball is high
                jump_reward = 5.0 if jump > 0.5 and ball_y > 0.5 else 0.0

                # Slight preference for right side, but much smaller than before
                right_side_preference = 1.0 if agent_x > 0.5 else 0.0

                # Combine all rewards
                combined_reward = (
                    reward * 2.0 +  # Base game reward
                    following_reward +
                    vertical_proximity_reward +
                    hit_ball_reward +
                    lose_point_penalty +
                    jump_reward +
                    right_side_preference
                )

                episode_reward += combined_reward

            total_reward += episode_reward

        # Assign fitness as the average reward over episodes
        genome.fitness = total_reward / 5

    env.close()
# Create directory for GIFs
os.makedirs("training_gifs", exist_ok=True)

def render_best_genome(winner, config, num_episodes=1, generation=None):
    env = gym.make("SlimeVolley-v0")
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        frame = 0
        frames = []  # Store frames for GIF

        while not done:
            # Get the pygame surface and convert to PIL Image
            screen = env.render(mode='rgb_array')
            if screen is not None:
                frames.append(Image.fromarray(screen))
            
            frame += 1

            action_values = net.activate(observation)
            move_left = np.clip(action_values[0], 0, 1)
            move_right = np.clip(action_values[1], 0, 1)
            jump = np.clip(action_values[2], 0, 1)
            action = [move_left, move_right, jump]

            observation, reward, done, info = env.step(action)
            total_reward += reward

            pygame.time.wait(10)

            if frame >= 3000:  # Prevent infinite loops
                break

        print(f"Episode {episode + 1} finished with reward: {total_reward}")

        # Save the GIF if we have frames
        if frames:
            suffix = f"gen_{generation}" if generation is not None else "final"
            gif_path = f"training_gifs/gameplay_{suffix}_ep{episode}.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=40,
                loop=0,
                optimize=True
            )
            print(f"Saved gameplay GIF: {gif_path}")

    env.close()

class VisualizeBestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, config, save_interval=10):
        self.config = config
        self.generation = 0
        self.save_interval = save_interval

    def post_evaluate(self, config, population, species, best_genome):
        self.generation += 1
        
        if self.generation % TEST_INTERVAL == 0:
            print(f"\nRendering the best genome at generation {self.generation}...")
            render_best_genome(best_genome, self.config, 
                             num_episodes=1, 
                             generation=self.generation)

        if self.generation % self.save_interval == 0:
            with open(f"best_genome_gen_{self.generation}.pkl", "wb") as f:
                pickle.dump(best_genome, f)
            print(f"Saved best genome at generation {self.generation}")

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

    # Run NEAT for the specified number of generations
    winner = p.run(eval_genomes, MAX_ITER)

    # Save the best genome
    with open("best_genome_final.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Final best genome saved to 'best_genome_final.pkl'.")

    # Render the final best genome's gameplay
    print("Rendering the final best genome...")
    render_best_genome(winner, config, num_episodes=3)

if __name__ == '__main__':
    config_path = 'neat-config-slimevolley.ini'
    run_neat(config_path)
