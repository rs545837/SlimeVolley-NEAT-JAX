import neat
import gym
import slimevolleygym
import jax
import jax.numpy as jnp
from neat.reporting import StdOutReporter
from neat.population import CompleteExtinctionException
import pickle

# Check if GPU is available
def check_gpu():
    if jax.default_backend() == 'gpu':
        print("Using GPU for computation.")
    else:
        print("Using CPU for computation.")

# Evaluation function for genomes in the population
def eval_genomes(genomes, config):
    env = gym.make("SlimeVolley-v0")
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0
        
        for episode in range(5):
            observation = env.reset()
            done = False
            while not done:
                # Get neural network outputs using JAX arrays
                action_values = jnp.array(net.activate(observation))

                # Use continuous values for left/right/jump movements
                move_left = jax.lax.clamp(-1.0, action_values[0], 1.0)
                move_right = jax.lax.clamp(-1.0, action_values[1], 1.0)
                jump = jax.lax.clamp(-1.0, action_values[2], 1.0)

                action = [move_left, move_right, jump]  # Continuous actions
                
                # Convert JAX array to Python list for compatibility with Gym
                action = jnp.array(action).tolist()
                
                observation, reward, done, info = env.step(action)

                # Additional custom rewards for the agent
                if observation[1] < observation[3]:  # Example: ball on our side
                    reward += 0.1  # Encourage moving toward the ball
                
                total_reward += reward

        genome.fitness = total_reward / 5  # Average reward over episodes

    env.close()

# Function to visualize the best genome in action
def render_best_genome(winner, config, num_episodes=1):
    env = gym.make("SlimeVolley-v0")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()  # Render the environment to visualize it
            
            # Get neural network outputs using JAX arrays
            action_values = jnp.array(net.activate(observation))

            # Use continuous values for left/right/jump movements
            move_left = action_values[0]
            move_right = action_values[1]
            jump = action_values[2]

            action = [move_left, move_right, jump]  # Continuous actions
            
            # Convert JAX array to Python list for compatibility with Gym
            action = jnp.array(action).tolist()
            
            # Step the environment with the chosen action
            observation, reward, done, info = env.step(action)
            total_reward += reward
        
        print(f"Episode {episode + 1} finished with reward: {total_reward}")
    
    env.close()

# Custom Reporter to render the best genome after each generation
class VisualizeBestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, config):
        self.config = config
    
    def post_evaluate(self, config, population, species, best_genome):
        print(f"Rendering best genome of the current generation with fitness {best_genome.fitness}")
        render_best_genome(best_genome, self.config)

# Run the NEAT algorithm
def run_neat(config_file):
    # Load the NEAT configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Check if using GPU
    check_gpu()

    # Create the population
    p = neat.Population(config)

    # Add a reporter to display progress
    p.add_reporter(StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Add custom reporter for visualizing the best genome
    p.add_reporter(VisualizeBestGenomeReporter(config))

    # Run NEAT for up to 50 generations
    winner = None
    try:
        winner = p.run(eval_genomes, 50)
    except CompleteExtinctionException:
        print("Complete extinction occurred in the population.")

    # Display the winning genome
    print('\nBest genome:\n{!s}'.format(winner))

    # Save the best genome for future use
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

    return winner

if __name__ == '__main__':
    # Path to the NEAT configuration file
    config_path = 'neat-config-slimevolley.ini'

    # Run NEAT to evolve agents to play SlimeVolley
    winner = run_neat(config_path)

    # Render the gameplay of the best genome
    print("Rendering the best genome in action...")
    render_best_genome(winner, neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                           neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path))
