import neat
import gym
import slimevolleygym
import numpy as np
import pygame
import pickle
import os
import os
from PIL import Image
import imageio
import visualize  # Import the NEAT-Python visualization module
import math

os.environ['PYGLET_SHADOW_WINDOW'] = '0'
os.environ['PYTHONUNBUFFERED'] = '1'

# Set training parameters
MAX_ITER = 500000            # Max number of generations
TEST_INTERVAL = 150        # Interval for testing/rendering the best genome
CURRENT_GENERATION = 0

# Initialize pygame to capture keyboard input
pygame.init()

# Helper function to capture frames for GIF recording
def capture_frame(env):
    """Capture the current frame from the environment"""
    return Image.fromarray(env.render(mode='rgb_array'))

def create_network_visualization(config, genome, filename):
    """Create network visualization using NEAT's visualization module"""
    if not os.path.exists('network_visualizations'):
        os.makedirs('network_visualizations')

    # Define node names for better visualization
    node_names = {
        -1: 'player_x', -2: 'player_y', -3: 'player_vx',
        -4: 'player_vy', -5: 'ball_x', -6: 'ball_y',
        -7: 'ball_vx', -8: 'ball_vy', -9: 'opponent_x',
        -10: 'opponent_y', -11: 'opponent_vx', -12: 'opponent_vy',
        0: 'move_left', 1: 'move_right', 2: 'jump'
    }

    # Add activation function to node names for hidden nodes
    for node_id, node in genome.nodes.items():
        if node_id not in node_names:  # If it's a hidden node
            node_names[node_id] = f'{node_id}\n{node.activation}'

    # Define node colors
    node_colors = {
        -1: 'lightblue', -2: 'lightblue', -3: 'lightblue',
        -4: 'lightblue', -5: 'lightblue', -6: 'lightblue',
        -7: 'lightblue', -8: 'lightblue', -9: 'lightblue',
        -10: 'lightblue', -11: 'lightblue', -12: 'lightblue',
        0: 'lightgreen', 1: 'lightgreen', 2: 'lightgreen'
    }

    # Create the visualization
    return visualize.draw_net(config, genome, 
                            filename=os.path.join('network_visualizations', filename),
                            node_names=node_names,
                            node_colors=node_colors,
                            fmt='png')

def eval_genomes(genomes, config):
    env = gym.make("SlimeVolley-v0")
    BALL_RADIUS = 0.5
    SLIME_RADIUS = 1.5
    TOTAL_RADIUS = BALL_RADIUS + SLIME_RADIUS  # Combined radius for collision

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        total_reward = 0

        for episode in range(20):
            observation = env.reset()
            done = False
            episode_reward = 0
            hit_count = 0
            time_with_ball = 0
            
            while not done:
                action_values = net.activate(observation)
                move_left = np.clip(action_values[0], 0, 1)
                move_right = np.clip(action_values[1], 0, 1)
                jump = np.clip(action_values[2], 0, 1)
                action = [move_left, move_right, jump]

                observation, reward, done, info = env.step(action)
                
                # Get state values
                state = info['state']
                agent_x = state[0]  # Agent's x position
                agent_y = state[1]  # Agent's y position
                ball_x = state[4]   # Ball's x position
                ball_y = state[5]   # Ball's y position
                ball_vx = state[6]  # Ball velocity in x direction
                ball_vy = state[7]  # Ball velocity in y direction

                # Calculate hit rewards
                hit_reward = 0
                hit_type = "NO HIT"

                if reward > 0:  # Ball was hit
                    hit_count += 1
                    # Calculate 2D distance between ball and agent centers
                    dx = ball_x - agent_x
                    dy = ball_y - agent_y
                    distance = math.sqrt(dx*dx + dy*dy)

                    # Check if hit is within valid collision range
                    if distance <= TOTAL_RADIUS:
                        # Determine hit quality based on relative x position
                        if dx < 0:  # Front-side hit (ball is to the left of agent)
                            if ball_vx < -0.5:  # Strong hit towards opponent
                                hit_reward = 300.0
                                hit_type = "STRONG FRONT HIT"
                            elif ball_vx < 0:  # Weak hit towards opponent
                                hit_reward = 150.0
                                hit_type = "WEAK FRONT HIT"
                        else:  # Back-side hit (ball is to the right of agent)
                            hit_reward = -200.0
                            hit_type = "BACK HIT"
                    
                    print(f"Hit #{hit_count} - Type: {hit_type}")
                    print(f"Distance: {distance:.3f}, dx: {dx:.3f}, dy: {dy:.3f}")
                    print(f"Ball VX: {ball_vx:.3f}")

                # Time penalty when ball is on agent's side
                time_penalty = 0
                if ball_x > 0:  # Ball on agent's side
                    time_with_ball += 1
                    time_penalty = -1.0 * (time_with_ball / 100)
                else:
                    time_with_ball = 0

                # Ball tracking reward
                tracking_reward = 0
                if ball_x > 0:  # Ball is on agent's side
                    dx = ball_x - agent_x
                    dy = ball_y - agent_y
                    distance = math.sqrt(dx*dx + dy*dy)
                    # Use combined radius for tracking reward scaling
                    tracking_reward = 5.0 * (1.0 - min(distance, TOTAL_RADIUS) / TOTAL_RADIUS)

                # Height-based jumping reward
                jumping_reward = 0
                if ball_x > 0 and ball_y > agent_y and action[2] > 0.5:
                    optimal_jump_height = min(TOTAL_RADIUS, ball_y - agent_y)
                    jumping_reward = 5.0 * (1.0 - optimal_jump_height/TOTAL_RADIUS)

                # Scoring rewards
                scoring_reward = 1000.0 if info.get('ale.otherScore', 0) > 0 else 0.0
                losing_penalty = -200.0 if reward < 0 else 0.0

                # Combine all rewards
                combined_reward = (
                    hit_reward +
                    scoring_reward +
                    losing_penalty +
                    tracking_reward +
                    jumping_reward +
                    time_penalty
                )
                
                episode_reward += combined_reward

            total_reward += episode_reward

        genome.fitness = total_reward / 5

    env.close()


# Modified capture_frame function with error handling
def capture_frame(env):
    """Capture the current frame from the environment with error handling"""
    try:
        # First try the standard method
        return Image.fromarray(env.render(mode='rgb_array'))
    except (AttributeError, pyglet.gl.GLException) as e:
        try:
            # Fallback method using alternative rendering
            if not hasattr(env, 'viewer') or env.viewer is None:
                env.render(mode='human')
            
            # Get the viewer's dimensions
            width = env.viewer.width
            height = env.viewer.height
            
            # Create a frame buffer
            buffer = (GLubyte * (3 * width * height))(0)
            
            # Read the pixels
            glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer)
            
            # Convert buffer to numpy array
            arr = np.frombuffer(buffer, dtype=np.uint8)
            arr = arr.reshape(height, width, 3)
            
            # Flip the image vertically (OpenGL convention)
            arr = np.flipud(arr)
            
            return Image.fromarray(arr)
        except Exception as e2:
            print(f"Warning: Failed to capture frame: {str(e2)}")
            # Return a blank frame as fallback
            return Image.new('RGB', (400, 300), color='white')

# Modified render_best_genome function
def render_best_genome(winner, config, num_episodes=1, save_gif=True, generation=None):
    env = gym.make("SlimeVolley-v0")
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    # Create directories if they don't exist
    os.makedirs('gameplay_gifs', exist_ok=True)
    os.makedirs('network_visualizations', exist_ok=True)
    
    # Create network visualization
    network_filename = f'network_gen_{generation}' if generation is not None else 'network_final'
    try:
        create_network_visualization(config, winner, network_filename)
    except Exception as e:
        print(f"Warning: Failed to create network visualization: {str(e)}")

    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        frames = []
        
        try:
            while not done:
                if save_gif:
                    frame = capture_frame(env)
                    if frame is not None:
                        frames.append(frame)
                else:
                    env.render()
                
                # Get network outputs
                action_values = net.activate(observation)
                action = [
                    np.clip(action_values[0], 0, 1),
                    np.clip(action_values[1], 0, 1),
                    np.clip(action_values[2], 0, 1)
                ]
                
                observation, reward, done, info = env.step(action)
                total_reward += reward
                
                # Optional delay for visualization
                pygame.time.wait(10)
            
            print(f"Episode {episode + 1} finished with reward: {total_reward}")
            
            # Save the GIF for this episode
            if save_gif and frames:
                gif_filename = f"gameplay_gifs/episode_{episode+1}_gen_{generation if generation else 'final'}.gif"
                try:
                    imageio.mimsave(gif_filename, frames, fps=30)
                    print(f"Saved gameplay GIF: {gif_filename}")
                except Exception as e:
                    print(f"Warning: Failed to save GIF: {str(e)}")
                    
        except Exception as e:
            print(f"Warning: Episode {episode + 1} failed: {str(e)}")
            continue
        
    env.close()


# Modified VisualizeBestGenomeReporter class
class VisualizeBestGenomeReporter(neat.reporting.BaseReporter):
    def __init__(self, config, save_interval=10):
        self.config = config
        self.generation = 0
        self.save_interval = save_interval
        self.stats = neat.StatisticsReporter()
        
    def post_evaluate(self, config, population, species, best_genome):
        self.generation += 1
        
        # Update statistics
        try:
            self.stats.post_evaluate(config, population, species, best_genome)
        except Exception as e:
            print(f"Warning: Failed to update statistics: {str(e)}")
        
        # Render best genome at intervals
        if self.generation % TEST_INTERVAL == 0:
            print(f"\nRendering the best genome at generation {self.generation}...")
            try:
                render_best_genome(best_genome, self.config, num_episodes=1, 
                                 save_gif=True, generation=self.generation)
            except Exception as e:
                print(f"Warning: Failed to render best genome: {str(e)}")
            
            # Create visualization plots
            try:
                visualize.plot_stats(self.stats, ylog=False, 
                                   filename=f'fitness_history_gen_{self.generation}.svg')
                visualize.plot_species(self.stats, 
                                     filename=f'speciation_gen_{self.generation}.svg')
            except Exception as e:
                print(f"Warning: Failed to create visualization plots: {str(e)}")
        
        # Save best genome at intervals
        if self.generation % self.save_interval == 0:
            try:
                with open(f"best_genome_gen_{self.generation}.pkl", "wb") as f:
                    pickle.dump(best_genome, f)
                print(f"Saved best genome at generation {self.generation}")
            except Exception as e:
                print(f"Warning: Failed to save best genome: {str(e)}")

    def complete_extinction(self):
        print("Population extinct.")

    def species_stagnant(self, sid, species):
        print(f"Species {sid} is stagnant with {len(species.members)} members.")

def run_neat(config_file):
    global CURRENT_GENERATION
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    
    def eval_genomes_with_generation(genomes, config):
        global CURRENT_GENERATION
        eval_genomes(genomes, config)
        CURRENT_GENERATION += 1

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
    winner = p.run(eval_genomes_with_generation, MAX_ITER)

    # Save the best genome
    with open("best_genome_final.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Final best genome saved to 'best_genome_final.pkl'.")

    # Create final visualizations
    print("Creating final visualizations...")
    visualize.plot_stats(stats, ylog=False, filename='fitness_history_final.svg')
    visualize.plot_species(stats, filename='speciation_final.svg')

    # Render the final best genome's gameplay
    print("Rendering the final best genome...")
    render_best_genome(winner, config, num_episodes=3, save_gif=True)

if __name__ == '__main__':
    config_path = 'neat-config-slimevolley.ini'
    run_neat(config_path)
