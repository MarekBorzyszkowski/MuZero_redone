import cv2
import numpy as np
import pathlib
from vizdoom import DoomGame, Mode, ScreenResolution, Button
import datetime
import torch


from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (320, 60, 80)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(3))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(1))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = None  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = True
        self.max_moves = 2500  # Maximum number of moves if game is not finished before
        self.num_simulations = 10  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 5  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

        # Residual Network
        self.downsample = "resnet"  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 8  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 2  # Number of channels in policy head
        self.resnet_fc_reward_layers = [8]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [8]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [8]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 10
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [8]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [8]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = int(1000e3)  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 16  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 500  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 350e3



        ### Replay Buffer
        self.replay_buffer_size = int(1e4)  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 10  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = True



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it
        # fmt: on

    def visit_softmax_temperature_fn(self, trained_steps):
        if trained_steps < 500e3:
            return 1.0
        elif trained_steps < 750e3:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    def __init__(self, seed=None):
        self.game = DoomGame()
        self.game.load_config("basic.cfg")
        self.game.set_window_visible(False)
        # self.game.set_doom_scenario_path("basic.wad")
        if seed is not None:
            self.game.set_seed(seed)
        self.game.init()
        self.observation_shape = (320, 60, 80)
        self.action_space = [  # Definicja akcji
            [1, 0, 0],  # Strzał
            [0, 1, 0],  # Ruch w lewo
            [0, 0, 1],  # Ruch w prawo
        ]

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action: Action from the action_space to take.

        Returns:
            The new observation, the reward, and a boolean indicating if the game has ended.
        """
        reward = self.game.make_action(self.action_space[action])
        done = self.game.is_episode_finished()
        observation = self.get_observation() if not done else np.zeros(self.observation_shape, dtype=np.float32)
        return observation, reward, done

    def legal_actions(self):
        """
        Returns the legal actions available at the current step.
        """
        return list(range(len(self.action_space)))

    def reset(self):
        """
        Reset the game for a new episode.

        Returns:
            Initial observation of the game.
        """
        self.game.new_episode()
        return self.get_observation()

    def close(self):
        """
        Properly close the game.
        """
        self.game.close()

    def render(self):
        """
        Display the current game observation.
        """
        state = self.game.get_state().screen_buffer
        # cv2.imshow("ViZDoom", cv2.cvtColor(state, cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)

    def get_observation(self):
        """
        Retrieve and preprocess the current observation from the game.

        Returns:
            Processed observation as a NumPy array.
        """
        state = self.game.get_state().screen_buffer
        observation = cv2.resize(state, (80, 60), interpolation=cv2.INTER_AREA)
        observation = np.asarray(observation, dtype=np.float32) / 255.0
        observation = np.moveaxis(observation, -1, 0)  # Rearrange channels to (C, H, W)
        return observation
