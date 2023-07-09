import math
from typing import Any, Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from flax.linen import initializers
import flax.linen as nn
from flax.linen.activation import tanh
from flax.linen.linear import default_kernel_init
from flax.linen.module import compact
from tensorflow_probability.substrates import jax as tfp
from evosax import NetworkMapper
import gymnax

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = jax.Array

def get_model_ready(rng, config, speed=False):
    """Instantiate a model according to obs shape of environment."""
    # Get number of desired output units
    env, env_params = gymnax.make(config.env_name, **config.env_kwargs)

    model = None
    # Instantiate model class (flax-based)
    if config.train_type == "ES":
        model = NetworkMapper[config.network_name](
            **config.network_config, num_output_units=env.num_actions
        )
    elif config.train_type == "PPO":
        if config.network_name == "Categorical-MLP":
            model = CategoricalSeparateMLP(
                **config.network_config, num_output_units=env.num_actions
            )
        elif config.network_name == "Gaussian-MLP":
            model = GaussianSeparateMLP(
                **config.network_config, num_output_units=env.num_actions
            )
        elif config.network_name == "Gaussian-LSTM":
            model = GaussianSeparateRNN(
                **config.network_config, num_output_units=env.num_actions
            )
        elif config.network_name == "Categorical-LSTM":
            model = CategoricalSeparateRNN(
                **config.network_config, num_output_units=env.num_actions
            )
        elif config.network_name == "Categorical-STPN":
            model = CategoricalSeparateSTPN(
                **config.network_config, num_output_units=env.num_actions
            )

    # Only use feedforward MLP in speed evaluations!
    if speed and config.network_name == "LSTM":
        model = NetworkMapper["MLP"](
            num_hidden_units=64,
            num_hidden_layers=2,
            hidden_activation="relu",
            output_activation="categorical"
            if config.env_name != "PointRobot-misc"
            else "identity",
            num_output_units=env.num_actions,
        )

    # Initialize the network based on the observation shape
    obs_shape = env.observation_space(env_params).shape
    assert model is not None, "Requested model not found."
    if config.network_name not in ["LSTM", "Gaussian-LSTM", "Categorical-LSTM", "Categorical-STPN"] or speed:
        params = model.init(rng, jnp.zeros(obs_shape), rng=rng)
    elif config.network_name in ["Gaussian-LSTM", "Categorical-LSTM"]:
        params = model.init(
            rng, jnp.zeros(obs_shape), model.initialize_carry(rng), rng=rng
        )
    elif config.network_name in ["Categorical-STPN"]:
        params = model.init(
            rng, jnp.zeros(obs_shape), model.initialize_carry(rng, obs_shape), rng=rng
        )
    else:
        params = model.init(
            rng, jnp.zeros(obs_shape), model.initialize_carry(), rng=rng
        )
    return model, params


def default_mlp_init(scale=0.05):
    return nn.initializers.uniform(scale)


class CategoricalSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    model_name: str = "separate-mlp"
    flatten_2d: bool = False  # Catch case
    flatten_3d: bool = False  # Rooms/minatar case

    @nn.compact
    def __call__(self, x, rng):
        # Flatten a single 2D image
        if self.flatten_2d and len(x.shape) == 2:
            x = x.reshape(-1)
        # Flatten a batch of 2d images into a batch of flat vectors
        if self.flatten_2d and len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)

        # Flatten a single 3D image
        if self.flatten_3d and len(x.shape) == 3:
            x = x.reshape(-1)
        # Flatten a batch of 3d images into a batch of flat vectors
        if self.flatten_3d and len(x.shape) > 3:
            x = x.reshape(x.shape[0], -1)
        x_v = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_critic + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        logits = nn.Dense(
            self.num_output_units,
            bias_init=default_mlp_init(),
        )(x_a)
        # pi = distrax.Categorical(logits=logits)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi


class GaussianSeparateMLP(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-mlp"

    @nn.compact
    def __call__(self, x, rng):
        x_v = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_critic + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                name=self.prefix_actor + "_fc_1",
                bias_init=default_mlp_init(),
            )(x)
        )
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_actor + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        mu = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_mu",
            bias_init=default_mlp_init(),
        )(x_a)
        log_scale = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_scale",
            bias_init=default_mlp_init(),
        )(x_a)
        scale = jax.nn.softplus(log_scale) + self.min_std
        pi = tfp.distributions.MultivariateNormalDiag(mu, scale)
        return v, pi
    

class GaussianSeparateRNN(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-rnn"

    @nn.compact
    def __call__(self, x, state, rng):
        state, x_rec = nn.LSTMCell(
            bias_init=nn.initializers.uniform(scale=0.05),
            kernel_init=jax.nn.initializers.lecun_normal(),
        )(state, x)

        x_v = x_rec
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = x_rec
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_actor + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        mu = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_mu",
            bias_init=default_mlp_init(),
        )(x_a)
        log_scale = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor + "_fc_scale",
            bias_init=default_mlp_init(),
        )(x_a)
        scale = jax.nn.softplus(log_scale) + self.min_std
        pi = tfp.distributions.MultivariateNormalDiag(mu, scale)
        return v, pi, state

    def initialize_carry(self, rng, init_fn=initializers.zeros_init()):
        """Initialize the RNN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        batch_dims: a tuple providing the shape of the batch dimensions.
        size: the size or number of features of the memory.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        return init_fn(key1, self.num_hidden_units), init_fn(key2, self.num_hidden_units)


class CategoricalSeparateRNN(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-rnn"

    @nn.compact
    def __call__(self, x, state, rng):
        state, x_rec = nn.LSTMCell(
            bias_init=nn.initializers.uniform(scale=0.05),
            kernel_init=jax.nn.initializers.lecun_normal(),
        )(state, x)

        x_v = x_rec
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = x_rec
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_actor + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        logits = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor  + f"_fc_a",
            bias_init=default_mlp_init(),
        )(x_a)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi, state

    def initialize_carry(self, rng, init_fn=initializers.zeros_init()):
        """Initialize the RNN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        batch_dims: a tuple providing the shape of the batch dimensions.
        size: the size or number of features of the memory.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        return init_fn(key1, self.num_hidden_units), init_fn(key2, self.num_hidden_units)


class STPNCell(nn.RNNCellBase):
  r"""LSTM cell.

  The mathematical definition of the cell is as follows

  .. math::
      \begin{array}{ll}
      h_t = \tanh((W + S_t)[x_t, h_{t-1}] + b) \\
      S_t = \lambda S_{t-1} + \gamma h_t x_t
      \end{array}

  where x is the input, h is the output of the previous time step, and S is
  the synaptic memory.

  Attributes:
    activation_fn: activation function used for output and memory update
      (default: tanh).
    kernel_init: initializer function for the kernels that transform
      the input (default: lecun_normal).
    recurrent_kernel_init: initializer function for the kernels that transform
      the hidden state (default: initializers.orthogonal()).
    bias_init: initializer for the bias parameters (default: initializers.zeros_init())
    dtype: the dtype of the computation (default: infer from inputs and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
  """
  activation_fn: Callable[..., Any] = tanh
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  recurrent_kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.orthogonal()
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros_init()
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32

  @compact
  def __call__(self, carry, inputs):
    r"""A long short-term memory (LSTM) cell.

    Args:
      carry: the hidden state of the LSTM cell,
        initialized using `STPNCell.initialize_carry`.
      inputs: an ndarray with the input for the current time step.
        All dimensions except the final are considered batch dimensions.

    Returns:
      A tuple with the new carry and the output.
    """
    h, s = carry
    hidden_features = h.shape[-1]
    init_k = 1 / math.sqrt(hidden_features)
    b = self.param('b', initializers.uniform(2*init_k), (hidden_features,),
                    self.param_dtype) - init_k
    # recurrent and external inputs treated separately
    wh = self.param('wh',
                    initializers.uniform(2*init_k),
                    (hidden_features, hidden_features),
                    self.param_dtype) - init_k
    wi = self.param('wi',
                    initializers.uniform(2*init_k),
                    (hidden_features, jnp.shape(inputs)[-1]),
                    self.param_dtype) - init_k
    Gi = wi + s[..., :-hidden_features]
    Gh = wh + s[..., -hidden_features:]
    y = (Gh @ h[..., None] + Gi @ inputs[..., None]).squeeze(-1)
    norm = jnp.linalg.norm(jnp.concatenate([Gi, Gh], axis=-1), axis=-1, keepdims=True)
    new_h = jnp.tanh(y/norm.squeeze(-1) + b)

    l = self.param("l", initializers.uniform(1), (hidden_features, hidden_features + jnp.shape(inputs)[-1]), self.param_dtype)
    g = self.param("g", initializers.uniform(2*0.001*init_k), (hidden_features, hidden_features + jnp.shape(inputs)[-1]), self.param_dtype) - 0.001*init_k
    # separate operations and unsqueeze, because original h might not be batched, but if input is batched, so will be h. 
    # Also there is the chance for both x and h not to be batched.
    hx = new_h[..., None] @ inputs[..., None, :]
    hh = new_h[..., None] @ h[..., None, :] 
    new_s = l * s / norm + g * jnp.concatenate([hx, hh], axis=-1)
    return (new_h, new_s), new_h

  def initialize_carry(self, rng, obs_shape, init_fn=initializers.zeros_init()):
    """Initialize the STPN cell carry.

    Args:
      rng: random number generator passed to the init_fn.
      obs_shape: a tuple providing the shape of the observations.
      init_fn: initializer function for the carry.
    Returns:
      An initialized carry for the given STPN cell.
    """
    # the second memory here is synaptic memory, which should have as second dim hidden_size + input_size
    assert len(obs_shape) == 1
    key1, key2 = jax.random.split(rng)
    return init_fn(key1, self.num_hidden_units), init_fn(key2, (self.num_hidden_units, self.num_hidden_units + obs_shape[-1]))



class CategoricalSeparateSTPN(nn.Module):
    """Split Actor-Critic Architecture for PPO."""

    num_output_units: int
    num_hidden_units: int
    num_hidden_layers: int
    prefix_actor: str = "actor"
    prefix_critic: str = "critic"
    min_std: float = 0.001
    model_name: str = "separate-rnn"

    @nn.compact
    def __call__(self, x, state, rng):
        state, x_rec = STPNCell(
            bias_init=nn.initializers.uniform(scale=0.05),
            kernel_init=jax.nn.initializers.lecun_normal(),
        )(state, x)

        x_v = x_rec
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_v = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_critic + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_v)
            )
        
        v = nn.Dense(
            1,
            name=self.prefix_critic + "_fc_v",
            bias_init=default_mlp_init(),
        )(x_v)

        x_a = x_rec
        # Loop over rest of intermediate hidden layers
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    name=self.prefix_actor + f"_fc_{i+1}",
                    bias_init=default_mlp_init(),
                )(x_a)
            )
        logits = nn.Dense(
            self.num_output_units,
            name=self.prefix_actor  + f"_fc_a",
            bias_init=default_mlp_init(),
        )(x_a)
        pi = tfp.distributions.Categorical(logits=logits)
        return v, pi, state

    def initialize_carry(self, rng, obs_shape, init_fn=initializers.zeros_init()):
        """Initialize the STPN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        obs_shape: a tuple providing the shape of the observations.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given STPN cell.
        # """
        return STPNCell.initialize_carry(self, rng, obs_shape, init_fn)