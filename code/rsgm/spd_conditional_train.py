import os
import numpy as np
import pandas as pd
import random as rn

# JAX imports
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random, jit, vmap, grad
import haiku as hk
import optax

os.environ["GEOMSTATS_BACKEND"] = "jax"

# Geomstats imports
import geomstats.backend as gs
import geomstats.geometry.spd_matrices

# SDE-related imports from the framework
# Now, with the fix, we can import Langevin from riemannian_score_sde.sde
from riemannian_score_sde.sde import Langevin, VPSDE # Assuming RSDE is also defined there
# Import BetaSchedule from score_sde.sde for Langevin constructor
from score_sde.schedule import LinearBetaSchedule
from score_sde.models import SDEPushForward
from score_sde.models import get_score_fn # Will be used to wrap our Haiku model
from score_sde.utils import TrainState, replicate # For training state management
from riemannian_score_sde.losses import get_dsm_loss_fn # Loss functions
from score_sde.losses import get_ema_loss_step_fn

# Config-related imports (assuming these exist from previous steps)
# from config_specs.base_config import TrainConfig, SampleConfig, ConfigSpec
from time import strftime
import argparse
from dataclasses import dataclass # For simple config classes


# --- Simple Configuration Classes (defined directly here) ---
# These replace the imports from config_specs.base_config
@dataclass
class TrainConfig:
    job_name: str
    batch_size: int
    optimizer: str
    lr: float
    grad_clip: float
    use_ema: bool
    ema_decay: float
    epochs: int
    log_every: int
    val_every: int
    save_every: int
    loss_type: str
    seed: int
    wandb_project_name: str
    wandb_usr: str

@dataclass
class SampleConfig:
    job_name: str
    num_samples: int
    num_steps: int

@dataclass
class ConfigSpec:
    data_params: dict
    manifold_params: dict
    sde_params: dict
    score_net_params: dict
    train_params: TrainConfig
    sample_params: SampleConfig

# --- Seed Fixing ---
def set_all_seeds_jax(seed_value=42):
    np.random.seed(seed_value)
    rn.seed(seed_value)
    # JAX PRNGKey is managed explicitly

# --- Your make_sine_ts function ---
def make_sine_ts(n_points, start_time=0, dimension=1, n_periods=4, ampl=10):
    sigma = ampl / 10
    time = np.arange(1, n_points + 1)
    series_sine = ampl * np.sin(np.tile(time * (2 * np.pi * n_periods) / n_points, (dimension, 1)).T + start_time) + \
                  sigma * np.random.randn(n_points, dimension)
    table = np.column_stack((time, series_sine))
    columns = ['Time'] + [f'Sine_{i}' for i in range(dimension)]
    ts = pd.DataFrame(table, columns=columns)
    return ts

# --- Data Generation for SPD Matrices ---
def generate_spd_data_from_custom_ts(total_time_steps, num_series, make_ts_func, reg=1e-5, min_t_for_cov=5):
    time_series_data_list = []
    for i in range(num_series):
        ts_df = make_ts_func(total_time_steps, dimension=1, start_time=i * 10)
        time_series_data_list.append(ts_df[f'Sine_0'].values)

    series_np = np.column_stack(time_series_data_list)

    print(series_np.shape)

    spd_matrices = []
    actual_min_t_for_cov = max(min_t_for_cov, num_series + 1)

    for t_end in range(actual_min_t_for_cov, total_time_steps + 1):
        X_data = series_np[:t_end, :]
        mu = np.mean(X_data, axis=0)
        Y = X_data - mu
        C = (Y.T @ Y) / (t_end - 1) + reg * np.eye(num_series)
        
        if np.all(np.linalg.eigvalsh(C) > 1e-7):
            spd_matrices.append(C)
        else:
            pass

    if not spd_matrices:
        raise ValueError("No SPD matrices generated. Check data generation parameters.")
    print(spd_matrices[0])
    print(spd_matrices[-1])
    print(len(spd_matrices))
    return spd_matrices

# --- Dataset (for JAX, it's typically an iterator of numpy arrays) ---
class SPDRawDatasetNP:
    def __init__(self, data_list_np):
        self.data_list_np = data_list_np

    def __len__(self):
        return len(self.data_list_np)

    def __getitem__(self, idx):
        return self.data_list_np[idx] 

def jax_dataloader(dataset_obj, batch_size, shuffle=True, rng_key=None):
    indices = np.arange(len(dataset_obj))
    if shuffle:
        # ИСПРАВЛЕНИЕ: Преобразуем элемент JAX rng_key в Python int для numpy.random.default_rng
        seed_value_for_numpy = None
        if rng_key is not None:
            # Преобразуем JAX Array к Python int
            # .item() работает для одноэлементных JAX Array.
            # Если rng_key[0] - это JAX scalar Array, .item() даст Python int.
            # Если это обычный JAX Array, то np.array() или .tolist()
            if isinstance(rng_key[0], (jax.Array, jnp.ndarray)):
                seed_value_for_numpy = int(rng_key[0].item()) # .item() для скаляров
            else: # Если это уже Python int, float, etc.
                seed_value_for_numpy = int(rng_key[0])

        rng = np.random.default_rng(seed_value_for_numpy) # <--- КОРРЕКТНАЯ ПЕРЕДАЧА СИДА
        rng.shuffle(indices)
    
    for i in range(0, len(dataset_obj), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_data = np.stack([dataset_obj[idx] for idx in batch_indices])
        yield batch_data

# --- SPD U-Net Score Network (Haiku implementation) ---
# These classes will be moved to riemannian_score_sde/models/score_network.py
# (See instructions below for that file)
class GaussianFourierProjection(hk.Module):
  def __init__(self, embed_dim, scale=30., name=None):
    super().__init__(name=name)
    self.W = hk.get_parameter("W", shape=(embed_dim // 2,), dtype=jnp.float32, init=hk.initializers.RandomNormal(scale))
  
  def __call__(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class Dense(hk.Module):
  def __init__(self, output_dim, name=None):
    super().__init__(name=name)
    self.dense = hk.Linear(output_dim)
  
  def __call__(self, x):
    return jnp.expand_dims(jnp.expand_dims(self.dense(x), -2), -2)


class SPDUnetScoreNetwork(hk.Module):
  def __init__(self, manifold_dim, channels=[32, 64], embed_dim=256, activation=jax.nn.swish, name=None):
    super().__init__(name=name)
    self.manifold_dim = manifold_dim
    self.channels = channels
    self.embed_dim = embed_dim
    self.act = activation 

    self.embed = hk.Sequential([
        GaussianFourierProjection(embed_dim=embed_dim, name='gfp_embed'),
        hk.Linear(embed_dim, name='linear_embed')
    ])

    # Input is (batch, 1, dim, dim)
    # Ensure kernel_shape allows for sufficient output size (dim-1 for 2x2, stride 1)
    # For dim=3, this is (B, C, 2, 2)
    # For dim=2, this is (B, C, 1, 1)
    # Be careful if dim is too small!
    # Your example uses kernel 2, stride 1.
    self.conv1 = hk.Conv2D(channels[0], kernel_shape=2, stride=1, with_bias=False, name='conv1')
    self.dense1 = Dense(channels[0], name='dense1')
    self.gnorm1 = hk.GroupNorm(4, create_scale=True, create_offset=True, name='gnorm1')

    # From (B, C0, dim-1, dim-1)
    # For dim=3, this is (B, C0, 2, 2) -> (B, C1, 1, 1)
    # For dim=2, this would fail (1-1=0). So manifold_dim must be >=3 for this setup.
    self.conv2 = hk.Conv2D(channels[1], kernel_shape=2, stride=1, with_bias=False, name='conv2')
    self.dense2 = Dense(channels[1], name='dense2')
    self.gnorm2 = hk.GroupNorm(32, create_scale=True, create_offset=True, name='gnorm2')

    # Decoding layers
    # From (B, C1, 1, 1) -> (B, C0, 2, 2) for dim=3
    self.tconv2 = hk.Conv2DTranspose(channels[0], kernel_shape=2, stride=1, with_bias=False, name='tconv2')
    self.dense7 = Dense(channels[0], name='dense7')
    self.tgnorm2 = hk.GroupNorm(32, create_scale=True, create_offset=True, name='tgnorm2')

    # Skip connection from h1 (B, C0, dim-1, dim-1) to h_after_tconv2 (B, C0, dim-1, dim-1)
    # Concatenated h is (B, 2*C0, dim-1, dim-1)
    # Final layer to output (batch, 1, dim, dim)
    self.tconv1 = hk.Conv2DTranspose(1, kernel_shape=2, stride=1, name='tconv1')


  def __call__(self, x, t):
    x_reshaped = jnp.expand_dims(x, axis=-1) # (batch, 1, dim, dim)

    embed = self.act(self.embed(t))
    print(x_reshaped.shape)

    h1 = self.conv1(x_reshaped)
    h1_dense_out = self.dense1(embed)
    print(f"h1 before add: {h1.shape}, dense1_out shape: {h1_dense_out.shape}")
    h1 += h1_dense_out # Use variable to avoid confusion
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)

    h2 = self.conv2(h1)
    h2_dense_out = self.dense2(embed)
    print(f"h2 before add: {h2.shape}, dense2_out shape: {h2_dense_out.shape}")
    h2 += h2_dense_out # THIS IS THE PROBLEMATIC LINE
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)

    h = self.tconv2(h2)
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)

    h = jnp.concatenate([h, h1], axis=-1) # Skip connection
    
    score_output = self.tconv1(h)
    score = jnp.squeeze(score_output, axis=-1)
    
    score = (score + jnp.swapaxes(score, -1, -2)) / 2.0 # Enforce symmetry
    
    return score


# --- Default Config for SPD with U-Net ---
def create_default_config_spec_spd_unet():
    num_series_for_data = 3 # This will be the dimension of our SPD matrices
    # IMPORTANT: For kernel_shape=2, stride=1, manifold_dim must be >=3
    # (dim-1)-1 >= 1. For dim=2, this leads to (1,1) after conv1, then (0,0) after conv2.
    if num_series_for_data < 3:
        print("WARNING: manifold_dim < 3. U-Net Conv2D layers might reduce dimensions to zero. Setting to 3.")
        num_series_for_data = 3 # Enforce minimum for this U-Net design

    time_steps_for_data = 1000 
    
    matrix_shape = (num_series_for_data, num_series_for_data)
    job_name = f"spd_unet_d{num_series_for_data}_{strftime('%b%d_%H%M%S')}"

    data_params = {
        'total_time_steps': time_steps_for_data,
        'num_series': num_series_for_data,
        'make_ts_func_name': 'make_sine_ts',
        'n_data_points': 500,
        'reg': 1e-4,
        'min_t_for_cov': num_series_for_data + 20,
        'dataset_path': None
    }
    manifold_params = {'name': 'SPD', 'dim': num_series_for_data, 'metric_type': 'affine', 'backend': 'jax'}
    sde_params = {
        'sde_type': 'vpsde',
        'T': 1.0, 
        'beta_min': 0.1, # For BetaSchedule
        'beta_max': 20.0 # For BetaSchedule
    }
    score_net_params = {
        'model_type': 'spd_unet',
        'manifold_dim': num_series_for_data,
        'channels': [32, 64],
        'embed_dim': 256,
        'activation': 'swish' # JAX activation: 'swish', 'elu', 'relu'
    }
    train_params = TrainConfig(
        job_name=job_name,
        batch_size=32, optimizer='Adam', lr=2e-4, grad_clip=jnp.inf, use_ema=True, ema_decay=0.999,
        epochs=100, log_every=5, val_every=50, save_every=50, loss_type='dsm', seed=42,
        wandb_project_name="spd_unet_rsgm_cond",
        wandb_usr = "your_wandb_username"
    )
    sample_params = SampleConfig(job_name=job_name, num_samples=10, num_steps=1000)

    return ConfigSpec(data_params, manifold_params, sde_params, score_net_params, train_params, sample_params)


# --- Main Training Function ---
def main_jax(cfg_spec: ConfigSpec):
    set_all_seeds_jax(cfg_spec.train_params.seed)
    key = random.PRNGKey(cfg_spec.train_params.seed)

    # 1. Data Generation
    print("Generating SPD data from custom time series (numpy)...")
    ts_func = make_sine_ts
    try:
        all_spd_data_np = generate_spd_data_from_custom_ts(
            total_time_steps=cfg_spec.data_params['total_time_steps'],
            num_series=cfg_spec.data_params['num_series'],
            make_ts_func=ts_func,
            reg=cfg_spec.data_params['reg'],
            min_t_for_cov=cfg_spec.data_params['min_t_for_cov']
        )
    except ValueError as e:
        print(f"Error generating data: {e}")
        return

    n_data_points = min(len(all_spd_data_np), cfg_spec.data_params.get('n_data_points', len(all_spd_data_np)))
    all_spd_data_np = all_spd_data_np[:n_data_points]

    if not all_spd_data_np:
        print("No SPD data generated. Exiting.")
        return
    print(f"Generated {len(all_spd_data_np)} SPD matrices as data points.")

    train_dataset_obj = SPDRawDatasetNP(all_spd_data_np)
    # The dataloader needs a JAX RNG key for shuffling
    key, dataloader_rng = random.split(key)
    train_loader_jax = jax_dataloader(
        train_dataset_obj, cfg_spec.train_params.batch_size, rng_key=dataloader_rng
    )

    # 2. Manifold Setup
    print("Setting up manifold (geomstats with JAX backend)...")
    manifold_dim = cfg_spec.manifold_params['dim']
    if manifold_dim != cfg_spec.data_params['num_series']:
        print(f"Warning: Manifold dim ({manifold_dim}) differs from data dim ({cfg_spec.data_params['num_series']}). Using data dim.")
        manifold_dim = cfg_spec.data_params['num_series']
        cfg_spec.score_net_params['manifold_dim'] = manifold_dim # Update for score net

    manifold = geomstats.geometry.spd_matrices.SPDMatrices(n=manifold_dim)
    if cfg_spec.manifold_params['metric_type'] == 'affine':
        pass
    elif cfg_spec.manifold_params['metric_type'] == 'log_euclidean':
        manifold.metric = geomstats.geometry.spd_matrices.SPDLogEuclideanMetric(n=manifold_dim)
    else:
        raise ValueError(f"Unknown metric_type: {cfg_spec.manifold_params['metric_type']}")


    # 3. Score Network Setup (Haiku U-Net)
    print("Setting up score network (Haiku U-Net)...")
    key, subkey = random.split(key)
    
    activation_fn = getattr(jax.nn, cfg_spec.score_net_params.get('activation', 'elu'))

    def score_model_forward(y, t, context=None): # Haiku transform expects (y, t) for main input
        model = SPDUnetScoreNetwork(
            manifold_dim=cfg_spec.score_net_params['manifold_dim'],
            channels=cfg_spec.score_net_params['channels'],
            embed_dim=cfg_spec.score_net_params['embed_dim'],
            activation=activation_fn
        )
        return model(y, t)
    
    score_model_transformed = hk.transform_with_state(score_model_forward)

    dummy_x = jnp.zeros((cfg_spec.train_params.batch_size, manifold_dim, manifold_dim))
    dummy_t = jnp.zeros((cfg_spec.train_params.batch_size,))
    
    init_params, init_state = score_model_transformed.init(
        subkey, dummy_x, dummy_t
    )

    # 4. SDE Setup (Langevin only)
    print("Setting up SDE (Langevin)...")
    if cfg_spec.sde_params['sde_type'].lower() == 'langevin':
        beta_schedule = LinearBetaSchedule(
            beta_0=cfg_spec.sde_params['beta_min'],
            beta_f=cfg_spec.sde_params['beta_max'],
            tf=cfg_spec.sde_params['T'],
            t0=0.0
        )
        sde_instance = Langevin(
            beta_schedule=beta_schedule,
            manifold=manifold,
            ref_scale=0.5, 
            ref_mean=gs.eye(manifold.n), 
            N=1000
        )
    elif cfg_spec.sde_params['sde_type'].lower() == 'vpsde': # This allows VPSDE from config
        # VPSDE expects beta_schedule
        beta_schedule = LinearBetaSchedule(
            beta_0=cfg_spec.sde_params['beta_min'],
            beta_f=cfg_spec.sde_params['beta_max'],
            tf=cfg_spec.sde_params['T'],
            t0=0.0
        )
        
        sde_instance = VPSDE(
            beta_schedule=beta_schedule, # Pass the schedule
            manifold=manifold, # Pass the manifold
            ref_mean=gs.eye(manifold.n),
            # N=1000 # N for discretization
        )
    else:
        raise ValueError(f"SDE type '{cfg_spec.sde_params['sde_type']}' not supported in this script. Use 'langevin'.")

    # 5. Optimizer Setup
    optimiser = optax.chain(
        optax.clip_by_global_norm(cfg_spec.train_params.grad_clip),
        optax.adam(learning_rate=cfg_spec.train_params.lr, b1=0.9, b2=0.999, eps=1e-8)
    )
    opt_state = optimiser.init(init_params)

    train_state = TrainState(
        opt_state=opt_state,
        model_state=init_state,
        step=0,
        params=init_params,
        ema_rate=cfg_spec.train_params.ema_decay if cfg_spec.train_params.use_ema else None,
        params_ema=init_params if cfg_spec.train_params.use_ema else None,
        rng=key,
    )

    # 6. Loss function callable (get_dsm_loss_fn)

    from score_sde.models.transform import Id
    pushforward = SDEPushForward(sde_instance, manifold, transform=Id(domain=manifold))

    loss_fn_callable = get_dsm_loss_fn(
        pushforward=pushforward, 
        model=score_model_transformed, 
        train=True,
        like_w=False, 
        eps=1e-3,
        s_zero=True 
    )

    train_step_wrapped = get_ema_loss_step_fn(
        loss_fn_callable,
        optimizer=optimiser,
        train=True,
    )
    
    @jit
    def actual_train_step(current_train_state, batch_data_jax):
        # train_step_wrapped expects batch={'data': x_0, 'label': label}
        batch_for_step = {'data': batch_data_jax, 'context': None}
        
        (new_rng, new_train_state), loss_value = train_step_wrapped((current_train_state.rng, current_train_state), batch_for_step)
        return new_train_state, loss_value


    # Training loop
    key, dataloader_rng_for_loop = random.split(key) # Split key for the loop's RNG
    train_data_generator = jax_dataloader(
        train_dataset_obj, cfg_spec.train_params.batch_size, rng_key=dataloader_rng_for_loop
    )
    print(f"Starting training for job: {cfg_spec.train_params.job_name} (Geomstats backend: JAX)...")
    for epoch in range(cfg_spec.train_params.epochs):
        total_loss = 0.0
        num_batches = 0

        key, dataloader_rng_for_epoch = random.split(key) # Новый RNG для каждой эпохи
        train_data_generator = jax_dataloader(
            train_dataset_obj, cfg_spec.train_params.batch_size, rng_key=dataloader_rng_for_epoch
        )
        # ИСПРАВЛЕНИЕ: Итерируем по объекту-генератору
        for batch_data_np in train_data_generator: # <--- КОРРЕКТНАЯ ИТЕРАЦИЯ
            batch_data_jax = jnp.asarray(batch_data_np)

            key, subkey = random.split(key) # Split key for train_step_wrapped (if needed)
            # print("here")
            train_state, loss_value = actual_train_step(train_state, batch_data_jax)
            
            total_loss += loss_value
            num_batches += 1
            
            if (num_batches % cfg_spec.train_params.log_every) == 0:
                print(f"  Batch {num_batches}/{len(train_dataset_obj)//cfg_spec.train_params.batch_size}, Loss: {loss_value:.4f}")
        # print("here2")
        if num_batches:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{cfg_spec.train_params.epochs}, Avg Loss: {avg_loss:.4f}")

    print("Training finished.")

    # --- Sampling ---
    # 1. Получаем score_fn из обученной модели
    # get_score_fn возвращает callable(x, t, rng), который вызывает model.apply
    trained_score_fn = get_score_fn(
        sde=sde_instance, 
        model=score_model_transformed,
        params=train_state.params_ema if cfg_spec.train_params.use_ema else train_state.params,
        state=train_state.model_state, # Состояние модели (для GroupNorm и т.д.)
        train=False, # Режим инференса
        return_state=False, # По умолчанию False, явно указать
        std_trick=True, # По умолчанию True, явно указать
        residual_trick=True # Если скор масштабируется по std
    )

    reverse_sde_instance = sde_instance.reverse(trained_score_fn)
    
    from riemannian_score_sde.sampling import get_pc_sampler
    from score_sde.sampling import get_predictor # For "GRW" or "EulerMaruyamaPredictor"

    sampler = get_pc_sampler(
        sde=reverse_sde_instance, # <--- ПЕРВЫЙ АРГУМЕНТ: ОБЪЕКТ ОБРАТНОГО SDE
        N=cfg_spec.sample_params.num_steps, # <--- ВТОРОЙ АРГУМЕНТ: N (количество шагов)
        predictor="EulerMaruyamaPredictor", # <--- ТРЕТИЙ АРГУМЕНТ: Предиктор (строка)
        corrector="NoneCorrector", # <--- ЧЕТВЕРТЫЙ АРГУМЕНТ: Корректор (строка)
        return_hist=False,
    )
    
    # 4. Создаем начальные точки 'x' для сэмплера
    # Это будут сэмплы из предельного (noise) распределения SDE
    # sde_instance.sample_limiting_distribution(rng, shape)
    # shape должен быть (num_samples, dim, dim)
    key, sample_init_rng = random.split(key)
    initial_sample_points = sde_instance.sample_limiting_distribution(
        sample_init_rng, 
        (cfg_spec.sample_params.num_samples, manifold_dim, manifold_dim) # Сэмплы в виде матриц
    )

    print(f"initial_sample_points = {initial_sample_points.shape}")

    # 5. Вызываем sampler() с начальными точками 'x' и временем 'tf'
    # tf - это конечное время SDE (обычно 0 для обратного процесса)
    # По умолчанию в get_pc_sampler tf=0, но лучше явно передать
    rng, final_sample_rng = random.split(key)
    generated_samples = sampler(final_sample_rng, x=initial_sample_points, tf=0.0) 

    print(f"Generated samples shape: {generated_samples.shape}")

    # You can save/visualize generated_samples here.


if __name__ == '__main__':
    # Ensure JAX backend for geomstats is set early
    os.environ['GEOMSTATS_BACKEND'] = 'jax'
    # Import geomstats.backend AFTER setting the environment variable
    # import geomstats.backend as gs_check
    # gs_check.set_default_backend('jax')

    parser = argparse.ArgumentParser(description="Train RSGM for SPD matrices (NumPy data, JAX U-Net).")
    parser.add_argument('--config_path', type=str, default=None,
                        help="Path to the Python configuration file.")
    args = parser.parse_args()

    if args.config_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config_module", args.config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        cfg_spec_loaded = config_module.get_config()
        print(f"Loaded configuration from: {args.config_path}")
    else:
        print("No config_path provided, using default configuration.")
        cfg_spec_loaded = create_default_config_spec_spd_unet()

    main_jax(cfg_spec_loaded)
