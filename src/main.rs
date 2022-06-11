mod gym_env;

use gym_env::{GymEnv, Step};
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp;
use tch::nn::{Module, Optimizer, OptimizerConfig, VarStore};
use tch::{kind::FLOAT_CPU, nn, Tensor};

struct ReplayBuffer {
    obs: Tensor,
    next_obs: Tensor,
    actions: Tensor,
    rewards: Tensor,
    done: Tensor,
    max_size: i64,
    batch_size: i64,
    ptr: i64,
    size: i64,
}

impl ReplayBuffer {
    pub fn new(obs_dim: i64, size: i64, batch_size: i64) -> Self {
        Self {
            obs: Tensor::zeros(&[size as _, obs_dim as _], FLOAT_CPU),
            next_obs: Tensor::zeros(&[size as _, obs_dim as _], FLOAT_CPU),
            actions: Tensor::zeros(&[size as _, 1], FLOAT_CPU),
            rewards: Tensor::zeros(&[size as _, 1], FLOAT_CPU),
            done: Tensor::zeros(&[size as _], FLOAT_CPU),
            max_size: size,
            batch_size,
            ptr: 0,
            size: 0,
        }
    }

    pub fn store(&mut self, transition: &Transition) {
        self.obs
            .get(self.ptr)
            .copy_(&transition.obs.as_ref().unwrap());
        self.next_obs
            .get(self.ptr)
            .copy_(&transition.next_obs.as_ref().unwrap());
        self.actions
            .get(self.ptr)
            .copy_(&transition.action.as_ref().unwrap());
        self.rewards
            .get(self.ptr)
            .copy_(&transition.reward.as_ref().unwrap());
        self.done
            .get(self.ptr)
            .copy_(&transition.done.as_ref().unwrap());
        self.ptr = (self.ptr + 1) % self.max_size;
        self.size = cmp::min(self.size + 1, self.max_size);
    }

    pub fn sample_batch(&mut self) -> Vec<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let mut rng = &mut rand::thread_rng();
        let sizes: Vec<i64> = (0..self.size).collect();

        let idxs: Vec<i64> = sizes
            .choose_multiple(&mut rng, self.batch_size as usize)
            .cloned()
            .collect();

        let mut ret = Vec::with_capacity(self.batch_size as usize);
        for idx in idxs {
            ret.push((
                self.obs.get(idx).copy(),
                self.next_obs.get(idx).copy(),
                self.actions.get(idx).copy(),
                self.rewards.get(idx).copy(),
                self.done.get(idx).copy(),
            ));
        }

        ret
    }

    pub fn len(&self) -> usize {
        self.size as usize
    }
}

fn network(vs: &nn::Path, in_dim: i64, out_dim: i64) -> impl Module {
    nn::seq()
        .add(nn::linear(vs, in_dim, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, 128, out_dim, Default::default()))
}

struct Transition {
    pub obs: Option<Tensor>,
    pub next_obs: Option<Tensor>,
    pub action: Option<Tensor>,
    pub reward: Option<Tensor>,
    pub done: Option<Tensor>,
}

impl Transition {
    pub fn new() -> Self {
        Self {
            obs: None,
            next_obs: None,
            action: None,
            reward: None,
            done: None,
        }
    }
}

struct DQNAgent {
    env: GymEnv,
    memory: ReplayBuffer,
    batch_size: i64,
    epsilon: f32,
    epsilon_decay: f32,
    max_epsilon: f32,
    min_epsilon: f32,
    target_update: i64,
    gamma: f32,
    var_store: VarStore,
    dqn: Box<dyn Module>,
    dqn_target: Box<dyn Module>,
    transition: Transition,
    optimizer: Optimizer,
    is_test: bool,
}

impl DQNAgent {
    pub fn new(
        env: GymEnv,
        memory_size: i64,
        batch_size: i64,
        target_update: i64,
        epsilon_decay: f32,
        learning_rate: f64,
        max_epsilon: f32,
        min_epsilon: f32,
        gamma: f32,
    ) -> Self {
        let obs_dim = env.observation_space().iter().fold(1, |acc, x| acc * x);
        let action_dim = env.action_space();

        let var_store = VarStore::new(tch::Device::Cpu);
        let dqn: Box<dyn Module> = Box::new(network(&var_store.root(), obs_dim, action_dim));
        let dqn_target: Box<dyn Module> = Box::new(network(&var_store.root(), obs_dim, action_dim));
        let optimizer = nn::Adam::default().build(&var_store, learning_rate).unwrap();

        Self {
            env,
            memory: ReplayBuffer::new(obs_dim, memory_size, batch_size),
            batch_size,
            epsilon: max_epsilon,
            epsilon_decay,
            max_epsilon,
            min_epsilon,
            target_update,
            gamma,
            var_store,
            dqn,
            dqn_target,
            transition: Transition::new(),
            optimizer,
            is_test: false,
        }
    }

    pub fn select_action(&mut self, state: &Tensor) -> Tensor {
        let mut rng = &mut rand::thread_rng();

        let selected_action = if self.epsilon > rng.gen_range(0.0..1.0) {
            rng.gen_range(0..self.env.action_space()).into()
        } else {
            self.dqn.as_mut().forward(state)
        };

        if !self.is_test {
            self.transition.obs = Some(state.copy());
            self.transition.action = Some(selected_action.copy());
        }

        selected_action
    }

    pub fn step(&mut self, action: &[f64]) -> (Tensor, f64, bool) {
        let Step {
            next_state,
            reward,
            is_done,
            ..
        } = self.env.step(&action).unwrap();

        if !self.is_test {
            self.transition.reward = Some(reward.into());
            self.transition.next_obs = Some(next_state.copy());
            self.transition.done = Some(is_done.into());

            self.memory.store(&self.transition);
        }

        (next_state, reward, is_done)
    }
}

fn main() {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}
