mod gym_env;

use gym_env::{GymEnv, Step};
use ndarray::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use std::cmp;
use tch::nn::{Optimizer, OptimizerConfig, VarStore};
use tch::{nn, Device, Reduction, Tensor};

struct ReplayBuffer {
    obs: Array2<f64>,
    next_obs: Array2<f64>,
    actions: Array1<i64>,
    rewards: Array1<f64>,
    done: Array1<f64>,
    max_size: usize,
    batch_size: usize,
    ptr: usize,
    size: usize,
}

impl ReplayBuffer {
    pub fn new(obs_dim: usize, size: usize, batch_size: usize) -> Self {
        Self {
            obs: Array2::<f64>::zeros((size, obs_dim)),
            next_obs: Array2::<f64>::zeros((size, obs_dim)),
            actions: Array1::<i64>::zeros(size),
            rewards: Array1::<f64>::zeros(size),
            done: Array1::<f64>::zeros(size),
            max_size: size,
            batch_size,
            ptr: 0,
            size: 0,
        }
    }

    pub fn store(
        &mut self,
        obs: Array1<f64>,
        next_obs: Array1<f64>,
        actions: i64,
        rewards: f64,
        done: bool,
    ) {
        self.obs
            .row_mut(self.ptr)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, val)| *val = obs[idx]);
        self.next_obs
            .row_mut(self.ptr)
            .iter_mut()
            .enumerate()
            .for_each(|(idx, val)| *val = next_obs[idx]);
        self.actions.iter_mut().for_each(|val| *val = actions);
        self.rewards.iter_mut().for_each(|val| *val = rewards);
        self.done
            .iter_mut()
            .for_each(|val| *val = if done { 1.0 } else { 0.0 });
        self.ptr = (self.ptr + 1) % self.max_size;
        self.size = cmp::min(self.size + 1, self.max_size);
    }

    pub fn sample_batch(
        &mut self,
    ) -> (
        Array2<f64>,
        Array2<f64>,
        Array1<i64>,
        Array1<f64>,
        Array1<f64>,
    ) {
        let mut rng = &mut rand::thread_rng();
        let sizes: Vec<usize> = (0..self.size).collect();

        let idxs: Vec<usize> = sizes
            .choose_multiple(&mut rng, self.batch_size)
            .cloned()
            .collect();

        let shape = self.obs.shape();
        let mut obs = Array2::<f64>::zeros((self.batch_size, shape[1]));
        let mut next_obs = Array2::<f64>::zeros((self.batch_size, shape[1]));
        let mut actions = Array1::<i64>::zeros(self.batch_size);
        let mut rewards = Array1::<f64>::zeros(self.batch_size);
        let mut done = Array1::<f64>::zeros(self.batch_size);

        for i in 0..self.batch_size {
            obs.row_mut(i)
                .iter_mut()
                .enumerate()
                .for_each(|(idx, val)| *val = self.obs[[idxs[i], idx]]);
            next_obs
                .row_mut(i)
                .iter_mut()
                .enumerate()
                .for_each(|(idx, val)| *val = self.next_obs[[idxs[i], idx]]);
            actions
                .iter_mut()
                .for_each(|val| *val = self.actions[idxs[i]]);
            rewards
                .iter_mut()
                .for_each(|val| *val = self.rewards[idxs[i]]);
            done.iter_mut().for_each(|val| *val = self.done[idxs[i]]);
        }

        (obs, next_obs, actions, rewards, done)
    }

    pub fn len(&self) -> usize {
        self.size as usize
    }
}

struct Network {
    network: nn::Sequential,
    device: Device,
}

impl Network {
    fn new(in_dim: i64, out_dim: i64) -> Self {
        let var_store = nn::VarStore::new(tch::Device::Cpu);
        let path = &var_store.root();

        Self {
            network: nn::seq()
                .add(nn::linear(path, in_dim, 128, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(path, 128, 128, Default::default()))
                .add_fn(|xs| xs.relu())
                .add(nn::linear(path, 128, out_dim, Default::default())),
            device: path.device(),
        }
    }

    fn forward(&self, obs: &Tensor) -> Tensor {
        obs.to_device(self.device).apply(&self.network)
    }
}

struct Transition {
    pub obs: Option<Array1<f64>>,
    pub next_obs: Option<Array1<f64>>,
    pub action: Option<i64>,
    pub reward: Option<f64>,
    pub done: Option<bool>,
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
    batch_size: usize,
    epsilon: f32,
    epsilon_decay: f32,
    max_epsilon: f32,
    min_epsilon: f32,
    target_update: i64,
    gamma: f32,
    dqn: Network,
    dqn_target: Network,
    transition: Transition,
    optimizer: Optimizer,
    is_test: bool,
}

impl DQNAgent {
    pub fn new(
        env: GymEnv,
        memory_size: usize,
        batch_size: usize,
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
        let optimizer = nn::Adam::default()
            .build(&var_store, learning_rate)
            .unwrap();

        Self {
            env,
            memory: ReplayBuffer::new(obs_dim as usize, memory_size, batch_size),
            batch_size,
            epsilon: max_epsilon,
            epsilon_decay,
            max_epsilon,
            min_epsilon,
            target_update,
            gamma,
            dqn: Network::new(obs_dim, action_dim),
            dqn_target: Network::new(obs_dim, action_dim),
            transition: Transition::new(),
            optimizer,
            is_test: false,
        }
    }

    pub fn select_action(&mut self, state: &Array1<f64>) -> i64 {
        let rng = &mut rand::thread_rng();

        let selected_action: i64;
        if self.epsilon > rng.gen_range(0.0..1.0) {
            selected_action = rng.gen_range(0..self.env.action_space());
        } else {
            let val = self.dqn.forward(&Tensor::try_from(state.clone()).unwrap());
            let arr: ArrayD<i64> = (&val).try_into().unwrap();
            selected_action = *arr.iter().max().unwrap();
        };

        if !self.is_test {
            self.transition.obs = Some(state.clone());
            self.transition.action = Some(selected_action);
        }

        selected_action
    }

    pub fn step(&mut self, action: &i64) -> (Array1<f64>, f64, bool) {
        let Step {
            next_state,
            reward,
            is_done,
            ..
        } = self.env.step(&action).unwrap();

        let next_state: ArrayD<f64> = (&next_state).try_into().unwrap();
        let next_state = Array1::from_vec(Vec::from(next_state.as_slice().unwrap()));

        if !self.is_test {
            self.transition.next_obs = Some(next_state.clone());
            self.transition.reward = Some(reward);
            self.transition.done = Some(is_done);

            self.memory.store(
                self.transition.obs.as_ref().unwrap().clone(),
                self.transition.next_obs.as_ref().unwrap().clone(),
                self.transition.action.unwrap(),
                self.transition.reward.unwrap(),
                self.transition.done.unwrap(),
            );
        }

        (next_state, reward, is_done)
    }

    pub fn update_model(&mut self) -> Tensor {
        let samples = self.memory.sample_batch();
        let loss = self.compute_dqn_loss(&samples);

        self.optimizer.zero_grad();
        loss.backward();
        self.optimizer.step();

        loss.get(0)
    }

    pub fn train(&mut self, num_frames: i64) {
        self.is_test = false;

        let mut state = self.env.reset().unwrap();
        let mut update_cnt = 0;
        let mut epsilons = Vec::new();
        let mut losses = Vec::new();
        let mut scores = Vec::new();
        let mut score = 0.0;

        for i in 1..=num_frames {
            let converted_state: ArrayD<f64> = (&state).try_into().unwrap();
            let converted_state = Array1::from_vec(Vec::from(converted_state.as_slice().unwrap()));
            let action = self.select_action(&converted_state);
            let (next_state, reward, done) = self.step(&action);

            state = Tensor::try_from(next_state).unwrap();
            score += reward;

            if done {
                state = self.env.reset().unwrap();
                scores.push(score);
                score = 0.0;
            }

            println!("frame {} with score of {}", i, score);

            if self.memory.len() >= self.batch_size {
                let loss = self.update_model();
                losses.push(loss);
                update_cnt += 1;

                self.epsilon = f32::max(
                    self.min_epsilon,
                    self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay,
                );
                epsilons.push(self.epsilon);

                if update_cnt % self.target_update == 0 {
                    self.target_hard_update();
                }
            }
        }
    }

    fn compute_dqn_loss(
        &mut self,
        samples: &(
            Array2<f64>,
            Array2<f64>,
            Array1<i64>,
            Array1<f64>,
            Array1<f64>,
        ),
    ) -> Tensor {
        let state = Tensor::try_from(samples.0.clone()).unwrap();
        let next_state = Tensor::try_from(samples.1.clone()).unwrap();
        let action = Tensor::try_from(samples.2.clone()).unwrap();
        let reward = Tensor::try_from(samples.3.clone()).unwrap();
        let done = Tensor::try_from(samples.4.clone()).unwrap();

        let curr_q_value = self.dqn.forward(&state).gather(1, &action, false);
        let next_q_value = self.dqn_target.forward(&next_state).max_dim(1, true).0;
        let mask = 1 - done;
        let target = reward + self.gamma * next_q_value * mask;

        let loss = curr_q_value.smooth_l1_loss(&target, Reduction::None, 1.0);

        loss
    }

    fn target_hard_update(&mut self) {}
}

fn main() {
    let env = GymEnv::new("CartPole-v1").unwrap();
    let mut dqn_agent = DQNAgent::new(env, 1000, 32, 100, 1.0 / 2000.0, 1e-4, 1.0, 0.1, 0.99);
    dqn_agent.train(10000);
}
