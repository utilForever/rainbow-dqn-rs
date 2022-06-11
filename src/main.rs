mod gym_env;

use rand::seq::SliceRandom;
use std::cmp;
use tch::nn::Module;
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

    pub fn store(
        &mut self,
        obs: Tensor,
        next_obs: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
    ) {
        self.obs.get(self.ptr).copy_(&obs);
        self.next_obs.get(self.ptr).copy_(&next_obs);
        self.actions.get(self.ptr).copy_(&action);
        self.rewards.get(self.ptr).copy_(&reward);
        self.done.get(self.ptr).copy_(&done);
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

fn main() {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}
