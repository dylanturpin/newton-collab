# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Public HORA stage-two network and observation history, entirely on CUDA."""

from pathlib import Path

import numpy as np
import warp as wp


@wp.kernel
def normalize(x: wp.array2d[float], mean: wp.array2d[float], var: wp.array2d[float], out: wp.array2d[float]):
    i, j = wp.tid()
    out[i, j] = wp.clamp((x[i, j] - mean[i, j]) / wp.sqrt(var[i, j] + 1.0e-5), -5.0, 5.0)


@wp.kernel
def linear(
    x: wp.array2d[float], weight: wp.array2d[float], bias: wp.array[float], activation: int, out: wp.array2d[float]
):
    i, j = wp.tid()
    v = bias[j]
    for k in range(x.shape[1]):
        v += x[i, k] * weight[j, k]
    if activation == 1:
        v = wp.max(v, 0.0)
    elif activation == 2:
        v = wp.where(v >= 0.0, v, wp.exp(v) - 1.0)
    out[i, j] = v


@wp.kernel
def transpose(x: wp.array2d[float], out: wp.array2d[float]):
    i, j = wp.tid()
    out[j, i] = x[i, j]


@wp.kernel
def convolution(
    x: wp.array2d[float], weight: wp.array3d[float], bias: wp.array[float], stride: int, out: wp.array2d[float]
):
    c, t = wp.tid()
    v = bias[c]
    for k in range(weight.shape[1]):
        for s in range(weight.shape[2]):
            v += x[k, t * stride + s] * weight[c, k, s]
    out[c, t] = wp.max(v, 0.0)


@wp.kernel
def join(obs: wp.array2d[float], latent: wp.array2d[float], out: wp.array2d[float]):
    j = wp.tid()
    if j < 96:
        out[0, j] = obs[0, j]
    else:
        out[0, j] = wp.tanh(latent[0, j - 96])


@wp.kernel
def history_update(
    q: wp.array[float],
    indices: wp.array[int],
    lower: wp.array[float],
    upper: wp.array[float],
    targets: wp.array[float],
    old: wp.array2d[float],
    updated: wp.array2d[float],
    obs: wp.array2d[float],
    initialize: int,
):
    i, j = wp.tid()
    value = float(0.0)
    if initialize == 1 or i == 29:
        if j < 16:
            k = indices[j]
            value = (2.0 * q[k] - upper[j] - lower[j]) / (upper[j] - lower[j])
        else:
            value = targets[indices[j - 16]]
    else:
        value = old[i + 1, j]
    updated[i, j] = value
    if i >= 27:
        obs[0, (i - 27) * 32 + j] = value


@wp.kernel
def set_targets(
    action: wp.array2d[float],
    indices: wp.array[int],
    lower: wp.array[float],
    upper: wp.array[float],
    targets: wp.array[float],
):
    i = wp.tid()
    k = indices[i]
    targets[k] = wp.clamp(targets[k] + wp.clamp(action[0, i], -1.0, 1.0) / 24.0, lower[i], upper[i])


@wp.kernel
def copy_targets(source: wp.array[float], indices: wp.array[int], out: wp.array[float]):
    i = wp.tid()
    out[indices[i]] = source[indices[i]]


class HoraPolicy:
    def __init__(self, world, model, device="cuda:0"):
        self.world = world
        self.device = device
        data = dict(np.load(Path(__file__).parent / "assets/hora/weights.npz"))
        self.weights = {k: wp.array(v, device=device) for k, v in data.items() if v.ndim}
        cfg = world["hora_policy"]
        self.indices = wp.array(np.asarray(cfg["q_indices"]) + world["q_start"], dtype=int, device=device)
        self.lower = wp.array(np.asarray(cfg["lower"], dtype="f4"), device=device)
        self.upper = wp.array(np.asarray(cfg["upper"], dtype="f4"), device=device)
        self.initial_targets = model.joint_q
        self.targets = wp.clone(model.joint_q)
        self.history = wp.zeros((30, 32), device=device)
        self.next_history = wp.zeros((30, 32), device=device)
        self.obs = wp.zeros((1, 96), device=device)
        self.hnorm = wp.zeros((30, 32), device=device)
        self.onorm = wp.zeros((1, 96), device=device)
        for group, shape in (("running_mean_std", (1, 96)), ("sa_mean_std", (30, 32))):
            for key in ("running_mean", "running_var"):
                self.weights[f"{group}.{key}"] = wp.array(data[f"{group}.{key}"].reshape(shape), device=device)
        self.channels = [wp.zeros((30, 32), device=device) for _ in range(2)]
        self.transposed = wp.zeros((32, 30), device=device)
        self.conv = [wp.zeros((32, n), device=device) for n in (11, 7, 3)]
        self.latent = wp.zeros((1, 8), device=device)
        self.combined = wp.zeros((1, 104), device=device)
        self.actor = [wp.zeros((1, n), device=device) for n in (512, 256, 128, 16)]
        self.previous_tick = -1

    def dense(self, x, name, activation, out):
        wp.launch(
            linear,
            dim=out.shape,
            inputs=[x, self.weights[name + ".weight"], self.weights[name + ".bias"], activation, out],
            device=self.device,
        )

    def infer(self, obs, history):
        for src, group, out in ((obs, "running_mean_std", self.onorm), (history, "sa_mean_std", self.hnorm)):
            wp.launch(
                normalize,
                dim=out.shape,
                inputs=[src, self.weights[group + ".running_mean"], self.weights[group + ".running_var"], out],
                device=self.device,
            )
        x = self.hnorm
        for i, out in enumerate(self.channels):
            self.dense(x, f"adapt_tconv.channel_transform.{i * 2}", 1, out)
            x = out
        wp.launch(transpose, dim=x.shape, inputs=[x, self.transposed], device=self.device)
        x = self.transposed
        for i, out in enumerate(self.conv):
            name = f"adapt_tconv.temporal_aggregation.{i * 2}"
            wp.launch(
                convolution,
                dim=out.shape,
                inputs=[x, self.weights[name + ".weight"], self.weights[name + ".bias"], 2 if i == 0 else 1, out],
                device=self.device,
            )
            x = out
        self.dense(x.reshape((1, 96)), "adapt_tconv.low_dim_proj", 0, self.latent)
        wp.launch(join, dim=104, inputs=[self.onorm, self.latent, self.combined], device=self.device)
        x = self.combined
        for i, out in enumerate(self.actor):
            self.dense(x, f"actor_mlp.mlp.{i * 2}" if i < 3 else "mu", 2 if i < 3 else 0, out)
            x = out
        return x

    def step(self, frame, fps, state, control):
        tick = int(frame * 20 / fps)
        if tick != self.previous_tick:
            wp.launch(
                history_update,
                dim=(30, 32),
                inputs=[
                    state.joint_q,
                    self.indices,
                    self.lower,
                    self.upper,
                    self.targets,
                    self.history,
                    self.next_history,
                    self.obs,
                    int(self.previous_tick < 0),
                ],
                device=self.device,
            )
            self.history, self.next_history = self.next_history, self.history
            action = self.infer(self.obs, self.history)
            wp.launch(
                set_targets,
                dim=16,
                inputs=[action, self.indices, self.lower, self.upper, self.targets],
                device=self.device,
            )
            self.previous_tick = tick
        wp.launch(copy_targets, dim=16, inputs=[self.targets, self.indices, control.joint_target_q], device=self.device)

    def reset(self):
        self.previous_tick = -1
        wp.copy(self.targets, self.initial_targets)
