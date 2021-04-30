"""Visualize a learned residual controller.
"""
from residual_learning.residual_sac import ResidualSAC
import dl
import os
import torch
import numpy as np
from dl import nest
import argparse
from imageio import imwrite
import tempfile
import subprocess as sp
import shutil


class VideoWriter(object):
    def __init__(self, fps=30, img_dir=None):
        self.counter = 0
        if img_dir:
            self.img_dir = img_dir
            self.img_dir_name = img_dir
            os.makedirs(self.img_dir)
        else:
            self.img_dir = tempfile.TemporaryDirectory()
            self.img_dir_name = self.img_dir.name
        self.fps = str(fps)

    def __del__(self):
        if hasattr(self.img_dir, 'cleanup'):
            self.img_dir.cleanup()
        else:
            shutil.rmtree(self.img_dir)

    def add_frame(self, img):
        img_path = os.path.join(self.img_dir_name, f'{self.counter:06d}.png')
        imwrite(img_path, img)
        self.counter += 1

    def make_video(self, output_path):
        if self.counter > 0:
            sp.call(['ffmpeg', '-r', self.fps, '-f', 'image2', '-i',
                     os.path.join(self.img_dir_name, '%06d.png'),
                     '-vcodec', 'libx264',
                     '-pix_fmt', 'yuv420p',
                     os.path.join(self.img_dir_name, 'out.mp4')])
            sp.call(['mv', os.path.join(self.img_dir_name, 'out.mp4'), output_path])

            if hasattr(self.img_dir, 'cleanup'):
                self.img_dir.cleanup()
                self.img_dir = tempfile.TemporaryDirectory()
            else:
                shutil.rmtree(self.img_dir)
                os.makedirs(self.img_dir)
            self.counter = 0


def _load_env_and_policy(logdir, t=None):

    gin_bindings = [
        "make_training_env.sim=True",
        "make_training_env.visualization=True",
        "make_training_env.monitor=True",
    ]

    config = os.path.join(logdir, 'config.gin')
    dl.load_config(config, gin_bindings)
    alg = ResidualSAC(logdir)
    alg.load(t)
    env = alg.env
    pi = alg.pi
    dl.rl.set_env_to_eval_mode(env)
    pi.eval()
    init_ob = alg.data_manager._ob
    if t is None:
        try:
            t = max(alg.ckptr.ckpts())
        except Exception:
            t = 0
    return env, pi, alg.device, init_ob, t


def get_best_eval():
    if not os.path.exists('/logdir/eval/'):
        return None
    best_t = None
    best_r = -10 ** 9
    for eval in os.listdir('/logdir/eval/'):
        data = torch.load(os.path.join('/logdir/eval', eval))
        if best_r < data['mean_reward']:
            best_r = data['mean_reward']
            best_t = int(eval.split('.')[0])
            print(best_r, best_t)
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=int, default=None, help="checkpoint timestep")
    parser.add_argument('-n', type=int, default=1, help="number of episodes")
    parser.add_argument('--base', default=False, action='store_true', help="visualize the base_policy")
    args = parser.parse_args()

    t = get_best_eval() if args.t is None else args.t
    env, pi, device, obs, ckpt = _load_env_and_policy('/logdir', t)

    def _to_torch(x):
        return torch.from_numpy(x).to(device)

    def _to_numpy(x):
        return x.cpu().numpy()

    video_dir = '/logdir/video'
    os.makedirs(video_dir, exist_ok=True)
    tmp_dir = os.path.join(video_dir, 'tmp')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if args.base:
        output_path = os.path.join(video_dir, 'base_policy.mp4')
    else:
        output_path = os.path.join(video_dir, f'{ckpt:09d}.mp4')

    if os.path.exists(output_path):
        return

    video_writer = VideoWriter(img_dir=tmp_dir)
    for i in range(args.n):
        obs = env.reset()
        video_writer.add_frame(env.render(mode='rgb_array'))

        done = False
        while not done:
            if args.base:
                action = np.zeros_like(obs['action']['torque'])
            else:
                obs = nest.map_structure(_to_torch, obs)
                with torch.no_grad():
                    action = pi(obs).action
                action = nest.map_structure(_to_numpy, action)
            obs, _, done, _ = env.step(action)
            video_writer.add_frame(env.render(mode='rgb_array'))

    video_writer.make_video(output_path)


if __name__ == "__main__":
    main()
