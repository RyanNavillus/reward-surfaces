from eval_ppo import generate_video
import sys
import json
import os
from PIL import Image
import shutil
import tempfile
import subprocess

def save_frames(frames, dir):
    results = []
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        img.save(os.path.join(dir,f'{i:03}.png'))

def save_video(pred_file, out_file):
    base_file = os.path.dirname(os.path.dirname(pred_file))
    env_name = json.load(open(os.path.join(base_file, "hyperparams.json")))['env_name']

    frames = generate_video(pred_file, out_file, env_name)

    with tempfile.TemporaryDirectory() as dirname:
        save_frames(frames, dirname)

        ffmpeg_command = [
            "convert",
            os.path.join(dirname,"*.png"),
            "-delay", "20",
            out_file
        ]
        print(" ".join(ffmpeg_command))
        subprocess.run(ffmpeg_command)

if __name__ == "__main__":
    pred_file = "train_results/test_save3/models/0.zip"
    out_file = "bob.gif"

    save_video(pred_file, out_file)
