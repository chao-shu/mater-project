import os
import subprocess

raw_dir = "./data/raw_data"
converted_dir = "./data/converted"
os.makedirs(converted_dir, exist_ok=True)

for filename in os.listdir(raw_dir):
    if filename.endswith(".webm"):
        raw_path = os.path.join(raw_dir, filename)
        mp4_name = os.path.splitext(filename)[0] + ".mp4"
        mp4_path = os.path.join(converted_dir, mp4_name)

        cmd = [
            "ffmpeg",
            "-i", raw_path,
            "-c:v", "h264_nvenc",
            "-vf", "fps=1",
            "-crf", "23",
            mp4_path
        ]

        print(f"Converting {filename} to {mp4_name} ...")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0:
            print(f"Successfully converted {filename} to {mp4_name}")
        else:
            print(f"Failed to convert {filename}")
            print(result.stderr.decode("utf-8"))
