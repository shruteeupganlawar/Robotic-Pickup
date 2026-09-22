# src/mp4_to_gif.py
import sys

try:
    # moviepy 2.x
    from moviepy import VideoFileClip
    V2 = True
except ImportError:
    # moviepy 1.x
    from moviepy.editor import VideoFileClip
    V2 = False

src = sys.argv[1] if len(sys.argv) > 1 else "assets/videos/demo.mp4"
dst = sys.argv[2] if len(sys.argv) > 2 else "assets/videos/demo.gif"

clip = VideoFileClip(src)
end = min(15, clip.duration)

if V2:
    clip = clip.subclipped(0, end).resized(width=480)
else:
    clip = clip.subclip(0, end).resize(width=480)

clip.write_gif(dst, fps=10)
print(f"Saved {dst}")