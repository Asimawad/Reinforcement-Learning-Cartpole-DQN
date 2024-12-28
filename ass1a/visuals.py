import vlc
import time

# Path to your video file
video_path = "video/q_learning/eval/rl-video-episode-792.mp4"

# Create VLC instance
player = vlc.MediaPlayer(video_path)

# Play the video
player.play()

# Wait for the video to finish (assuming it's 10 seconds long)
time.sleep(10)

# Stop the player (optional)
player.stop()
