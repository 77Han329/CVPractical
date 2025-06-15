import cv2
import os
from natsort import natsorted
from tqdm import tqdm

def make_videos_from_steps(folder_path, output_path, image_prefix="step", batch_size=4, frame_repeat=3, fps=10, step_interval=5):
    for batch_idx in range(batch_size):
 
        image_files = [
            f for f in os.listdir(folder_path)
            if f.startswith(image_prefix) and f.endswith(f"{batch_idx:06d}.png")
        ]
        image_files = natsorted(image_files)

      
        filtered_files = []
        for name in image_files:
            try:
                step = int(name.split("_")[1])
                if step % step_interval == 0:
                    filtered_files.append(name)
            except ValueError:
                continue

        if not filtered_files:
            print(f"No images found for batch {batch_idx}")
            continue

        first_img = cv2.imread(os.path.join(folder_path, filtered_files[0]))
        height, width, _ = first_img.shape

        video_path = os.path.join(folder_path, f"batch{batch_idx:02d}_interval{step_interval}.mp4")
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        print(f"Creating video: {video_path}")
        for img_name in tqdm(filtered_files):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            step_str = img_name.split("_")[1]
            cv2.putText(img, f"Step {step_str}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            for _ in range(frame_repeat):
                writer.write(img)

        writer.release()
        print(f"Saved to {video_path}")


if __name__ == "__main__":
    folder = "samples/steps500/SiT-XL-2-pretrained-cfg-1.0-4-ODE-500-dopri5"
    make_videos_from_steps(
        folder_path=folder,
        output_path=folder,
        step_interval=5  
    )