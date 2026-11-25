
import json
import time

import cactus


def main():
    print("Loading Mirage...")
    start_time = time.perf_counter()

    vision = cactus.Mirage()

    load_time = time.perf_counter() - start_time
    print(f"Model initialized in {load_time:.2f} seconds")

    image_path = "test_image.jpg"

    print("\nRunning object detection...")
    inference_start = time.perf_counter()

    result = vision.describe(
        image_path,
        sections=["objects"],
        fields=["bbox", "score"],
    )

    inference_time = time.perf_counter() - inference_start

    print(f"Inference completed in {inference_time:.2f} seconds")
    print("\nModel output:\n")

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
