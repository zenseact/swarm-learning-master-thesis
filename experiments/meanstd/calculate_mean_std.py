#!/usr/bin/env python3

import numpy as np
from zod import ZodFrames
import concurrent.futures
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def process_image(i, zod_frames):
    try:
        image = zod_frames[i].get_image().astype(np.float32) / 255.0
        mean = np.mean(image, axis=(0, 1))
        squared_mean = np.mean(np.square(image), axis=(0, 1))
        return mean, squared_mean
    except Exception as e:
        logging.error(f"Error processing image {i}: {e}")
        return np.zeros(3), np.zeros(3)


def calculate_mean_std():
    logging.info("Calculating mean and std...")
    mean = np.zeros(3)
    squared_mean = np.zeros(3)

    zod_frames = ZodFrames("/mnt/ZOD", "full")

    logging.info("ZodFrames loaded")
    ids = zod_frames._frames.keys()
    num_images = len(ids)

    logging.info("Starting loop")
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        for step, (mean_i, squared_mean_i) in enumerate(executor.map(process_image, ids, [zod_frames] * num_images)):
            if step % 1000 == 0:
                logging.info(f"Processing image {step}/{num_images}")
            mean += mean_i
            squared_mean += squared_mean_i

    mean /= num_images
    squared_mean /= num_images
    std = np.sqrt(squared_mean - np.square(mean))

    return mean, std


def main():
    mean, std = calculate_mean_std()
    logging.info(f"Mean: {mean}, Std: {std}")


main()
