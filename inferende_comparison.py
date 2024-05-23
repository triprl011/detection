from os.path import isfile, join
from os import listdir
from ultralytics import YOLO
import cv2
import numpy as np

from utils import get_device

if __name__ == '__main__':
    images_path = "data/images/"
    labels_path = "data/labels/"
    model = YOLO("yolov8m.pt")

    objects_to_detect = [
        ["pedestrians", {"person": 0}],
        ["cars", {"car": 2, "bus": 5, "truck": 7}]
    ]
    colors = {
        "cars": (0, 0, 225),
        "pedestrians": (0, 225, 225)
    }
    thresholds_IoU = {
        "cars": 0.5,
        "pedestrians": 0.8
    }
    thresholds_conf = {
        "cars": 0.7,
        "pedestrians": 0.7
    }

    classes = [objects[0] for objects in objects_to_detect]
    metrics_report = {}
    for m in ["TP", "FP", "FN"]:
        for cl in classes:
            metrics_report[f"{m}_{cl}"] = 0

    IoUs = {obj: [] for obj in classes}
    images_names = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    device = get_device()

    for image_name in images_names:
        clean_name = image_name.split(sep=".")[0]
        label_file_path = f'{labels_path}{clean_name}.json'
        print(image_name)
        img_path = f"{images_path}/{image_name}"
        img = cv2.imread(img_path, 1)
        # results = model(img)

    inference_cpu = """
    /Users/vitalii_grebnev/miniforge3/envs/Detection1/bin/python /Users/vitalii_grebnev/PycharmProjects/Detection1/inferende_comparison.py
images79.jpeg

0: 384x640 1 person, 428.2ms
Speed: 16.4ms preprocess, 428.2ms inference, 494.9ms postprocess per image at shape (1, 3, 384, 640)
images5.jpeg

0: 384x640 1 person, 31.0ms
Speed: 1.2ms preprocess, 31.0ms inference, 52.3ms postprocess per image at shape (1, 3, 384, 640)
images38.jpeg

0: 448x640 1 person, 385.2ms
Speed: 8.5ms preprocess, 385.2ms inference, 38.6ms postprocess per image at shape (1, 3, 448, 640)
images80.jpeg

0: 512x640 (no detections), 367.5ms
Speed: 8.0ms preprocess, 367.5ms inference, 27.9ms postprocess per image at shape (1, 3, 512, 640)
images14.jpeg

0: 384x640 1 person, 43.3ms
Speed: 1.3ms preprocess, 43.3ms inference, 64.7ms postprocess per image at shape (1, 3, 384, 640)
images43.jpeg

0: 448x640 1 car, 44.4ms
Speed: 1.2ms preprocess, 44.4ms inference, 60.5ms postprocess per image at shape (1, 3, 448, 640)
images75.jpeg

0: 480x640 1 person, 532.1ms
Speed: 12.6ms preprocess, 532.1ms inference, 117.3ms postprocess per image at shape (1, 3, 480, 640)
00 (57).png

0: 448x640 1 car, 50.1ms
Speed: 1.9ms preprocess, 50.1ms inference, 26.8ms postprocess per image at shape (1, 3, 448, 640)
images9.jpeg

0: 384x640 (no detections), 37.9ms
Speed: 1.1ms preprocess, 37.9ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
images22.jpeg

0: 384x640 1 car, 27.0ms
Speed: 1.6ms preprocess, 27.0ms inference, 75.1ms postprocess per image at shape (1, 3, 384, 640)
00 (1).png

0: 640x608 1 car, 543.0ms
Speed: 11.8ms preprocess, 543.0ms inference, 131.8ms postprocess per image at shape (1, 3, 640, 608)
images34.jpeg

0: 448x640 1 person, 48.7ms
Speed: 1.2ms preprocess, 48.7ms inference, 11.9ms postprocess per image at shape (1, 3, 448, 640)
images63.jpeg

0: 416x640 1 person, 1 car, 608.8ms
Speed: 12.3ms preprocess, 608.8ms inference, 66.1ms postprocess per image at shape (1, 3, 416, 640)
00 (82).png

0: 352x640 (no detections), 522.0ms
Speed: 10.8ms preprocess, 522.0ms inference, 44.0ms postprocess per image at shape (1, 3, 352, 640)
images18.jpeg

0: 448x640 (no detections), 47.3ms
Speed: 1.3ms preprocess, 47.3ms inference, 1.2ms postprocess per image at shape (1, 3, 448, 640)
images59.jpeg

0: 352x640 1 car, 36.3ms
Speed: 1.4ms preprocess, 36.3ms inference, 37.5ms postprocess per image at shape (1, 3, 352, 640)
images58.jpeg

0: 384x640 1 car, 37.9ms
Speed: 1.2ms preprocess, 37.9ms inference, 3.6ms postprocess per image at shape (1, 3, 384, 640)
images19.jpeg

0: 384x640 (no detections), 27.1ms
Speed: 1.5ms preprocess, 27.1ms inference, 11.0ms postprocess per image at shape (1, 3, 384, 640)
images.jpeg

0: 416x640 1 person, 1 car, 49.9ms
Speed: 1.5ms preprocess, 49.9ms inference, 94.4ms postprocess per image at shape (1, 3, 416, 640)
images35.jpeg

0: 448x640 1 car, 43.8ms
Speed: 1.2ms preprocess, 43.8ms inference, 90.6ms postprocess per image at shape (1, 3, 448, 640)
images23.jpeg

0: 384x640 2 cars, 43.0ms
Speed: 2.0ms preprocess, 43.0ms inference, 107.7ms postprocess per image at shape (1, 3, 384, 640)
images8.jpeg

0: 448x640 (no detections), 43.6ms
Speed: 1.2ms preprocess, 43.6ms inference, 1.1ms postprocess per image at shape (1, 3, 448, 640)
images74.jpeg

0: 448x640 1 car, 35.4ms
Speed: 1.6ms preprocess, 35.4ms inference, 107.9ms postprocess per image at shape (1, 3, 448, 640)
images54.jpeg

0: 384x640 (no detections), 39.4ms
Speed: 1.2ms preprocess, 39.4ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
00 (37).png

0: 608x640 1 car, 644.6ms
Speed: 17.4ms preprocess, 644.6ms inference, 30.8ms postprocess per image at shape (1, 3, 608, 640)
images42.jpeg

0: 416x640 1 person, 1 car, 45.5ms
Speed: 1.1ms preprocess, 45.5ms inference, 3.7ms postprocess per image at shape (1, 3, 416, 640)
images15.jpeg

0: 384x640 1 person, 1 car, 38.2ms
Speed: 1.5ms preprocess, 38.2ms inference, 3.7ms postprocess per image at shape (1, 3, 384, 640)
images81.jpeg

0: 480x640 4 persons, 47.5ms
Speed: 1.7ms preprocess, 47.5ms inference, 135.3ms postprocess per image at shape (1, 3, 480, 640)
images39.jpeg

0: 384x640 2 cars, 41.8ms
Speed: 1.3ms preprocess, 41.8ms inference, 100.2ms postprocess per image at shape (1, 3, 384, 640)
images78.jpeg

0: 480x640 (no detections), 49.6ms
Speed: 1.3ms preprocess, 49.6ms inference, 1.2ms postprocess per image at shape (1, 3, 480, 640)
images73.jpeg

0: 384x640 (no detections), 44.4ms
Speed: 1.5ms preprocess, 44.4ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
images24.jpeg

0: 384x640 (no detections), 31.1ms
Speed: 1.5ms preprocess, 31.1ms inference, 11.8ms postprocess per image at shape (1, 3, 384, 640)
images32.jpeg

0: 384x640 (no detections), 29.5ms
Speed: 1.5ms preprocess, 29.5ms inference, 10.1ms postprocess per image at shape (1, 3, 384, 640)
images65.jpeg

0: 448x640 3 persons, 43.8ms
Speed: 1.6ms preprocess, 43.8ms inference, 130.7ms postprocess per image at shape (1, 3, 448, 640)
images49.jpeg

0: 448x640 (no detections), 35.9ms
Speed: 1.7ms preprocess, 35.9ms inference, 12.4ms postprocess per image at shape (1, 3, 448, 640)
images28.jpeg

0: 384x640 2 persons, 38.0ms
Speed: 1.6ms preprocess, 38.0ms inference, 90.0ms postprocess per image at shape (1, 3, 384, 640)
images3.jpeg

0: 416x640 1 person, 1 car, 44.9ms
Speed: 1.4ms preprocess, 44.9ms inference, 3.9ms postprocess per image at shape (1, 3, 416, 640)
images69.jpeg

0: 448x640 (no detections), 50.8ms
Speed: 1.6ms preprocess, 50.8ms inference, 1.2ms postprocess per image at shape (1, 3, 448, 640)
images12.jpeg

0: 480x640 1 person, 49.1ms
Speed: 1.7ms preprocess, 49.1ms inference, 36.1ms postprocess per image at shape (1, 3, 480, 640)
00 (316).jpg

0: 416x640 1 truck, 41.9ms
Speed: 1.4ms preprocess, 41.9ms inference, 83.8ms postprocess per image at shape (1, 3, 416, 640)
Unknown10.jpeg

0: 448x640 1 person, 49.9ms
Speed: 1.6ms preprocess, 49.9ms inference, 11.5ms postprocess per image at shape (1, 3, 448, 640)
images45.jpeg

0: 448x640 1 car, 30.7ms
Speed: 1.5ms preprocess, 30.7ms inference, 39.4ms postprocess per image at shape (1, 3, 448, 640)
00 (30).png

0: 576x640 (no detections), 826.2ms
Speed: 16.8ms preprocess, 826.2ms inference, 80.1ms postprocess per image at shape (1, 3, 576, 640)
00 (26).png

0: 640x544 1 car, 824.0ms
Speed: 16.2ms preprocess, 824.0ms inference, 67.6ms postprocess per image at shape (1, 3, 640, 544)
Unknown5.jpeg

0: 448x640 1 person, 1 truck, 50.0ms
Speed: 1.2ms preprocess, 50.0ms inference, 29.1ms postprocess per image at shape (1, 3, 448, 640)
Unknown4.jpeg

0: 352x640 (no detections), 36.2ms
Speed: 1.2ms preprocess, 36.2ms inference, 1.2ms postprocess per image at shape (1, 3, 352, 640)
images52.jpeg

0: 480x640 (no detections), 46.4ms
Speed: 1.8ms preprocess, 46.4ms inference, 1.1ms postprocess per image at shape (1, 3, 480, 640)
00 (31).png

0: 640x448 2 cars, 707.5ms
Speed: 17.6ms preprocess, 707.5ms inference, 114.8ms postprocess per image at shape (1, 3, 640, 448)
images44.jpeg

0: 384x640 1 car, 42.6ms
Speed: 1.1ms preprocess, 42.6ms inference, 3.6ms postprocess per image at shape (1, 3, 384, 640)
Unknown11.jpeg

0: 448x640 1 person, 43.7ms
Speed: 1.5ms preprocess, 43.7ms inference, 3.6ms postprocess per image at shape (1, 3, 448, 640)
00 (317).jpg

0: 448x640 1 truck, 30.5ms
Speed: 1.6ms preprocess, 30.5ms inference, 30.6ms postprocess per image at shape (1, 3, 448, 640)
images13.jpeg

0: 416x640 (no detections), 41.9ms
Speed: 1.2ms preprocess, 41.9ms inference, 1.2ms postprocess per image at shape (1, 3, 416, 640)
images87.jpeg

0: 384x640 (no detections), 42.9ms
Speed: 1.3ms preprocess, 42.9ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
images68.jpeg

0: 384x640 1 car, 32.2ms
Speed: 1.2ms preprocess, 32.2ms inference, 14.6ms postprocess per image at shape (1, 3, 384, 640)
images29.jpeg

0: 448x640 1 person, 44.4ms
Speed: 1.5ms preprocess, 44.4ms inference, 36.0ms postprocess per image at shape (1, 3, 448, 640)
images48.jpeg

0: 448x640 (no detections), 30.9ms
Speed: 1.2ms preprocess, 30.9ms inference, 14.8ms postprocess per image at shape (1, 3, 448, 640)
00 (85).png

0: 576x640 (no detections), 65.8ms
Speed: 2.2ms preprocess, 65.8ms inference, 1.2ms postprocess per image at shape (1, 3, 576, 640)
images33.jpeg

0: 480x640 (no detections), 47.1ms
Speed: 1.6ms preprocess, 47.1ms inference, 1.2ms postprocess per image at shape (1, 3, 480, 640)
images25.jpeg

0: 384x640 1 car, 38.0ms
Speed: 1.4ms preprocess, 38.0ms inference, 103.8ms postprocess per image at shape (1, 3, 384, 640)
images67.jpeg

0: 352x640 1 traffic light, 40.1ms
Speed: 1.1ms preprocess, 40.1ms inference, 39.4ms postprocess per image at shape (1, 3, 352, 640)
00 (12).png

0: 640x640 (no detections), 60.4ms
Speed: 17.7ms preprocess, 60.4ms inference, 60.7ms postprocess per image at shape (1, 3, 640, 640)
images30.jpeg

0: 448x640 1 truck, 50.3ms
Speed: 1.6ms preprocess, 50.3ms inference, 4.3ms postprocess per image at shape (1, 3, 448, 640)
images26.jpeg

0: 384x640 (no detections), 39.8ms
Speed: 1.4ms preprocess, 39.8ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
images71.jpeg

0: 384x640 1 person, 2 cars, 27.1ms
Speed: 1.3ms preprocess, 27.1ms inference, 43.9ms postprocess per image at shape (1, 3, 384, 640)
images51.jpeg

0: 320x640 (no detections), 830.5ms
Speed: 15.8ms preprocess, 830.5ms inference, 49.4ms postprocess per image at shape (1, 3, 320, 640)
images47.jpeg

0: 384x640 1 car, 43.5ms
Speed: 1.1ms preprocess, 43.5ms inference, 11.0ms postprocess per image at shape (1, 3, 384, 640)
images10.jpeg

0: 448x640 1 car, 43.4ms
Speed: 1.6ms preprocess, 43.4ms inference, 3.6ms postprocess per image at shape (1, 3, 448, 640)
images84.jpeg

0: 416x640 (no detections), 41.7ms
Speed: 1.5ms preprocess, 41.7ms inference, 1.1ms postprocess per image at shape (1, 3, 416, 640)
pedestrians-crosswalk-18256202_DieterHawlan-ml-500px.jpg

0: 480x640 1 car, 46.5ms
Speed: 1.8ms preprocess, 46.5ms inference, 35.0ms postprocess per image at shape (1, 3, 480, 640)
images1.jpeg

0: 448x640 (no detections), 50.1ms
Speed: 1.3ms preprocess, 50.1ms inference, 1.2ms postprocess per image at shape (1, 3, 448, 640)
00 (8).png

0: 640x576 2 cars, 691.9ms
Speed: 19.5ms preprocess, 691.9ms inference, 128.4ms postprocess per image at shape (1, 3, 640, 576)
images11.jpeg

0: 384x640 (no detections), 40.6ms
Speed: 1.1ms preprocess, 40.6ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
images46.jpeg

0: 448x640 (no detections), 42.7ms
Speed: 1.5ms preprocess, 42.7ms inference, 1.1ms postprocess per image at shape (1, 3, 448, 640)
images50.jpeg

0: 384x640 (no detections), 37.3ms
Speed: 1.5ms preprocess, 37.3ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
00 (72).png

0: 448x640 (no detections), 42.8ms
Speed: 1.8ms preprocess, 42.8ms inference, 1.1ms postprocess per image at shape (1, 3, 448, 640)
images70.jpeg

0: 384x640 1 bus, 42.6ms
Speed: 1.5ms preprocess, 42.6ms inference, 4.0ms postprocess per image at shape (1, 3, 384, 640)
00 (52).png

0: 448x640 (no detections), 50.0ms
Speed: 1.9ms preprocess, 50.0ms inference, 1.1ms postprocess per image at shape (1, 3, 448, 640)
images27.jpeg

0: 480x640 (no detections), 48.0ms
Speed: 1.5ms preprocess, 48.0ms inference, 1.2ms postprocess per image at shape (1, 3, 480, 640)
images31.jpeg

0: 384x640 1 car, 38.0ms
Speed: 1.6ms preprocess, 38.0ms inference, 3.7ms postprocess per image at shape (1, 3, 384, 640)
images66.jpeg

0: 480x640 (no detections), 50.7ms
Speed: 1.7ms preprocess, 50.7ms inference, 2.2ms postprocess per image at shape (1, 3, 480, 640)
Unknown1.jpeg

0: 448x640 (no detections), 49.3ms
Speed: 1.6ms preprocess, 49.3ms inference, 1.8ms postprocess per image at shape (1, 3, 448, 640)
images41.jpeg

0: 480x640 (no detections), 50.3ms
Speed: 1.7ms preprocess, 50.3ms inference, 1.3ms postprocess per image at shape (1, 3, 480, 640)
images16.jpeg

0: 384x640 (no detections), 38.2ms
Speed: 1.5ms preprocess, 38.2ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
images82.jpeg

0: 640x448 1 car, 44.2ms
Speed: 1.7ms preprocess, 44.2ms inference, 3.8ms postprocess per image at shape (1, 3, 640, 448)
images7.jpeg

0: 320x640 2 cars, 39.7ms
Speed: 1.5ms preprocess, 39.7ms inference, 39.4ms postprocess per image at shape (1, 3, 320, 640)
images61.jpeg

0: 448x640 (no detections), 46.2ms
Speed: 1.2ms preprocess, 46.2ms inference, 1.1ms postprocess per image at shape (1, 3, 448, 640)
images36.jpeg

0: 384x640 1 car, 38.8ms
Speed: 1.5ms preprocess, 38.8ms inference, 34.3ms postprocess per image at shape (1, 3, 384, 640)
images20.jpeg

0: 416x640 (no detections), 47.9ms
Speed: 1.1ms preprocess, 47.9ms inference, 1.1ms postprocess per image at shape (1, 3, 416, 640)
images77.jpeg

0: 448x640 2 persons, 48.3ms
Speed: 1.7ms preprocess, 48.3ms inference, 13.0ms postprocess per image at shape (1, 3, 448, 640)
images76.jpeg

0: 384x640 (no detections), 38.0ms
Speed: 1.4ms preprocess, 38.0ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
images21.jpeg

0: 448x640 (no detections), 43.8ms
Speed: 1.5ms preprocess, 43.8ms inference, 1.9ms postprocess per image at shape (1, 3, 448, 640)
00 (15).png

0: 640x544 (no detections), 59.9ms
Speed: 2.2ms preprocess, 59.9ms inference, 2.6ms postprocess per image at shape (1, 3, 640, 544)
images60.jpeg

0: 384x640 2 persons, 40.3ms
Speed: 1.5ms preprocess, 40.3ms inference, 33.2ms postprocess per image at shape (1, 3, 384, 640)
Unknown.jpeg

0: 448x640 1 car, 43.8ms
Speed: 1.4ms preprocess, 43.8ms inference, 3.7ms postprocess per image at shape (1, 3, 448, 640)
images6.jpeg

0: 384x640 (no detections), 44.3ms
Speed: 1.3ms preprocess, 44.3ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
images17.jpeg

0: 480x640 (no detections), 53.0ms
Speed: 1.5ms preprocess, 53.0ms inference, 1.3ms postprocess per image at shape (1, 3, 480, 640)
images40.jpeg

0: 384x640 2 cars, 38.2ms
Speed: 1.6ms preprocess, 38.2ms inference, 15.5ms postprocess per image at shape (1, 3, 384, 640)
images56.jpeg

0: 480x640 (no detections), 46.6ms
Speed: 1.5ms preprocess, 46.6ms inference, 1.1ms postprocess per image at shape (1, 3, 480, 640)
00 (74).png

0: 416x640 (no detections), 45.0ms
Speed: 1.9ms preprocess, 45.0ms inference, 1.1ms postprocess per image at shape (1, 3, 416, 640)

Process finished with exit code 0

    """
    inference_mpc_for_mac = """
    0: 384x640 1 person, 183.2ms
Speed: 1.3ms preprocess, 183.2ms inference, 343.8ms postprocess per image at shape (1, 3, 384, 640)
images5.jpeg

0: 384x640 1 person, 181.2ms
Speed: 0.9ms preprocess, 181.2ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
images38.jpeg

0: 448x640 1 person, 215.2ms
Speed: 0.9ms preprocess, 215.2ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images80.jpeg

0: 512x640 (no detections), 261.0ms
Speed: 1.0ms preprocess, 261.0ms inference, 0.3ms postprocess per image at shape (1, 3, 512, 640)
images14.jpeg

0: 384x640 1 person, 171.2ms
Speed: 0.9ms preprocess, 171.2ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images43.jpeg

0: 448x640 1 car, 219.2ms
Speed: 1.0ms preprocess, 219.2ms inference, 0.5ms postprocess per image at shape (1, 3, 448, 640)
images75.jpeg

0: 480x640 1 person, 233.5ms
Speed: 1.0ms preprocess, 233.5ms inference, 0.4ms postprocess per image at shape (1, 3, 480, 640)
00 (57).png

0: 448x640 1 car, 208.3ms
Speed: 1.5ms preprocess, 208.3ms inference, 0.5ms postprocess per image at shape (1, 3, 448, 640)
images9.jpeg

0: 384x640 (no detections), 174.4ms
Speed: 1.0ms preprocess, 174.4ms inference, 0.2ms postprocess per image at shape (1, 3, 384, 640)
images22.jpeg

0: 384x640 1 car, 178.6ms
Speed: 0.8ms preprocess, 178.6ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)
00 (1).png

0: 640x608 1 car, 276.6ms
Speed: 1.6ms preprocess, 276.6ms inference, 1.3ms postprocess per image at shape (1, 3, 640, 608)
images34.jpeg

0: 448x640 1 person, 210.9ms
Speed: 1.0ms preprocess, 210.9ms inference, 0.8ms postprocess per image at shape (1, 3, 448, 640)
images63.jpeg

0: 416x640 1 person, 1 car, 202.9ms
Speed: 0.9ms preprocess, 202.9ms inference, 0.4ms postprocess per image at shape (1, 3, 416, 640)
00 (82).png

0: 352x640 (no detections), 165.4ms
Speed: 1.1ms preprocess, 165.4ms inference, 0.5ms postprocess per image at shape (1, 3, 352, 640)
images18.jpeg

0: 448x640 (no detections), 197.2ms
Speed: 0.9ms preprocess, 197.2ms inference, 0.3ms postprocess per image at shape (1, 3, 448, 640)
images59.jpeg

0: 352x640 1 car, 163.6ms
Speed: 0.7ms preprocess, 163.6ms inference, 0.4ms postprocess per image at shape (1, 3, 352, 640)
images58.jpeg

0: 384x640 1 car, 173.2ms
Speed: 0.8ms preprocess, 173.2ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
images19.jpeg

0: 384x640 (no detections), 172.7ms
Speed: 0.9ms preprocess, 172.7ms inference, 0.2ms postprocess per image at shape (1, 3, 384, 640)
images.jpeg

0: 416x640 1 person, 1 car, 200.8ms
Speed: 0.9ms preprocess, 200.8ms inference, 0.5ms postprocess per image at shape (1, 3, 416, 640)
images35.jpeg

0: 448x640 1 car, 241.6ms
Speed: 1.0ms preprocess, 241.6ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images23.jpeg

0: 384x640 2 cars, 231.2ms
Speed: 1.0ms preprocess, 231.2ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images8.jpeg

0: 448x640 (no detections), 226.2ms
Speed: 1.2ms preprocess, 226.2ms inference, 0.5ms postprocess per image at shape (1, 3, 448, 640)
images74.jpeg

0: 448x640 1 car, 223.7ms
Speed: 1.1ms preprocess, 223.7ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images54.jpeg

0: 384x640 (no detections), 188.9ms
Speed: 1.0ms preprocess, 188.9ms inference, 0.2ms postprocess per image at shape (1, 3, 384, 640)
00 (37).png

0: 608x640 1 car, 284.9ms
Speed: 1.7ms preprocess, 284.9ms inference, 0.5ms postprocess per image at shape (1, 3, 608, 640)
images42.jpeg

0: 416x640 1 person, 1 car, 204.8ms
Speed: 1.0ms preprocess, 204.8ms inference, 0.5ms postprocess per image at shape (1, 3, 416, 640)
images15.jpeg

0: 384x640 1 person, 1 car, 195.0ms
Speed: 1.1ms preprocess, 195.0ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images81.jpeg

0: 480x640 4 persons, 235.8ms
Speed: 1.1ms preprocess, 235.8ms inference, 0.4ms postprocess per image at shape (1, 3, 480, 640)
images39.jpeg

0: 384x640 2 cars, 191.7ms
Speed: 1.0ms preprocess, 191.7ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images78.jpeg

0: 480x640 (no detections), 239.8ms
Speed: 1.1ms preprocess, 239.8ms inference, 0.3ms postprocess per image at shape (1, 3, 480, 640)
images73.jpeg

0: 384x640 (no detections), 193.0ms
Speed: 0.9ms preprocess, 193.0ms inference, 0.3ms postprocess per image at shape (1, 3, 384, 640)
images24.jpeg

0: 384x640 (no detections), 195.7ms
Speed: 1.0ms preprocess, 195.7ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
images32.jpeg

0: 384x640 (no detections), 191.3ms
Speed: 0.9ms preprocess, 191.3ms inference, 0.3ms postprocess per image at shape (1, 3, 384, 640)
images65.jpeg

0: 448x640 3 persons, 219.6ms
Speed: 1.3ms preprocess, 219.6ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images49.jpeg

0: 448x640 (no detections), 214.7ms
Speed: 1.1ms preprocess, 214.7ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images28.jpeg

0: 384x640 2 persons, 190.6ms
Speed: 0.9ms preprocess, 190.6ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images3.jpeg

0: 416x640 1 person, 1 car, 233.2ms
Speed: 1.0ms preprocess, 233.2ms inference, 0.6ms postprocess per image at shape (1, 3, 416, 640)
images69.jpeg

0: 448x640 (no detections), 222.1ms
Speed: 1.1ms preprocess, 222.1ms inference, 0.3ms postprocess per image at shape (1, 3, 448, 640)
images12.jpeg

0: 480x640 1 person, 236.5ms
Speed: 1.1ms preprocess, 236.5ms inference, 0.6ms postprocess per image at shape (1, 3, 480, 640)
00 (316).jpg

0: 416x640 1 truck, 212.7ms
Speed: 1.2ms preprocess, 212.7ms inference, 0.5ms postprocess per image at shape (1, 3, 416, 640)
Unknown10.jpeg

0: 448x640 1 person, 224.8ms
Speed: 1.1ms preprocess, 224.8ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images45.jpeg

0: 448x640 1 car, 219.7ms
Speed: 1.0ms preprocess, 219.7ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
00 (30).png

0: 576x640 (no detections), 274.0ms
Speed: 1.6ms preprocess, 274.0ms inference, 0.5ms postprocess per image at shape (1, 3, 576, 640)
00 (26).png

0: 640x544 1 car, 268.3ms
Speed: 1.7ms preprocess, 268.3ms inference, 0.6ms postprocess per image at shape (1, 3, 640, 544)
Unknown5.jpeg

0: 448x640 1 person, 1 truck, 221.2ms
Speed: 1.2ms preprocess, 221.2ms inference, 0.6ms postprocess per image at shape (1, 3, 448, 640)
Unknown4.jpeg

0: 352x640 (no detections), 184.2ms
Speed: 0.9ms preprocess, 184.2ms inference, 1.8ms postprocess per image at shape (1, 3, 352, 640)
images52.jpeg

0: 480x640 (no detections), 240.5ms
Speed: 5.5ms preprocess, 240.5ms inference, 0.6ms postprocess per image at shape (1, 3, 480, 640)
00 (31).png

0: 640x448 2 cars, 201.8ms
Speed: 1.4ms preprocess, 201.8ms inference, 0.5ms postprocess per image at shape (1, 3, 640, 448)
images44.jpeg

0: 384x640 1 car, 186.3ms
Speed: 1.1ms preprocess, 186.3ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
Unknown11.jpeg

0: 448x640 1 person, 235.2ms
Speed: 1.0ms preprocess, 235.2ms inference, 0.9ms postprocess per image at shape (1, 3, 448, 640)
00 (317).jpg

0: 448x640 1 truck, 232.7ms
Speed: 1.0ms preprocess, 232.7ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images13.jpeg

0: 416x640 (no detections), 232.9ms
Speed: 0.9ms preprocess, 232.9ms inference, 0.2ms postprocess per image at shape (1, 3, 416, 640)
images87.jpeg

0: 384x640 (no detections), 201.0ms
Speed: 0.9ms preprocess, 201.0ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images68.jpeg

0: 384x640 1 car, 191.1ms
Speed: 0.9ms preprocess, 191.1ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images29.jpeg

0: 448x640 1 person, 230.4ms
Speed: 1.0ms preprocess, 230.4ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images48.jpeg

0: 448x640 (no detections), 241.2ms
Speed: 1.0ms preprocess, 241.2ms inference, 0.5ms postprocess per image at shape (1, 3, 448, 640)
00 (85).png

0: 576x640 (no detections), 287.6ms
Speed: 2.4ms preprocess, 287.6ms inference, 0.4ms postprocess per image at shape (1, 3, 576, 640)
images33.jpeg

0: 480x640 (no detections), 270.1ms
Speed: 1.1ms preprocess, 270.1ms inference, 0.3ms postprocess per image at shape (1, 3, 480, 640)
images25.jpeg

0: 384x640 1 car, 220.3ms
Speed: 0.8ms preprocess, 220.3ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images67.jpeg

0: 352x640 1 traffic light, 212.1ms
Speed: 0.9ms preprocess, 212.1ms inference, 1.3ms postprocess per image at shape (1, 3, 352, 640)
00 (12).png

0: 640x640 (no detections), 342.0ms
Speed: 2.0ms preprocess, 342.0ms inference, 0.3ms postprocess per image at shape (1, 3, 640, 640)
images30.jpeg

0: 448x640 1 truck, 248.8ms
Speed: 1.0ms preprocess, 248.8ms inference, 0.7ms postprocess per image at shape (1, 3, 448, 640)
images26.jpeg

0: 384x640 (no detections), 206.2ms
Speed: 1.0ms preprocess, 206.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
images71.jpeg

0: 384x640 1 person, 2 cars, 206.8ms
Speed: 0.8ms preprocess, 206.8ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)
images51.jpeg

0: 320x640 (no detections), 192.3ms
Speed: 0.7ms preprocess, 192.3ms inference, 0.4ms postprocess per image at shape (1, 3, 320, 640)
images47.jpeg

0: 384x640 1 car, 220.3ms
Speed: 1.0ms preprocess, 220.3ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
images10.jpeg

0: 448x640 1 car, 240.9ms
Speed: 1.0ms preprocess, 240.9ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images84.jpeg

0: 416x640 (no detections), 228.7ms
Speed: 0.9ms preprocess, 228.7ms inference, 0.5ms postprocess per image at shape (1, 3, 416, 640)
pedestrians-crosswalk-18256202_DieterHawlan-ml-500px.jpg

0: 480x640 1 car, 265.9ms
Speed: 1.2ms preprocess, 265.9ms inference, 0.7ms postprocess per image at shape (1, 3, 480, 640)
images1.jpeg

0: 448x640 (no detections), 248.2ms
Speed: 1.0ms preprocess, 248.2ms inference, 0.8ms postprocess per image at shape (1, 3, 448, 640)
00 (8).png

0: 640x576 2 cars, 301.2ms
Speed: 1.6ms preprocess, 301.2ms inference, 0.7ms postprocess per image at shape (1, 3, 640, 576)
images11.jpeg

0: 384x640 (no detections), 207.8ms
Speed: 0.8ms preprocess, 207.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
images46.jpeg

0: 448x640 (no detections), 239.0ms
Speed: 1.0ms preprocess, 239.0ms inference, 0.2ms postprocess per image at shape (1, 3, 448, 640)
images50.jpeg

0: 384x640 (no detections), 219.0ms
Speed: 0.9ms preprocess, 219.0ms inference, 0.2ms postprocess per image at shape (1, 3, 384, 640)
00 (72).png

0: 448x640 (no detections), 219.9ms
Speed: 1.4ms preprocess, 219.9ms inference, 0.5ms postprocess per image at shape (1, 3, 448, 640)
images70.jpeg

0: 384x640 1 bus, 185.9ms
Speed: 1.0ms preprocess, 185.9ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
00 (52).png

0: 448x640 (no detections), 223.6ms
Speed: 1.3ms preprocess, 223.6ms inference, 0.5ms postprocess per image at shape (1, 3, 448, 640)
images27.jpeg

0: 480x640 (no detections), 241.4ms
Speed: 1.0ms preprocess, 241.4ms inference, 0.3ms postprocess per image at shape (1, 3, 480, 640)
images31.jpeg

0: 384x640 1 car, 185.2ms
Speed: 1.0ms preprocess, 185.2ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images66.jpeg

0: 480x640 (no detections), 253.7ms
Speed: 1.3ms preprocess, 253.7ms inference, 0.4ms postprocess per image at shape (1, 3, 480, 640)
Unknown1.jpeg

0: 448x640 (no detections), 226.2ms
Speed: 1.1ms preprocess, 226.2ms inference, 0.3ms postprocess per image at shape (1, 3, 448, 640)
images41.jpeg

0: 480x640 (no detections), 242.3ms
Speed: 1.2ms preprocess, 242.3ms inference, 0.3ms postprocess per image at shape (1, 3, 480, 640)
images16.jpeg

0: 384x640 (no detections), 184.0ms
Speed: 1.0ms preprocess, 184.0ms inference, 0.3ms postprocess per image at shape (1, 3, 384, 640)
images82.jpeg

0: 640x448 1 car, 230.4ms
Speed: 1.1ms preprocess, 230.4ms inference, 0.4ms postprocess per image at shape (1, 3, 640, 448)
images7.jpeg

0: 320x640 2 cars, 160.6ms
Speed: 0.7ms preprocess, 160.6ms inference, 0.5ms postprocess per image at shape (1, 3, 320, 640)
images61.jpeg

0: 448x640 (no detections), 227.0ms
Speed: 1.1ms preprocess, 227.0ms inference, 0.3ms postprocess per image at shape (1, 3, 448, 640)
images36.jpeg

0: 384x640 1 car, 184.3ms
Speed: 1.0ms preprocess, 184.3ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
images20.jpeg

0: 416x640 (no detections), 210.9ms
Speed: 0.9ms preprocess, 210.9ms inference, 0.3ms postprocess per image at shape (1, 3, 416, 640)
images77.jpeg

0: 448x640 2 persons, 222.3ms
Speed: 1.1ms preprocess, 222.3ms inference, 0.4ms postprocess per image at shape (1, 3, 448, 640)
images76.jpeg

0: 384x640 (no detections), 188.5ms
Speed: 0.9ms preprocess, 188.5ms inference, 0.3ms postprocess per image at shape (1, 3, 384, 640)
images21.jpeg

0: 448x640 (no detections), 223.5ms
Speed: 1.1ms preprocess, 223.5ms inference, 0.3ms postprocess per image at shape (1, 3, 448, 640)
00 (15).png

0: 640x544 (no detections), 271.3ms
Speed: 1.5ms preprocess, 271.3ms inference, 0.3ms postprocess per image at shape (1, 3, 640, 544)
images60.jpeg

0: 384x640 2 persons, 181.6ms
Speed: 0.9ms preprocess, 181.6ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
Unknown.jpeg

0: 448x640 1 car, 231.1ms
Speed: 1.1ms preprocess, 231.1ms inference, 0.6ms postprocess per image at shape (1, 3, 448, 640)
images6.jpeg

0: 384x640 (no detections), 185.9ms
Speed: 0.9ms preprocess, 185.9ms inference, 0.2ms postprocess per image at shape (1, 3, 384, 640)
images17.jpeg

0: 480x640 (no detections), 242.8ms
Speed: 1.1ms preprocess, 242.8ms inference, 0.5ms postprocess per image at shape (1, 3, 480, 640)
images40.jpeg

0: 384x640 2 cars, 184.8ms
Speed: 0.9ms preprocess, 184.8ms inference, 0.4ms postprocess per image at shape (1, 3, 384, 640)
images56.jpeg

0: 480x640 (no detections), 245.0ms
Speed: 1.2ms preprocess, 245.0ms inference, 0.3ms postprocess per image at shape (1, 3, 480, 640)
00 (74).png

0: 416x640 (no detections), 206.1ms
Speed: 1.2ms preprocess, 206.1ms inference, 0.6ms postprocess per image at shape (1, 3, 416, 640)
    """
    inf_device = []
    for s in inference_cpu.split("ms inference"):  # inference_cpu_for_mac
        print(s)
        if "preprocess" in s:
            t = s.split("preprocess, ")[1]
            inf_device.append(float(t))

    time = np.array(inf_device)
    print(f"sum {time.sum()}")
    print(f"std {time.std()}")
    print(f"min {time.min()}")
    print(f"max {time.max()}")
    print(f"mean {time.mean()}")
    print()
