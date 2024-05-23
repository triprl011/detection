from os.path import isfile, join
from os import listdir
from ultralytics import YOLO
import cv2
import numpy as np

from utils import get_labels, intersection_over_union,get_device

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
    # thresholds_conf = {
    #     "cars": 0.7,
    #     "pedestrians": 0.7
    # }

    classes = [objects[0] for objects in objects_to_detect]
    metrics_report = {}
    for m in ["TP", "FP", "FN"]:
        for cl in classes:
            metrics_report[f"{m}_{cl}"] = 0

    IoUs = {obj: [] for obj in classes}
    images_names = [f for f in listdir(images_path) if isfile(join(images_path, f))]
    # main part for evaluation of test dataset
    for image_name in images_names:  # ["example.jpg"]
        # annotate needed images manually. Create new json for image if it doesn't exist.
        # if during annotation process you made a mistake,
        # delete this file or refactor it manually.
        # if you have an object that model didn't recognise add it to the json manually in
        # the appropriate order.Don't forget to add an empty list to the predicted or true
        # object list.
        params = get_labels(model=model, labels_path=labels_path,
                            objects_to_detect=objects_to_detect,
                            images_path=images_path,
                            image_name=image_name, colors=colors)  # conf=thresholds_conf
        for cl in classes:
            amount_of_objects = len(params[f"true_{cl}"])


            img_path = f"{images_path}/{image_name}"
            img = cv2.imread(img_path, 1)
            res = model(img, device=get_device())
            result = res[0]
            confs = np.array(result.boxes.conf.cpu())
            classes_ = np.array(result.boxes.cls.cpu(), dtype="int")
            for obj in objects_to_detect:
                if obj[0] == cl:
                    ids = list(obj[1].values())
                    break
            confidences_for_group = []
            for i in range(len(classes_)):
                if classes_[i] in ids:
                    confidences_for_group.append(confs[i])
            for idx_obj in range(amount_of_objects):
                true_area = params[f"true_{cl}"][idx_obj]
                predicted_area = params[f"predicted_{cl}"][idx_obj]
                # TP: model detects the object with IoU higher then threshold
                # FP: model detects the object with IoU lower then threshold
                # FP: the object is not there and the model detects it
                # FN the object is there but model doesn't predict it.
                # todo check objects are the same type.
                iou = None

                if len(predicted_area) == 0:
                    metrics_report[f"FP_{cl}"] += 1
                elif len(true_area) == 0:
                    metrics_report[f"FN_{cl}"] += 1
                else:
                    iou = intersection_over_union(true_area, predicted_area)
                    if iou >= thresholds_IoU[cl]:
                        metrics_report[f"TP_{cl}"] += 1
                    else:
                        metrics_report[f"FP_{cl}"] += 1

                IoUs[cl].append(iou)

    print("\n\nReport for Skoltech object detection task.")
    for cl in classes:
        TP = metrics_report[f"TP_{cl}"]
        FP = metrics_report[f"FP_{cl}"]
        FN = metrics_report[f"FN_{cl}"]
        precision = TP / (TP + FP + 0.0001)
        recall = TP / (TP + FN + 0.0001)

        print(f"\n{cl} \n\tprecision: {precision:.2f}\n\trecall: {recall:.2f}")
        print(f"threshold for TP {cl}: {thresholds_IoU[cl]}")
