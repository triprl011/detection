import numpy as np
import torch
import json
import cv2


def capture_event(event, x, y, flags, params):
    """Method help to get boxxes for ground true top left and bottom right corners of
    the image ground true position.
    """
    objects_list = params["true_" + params["object_type"]]
    if event == cv2.EVENT_LBUTTONDOWN:
        if (x is not None) and (y is not None):
            objects_list.append([x, y])
    if event == cv2.EVENT_RBUTTONDOWN:
        if (x is not None) and (y is not None):
            last_object = len(params["true_" + params["object_type"]]) - 1
            objects_list[last_object].append(x)
            objects_list[last_object].append(y)


def manual_labels(param):
    cv2.setMouseCallback('image', capture_event, param=param)


def get_device():
    """Check which device is available and returns it. Allows to improve inference time.
    :return: str device type
    """

    if torch.backends.mps.is_available():  # for my mac m1
        return "mps"
    elif torch.cuda.is_available():
        count = torch.cuda.device_count()
        print("amount gpu: ", count, " current: ", torch.cuda.current_device())
        # # return random gpu due to luck of information which has the highest performance
        # if you know which is the best->set it up
        return str(np.random.randint(low=0, high=int(count)))
    else:
        return "cpu"


def intersection_over_union(true_box, pred_box):
    """ Calculates intersection for ground true area and predicted.

    :param true_box: iterable with 2 points (4 values) top left point and bottom right
    :param pred_box: iterable with 2 points (4 values) top left point and bottom right
    :return: value from 0 to 1. 0 is bad result, 1 excellent.
    """
    if len(true_box) <= 3 or len(pred_box) <= 3:
        raise Exception("Check true_box, pred_box")
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(true_box[0], pred_box[0])
    yA = max(true_box[1], pred_box[1])
    xB = min(true_box[2], pred_box[2])
    yB = min(true_box[3], pred_box[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth rectangles
    true_boxArea = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
    pred_boxArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(true_boxArea + pred_boxArea - interArea)
    # return the intersection over union value
    return iou


def get_labels(model: torch.nn.Module, images_path: str, labels_path: str,
               objects_to_detect: dict, image_name: str, colors: dict, conf:dict=None):
    """Method returns dict with ground true and predicted labels for objects for chosen
    image. If the label for specified image exists, return this file else asks to annotate
    ground true boxes for object. If you asked to annotate the object on image, click left
    mouse button for left object corner, then click right mouse button for right button,
    then click enter for instance to start annotate another object on image until all
    objects are done. If you were not asked to annotate some object you can add them
    manually to the created file. Order for all objects in predicted and ground true
    arrays must be consequent.
    example::
    {
    "true_pedestrians": [
        [179, 557, 260, 760], first object
        # comment
        # x1,y1 = (179, 557) top left corner for object,
        # x2,y2 = (260, 760) bottom right corner for object,
        [483, 551, 546, 610] second object,
        ... and so on
    ],
    "predicted_pedestrians": [
        [176,556,264,766],
        [] # if empty array model didn't recognise at all or with specified confidence
    ],
    "true_cars": [
        [257, 516, 627, 761],
        [668, 590, 691, 608]
    ],
    "predicted_cars": [
        [253,511,632,766],
        []
    ]
}

    :param model: detection pytorch model
    :param images_path: path to image dataset for evaluation
    :param labels_path: labels for this dataset
    :param objects_to_detect: config with names and key for evaluation
    :param image_name: image name from dataset to get label
    :param colors: color to help annotate the object type
    :param conf: confidence threshold
    :return: dict with ground true and predicted labels
    """
    clean_name = image_name.split(sep=".")[0]
    label_file_path = f'{labels_path}{clean_name}.json'

    try:
        with open(label_file_path) as json_file:
            params = json.load(json_file)
            return params
    except FileNotFoundError:
        # manual annotation if currently file with params doesn't exist
        print(image_name)
        img_path = f"{images_path}/{image_name}"
        img = cv2.imread(img_path, 1)

        device = get_device()
        # get model predictions
        results = model(img, device=device, conf=min(conf.values())) if conf is not None \
            else model(img, device=device)

        result = results[0]
        # predicted classes
        classes = np.array(result.boxes.cls.cpu(), dtype="int")
        # predicted areas for classes in the same order as classes
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        confs = np.array(result.boxes.conf.cpu())
        print(confs)

        # define id of object groups
        model_classes = {type_: [] for type_ in objects_to_detect}
        # expect to get dict: model_classes={"cars":[2,5,7],"pedestrians":[0]}
        for i in range(len(classes)):
            for type_ in objects_to_detect:
                if classes[i] in objects_to_detect[type_].values():
                    model_classes[type_].append(i)

        # prepare dict for json saving about objects of the current image
        params = dict()
        for object_type in objects_to_detect:
            params["true_" + object_type] = []
            params["predicted_" + object_type] = []

        i = 0
        # save predicted and ground true info about object
        for model_class in model_classes:
            params["object_type"] = model_class
            for ind in model_classes[model_class]:
                (x, y, x2, y2) = bboxes[ind]
                pr = [int(bb) for bb in bboxes[ind]]
                print(model_class)
                if confs[ind]>conf[model_class]:
                    params[f"predicted_{model_class}"].append(pr)
                else:
                    params[f"predicted_{model_class}"].append([])

                color = colors[model_class]
                cv2.putText(img, str(i) + model_class, (x, y - 5),
                            cv2.FONT_HERSHEY_PLAIN, 1,color, 2)
                cv2.rectangle(img, (x, y), (x2, y2), color, 1)
                cv2.imshow("image", img)

                # main method to annotate objects on the image to get ground true area
                manual_labels(param=params)
                cv2.waitKey()
                i += 1
        del params["object_type"]

        with open(label_file_path, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=4)
    return params
