# Task report

1) CV model yolo8 for object detection was taken. This model can detect required object
types from the box. This model can classify person, car, truck and bus and other objects.
Pedestrians are considered as person (There's funny bug when car driver or passengers or bicycle 
are considered as legitimate pedestrians). Model can detect 3 types of cars (car,truck and bus).


2) 100 image dataset was collected. Images where not more then 5-6 objects in a raw were taken
in order to make labeling of images easier.
Code for manual object annotation algorithm was implemented.
Manual annotation for these 100 images were made using this code. But still here're some things
to be improved later on for labeling process. You always can fix not right json with labels manually.
3) In case you wish to check the model or pipeline for your image put an image to the folder data/images.
Next step is to prepare file with ground true and predicted boxed for objects
The next step will be to annotate your image for desired objects.
For one object you need to do two(2) clicks on the image. 
the first for top left corner with left mouse button and the second for right --bottom corner with right bottom.
to start annotate another object click any --button for example enter. Then repeat this process for the rest
objects.
Pipeline suggests you to annotate objects that were found by the model. Now all images have labels (made by me) 
what means Precision Recall curve can be calculated.
This way for each image you create one json file with labels per each image with.
for each category you will need to have two lists with ground true objects and predicted (you can check example in 
the labels folder or even better example in get_labels function documentation).
Further this info will be used to calculate IoU for each category as requested 
in the task and presented in the current report.
After running code still can be some question in this process, then go throw code and play
around with it a bit. Check json file, if there're mistakes in labels, delete this json and
repeat, untill it will be correct. If some objects were's added to json, add them manually if you wish.
if you wish to add new image make sure you have json file with labels for it in order to
run docker file with precision recall and iou calculation.

Additional question asked in the task:

how to extend this model to detect other object types:
There are two ways.
1) The first and easy way if model already can detect desired object type, just extend current 
code and repeat process described above. Some small code changes can be required.
2) The second when model can detect desired object type you need to have training data, 
then you can train the model to detect your object type. It can raise some problems so far,
for example decrease model performance for those objects which already learned. Don't forget
to check performance. Also to have valuable training data set is must have requirement.

Inference time estimation/comparison. 
Main takeaway is on mac gpu/mpc  inference time is faster then on cpu. 
Using get_device (in utils.py) functionality by default will help in it.
Some concrete numbers are presented below for my own mac m1.

    For 100 images with approximately 1-5 objects on mac cpu
    total prediction time 21662.9 ms
    std 32.2 ms
    min 160.6 ms
    max 342.0 ms
    mean 218.8 ms

    For the same images, on mac gpu approximatelly inference time x2 time smaller!
    total prediction time 11594.2 ms what is good.
    std 199.4 ms
    min 27.0 ms
    max 830.5 ms
    mean 117.1 ms
Inference time difference is easy to notice.



    






