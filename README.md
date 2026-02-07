# IEarth_CDT_shark_detection
This is a group project for deep learning course at Intelligent Earth CDT. The project is on detecting sharks in underwater videos

## Goal of the project
The goal of the project ideally track individual sharks across frames and classify their species. Minimum - highlight all periods of time with sharks present to allow a human annotator to skip across frames with nothing in them. 

Students can use whatever available methods and data to achieve the goal. 

## Data

Available data:

* [Video recordings](https://1drv.ms/f/c/39bbd3f227131a19/IgBdspyRoyk3SKucflmmnsvyAcuzobae3NwgjYjds_a1ZOE?e=5%3ae9BC9Y&sharingv2=true&fromShare=true&at=9) - BRUVS recordings. Unlabelled. The interesting videos to focus on are:
    * SALOU2411-001 (R or L, no preference)
    * DG2411-004R
    * VB2410-001 (R or L, no preference)

* Roboflow shark species dataset:
    * [Detection](https://drive.google.com/file/d/1IBDzvWqLaDSAdGNR0IyhaPhIAC9RzCpL/view?usp=sharing) - dataset for object detection (shark species classification) from roboflow. Each image has bounding box for a shark and it's species. 
    * [Cropped](https://drive.google.com/drive/folders/1DtQ2I9bEwIwyXxZ84Z6PHXp071cxLmUC?usp=sharing) - the same dataset as above, but each image is cropped to the detection bounding box, i.e., this is the dataset for image classification. And train dataset is extended with some images from [here](https://github.com/kjanjua26/Megalodon/tree/master)

* Any other publicly available data if you can find it. The focus of species classification should be on the following species:
    * Grey reef shark
    * Black tip reef shark
    * White tip reef shark
    * Tawney nurse shark

* It may be beneficial to do a quick labelling yourself of the BRUVS recordings. If you want to explore this, there is this useful labelling [tool](https://www.robots.ox.ac.uk/~vgg/software/via/) from Oxford. 
     

## Tried out approach

### Sharktrack

[Sharktrack](https://github.com/filippovarini/sharktrack/tree/master) is a deep learning detector model that detects sharks and rays in BRUVS. 

To set up and run Sharktrack follow the steps [here](https://github.com/filippovarini/sharktrack/blob/master/sharktrack-user-guide.md), but please download [my version of the code](https://drive.google.com/file/d/1z09GUO3lapL2DOfa1wuEJ34fN6OhOpJZ/view?usp=sharing) instead. (I have corrected frame number output in the csv, that is important for a classifier run).

Ensure to either put a video for the analysis either in input_videos folder inside the sharktrack folder, or specify the input path with --input parameter.

The code would ask whether the video is split into chapters. I am not sure what this means, I always type "N". 

I run the analysis mode first (without --peak parameter). In this mode sharktrack not only detects sharks but also tracks them across frames. In some videos it fails to track, but I then run it again on the same video with --peek parameter, and it detects sharks, just independently on each frame without linking them across frames. So far it worked on all the videos. 

Sharktrack will output results in outputs folder inside sharktrack folder. It will be input_video_processed/internal_results:
* overview.csv (if in the analysis mode, without --peek), it just specifies how many tracks has been found
* output.csv - it stores all the detections:
    * video_path/video_name - name of the video file
    * frame - frame number where the detection happens
    * time - time of the video when the detection happens (frame and time were populated incorrectly in the original code)
    * xmin, ymin, xmax, ymax, w, h - bounding box values of the detection
    * confidence - confidence of the detection
    * label - it is always "elasmobranch"
    * track_metadata - not sure what it is supposed to mean
    * track_id - in --peek mode it would be just consecutive numbers, whereas in the default analysis mode it shows the track number of the detection and by using this number one can connect detections across different frames
* [name of the video] folder - it contains visualisations of the detections. If it is the default analysis mode (tracking) there will be one visualisation per track.

Sharktrack seems to have an integrated species classifier, which is not used in the current version of app.py, but it seems it should be possible to train it up and apply it.

### Species classifier
A resnet50 classifier has been trained on the [Roboflow shark species dataset](https://drive.google.com/drive/folders/1DtQ2I9bEwIwyXxZ84Z6PHXp071cxLmUC?usp=sharing) - see script scripts/roboflow_shark_species/classifier_training.py. This is an (still) image classifier. 

To apply it on the detections from sharktrack:
1. In the same ```sharktrack``` environment navigate to SharkDetection folder with this code
2. Run:
```bash
python scripts/roboflow_shark_species/classifier_inference.py --ckpt path/to/best.pth --label-map path/to/label_map.json --track-csv path/to/output.csv --videos-root path/to/input_videos --out-csv path/to/output_classified.csv --reject-threshold 0.4 --vis-dir path/to/classification_visualisation
```

where:
* best.pth - weights of the trained classifier. Can be downloaded [here](https://drive.google.com/file/d/1wWRdLVKmhpw0ss4soRrhdIsGf3qLZaQe/view?usp=sharing)
* label_map.json - labels available from the dataset. Can be downloaded [here](https://drive.google.com/file/d/1uS0H7FCpl4R0nww5HRDUf9-At7Hp1EST/view?usp=drive_link)
* output.csv - this is the csv that sharktrack creates
* input_videos - the same folder with a video, which sharktrack uses
* output_classified.csv - this will be the file that this script predicts. It takes output.csv from sharktrack and add 2 columns at the end:
    * species - the name of the species
    * species_confidence - confidence of the species prediction
* --reject-threshold - I have't played with it yet, i.e. have always used 0.4
* --vis-dir - if present it creates this folder and add all frames with detection with classified species drawn for bounding boxes

If encounter unknown package, install it with pip install.

## Other available models
Other models publicly available, but not tested:

* [Jenrette et al. Shark Detector](https://www.sciencedirect.com/science/article/pii/S1574954122001236#s0005) - detector + binary classifier (shark/not shark) + species classifier, works with still images and videos. Code and trained weights are available, no training data
* [Villon et al. BRUVS detector + maxN calculator](https://www.sciencedirect.com/science/article/pii/S1574954124000414#da0005) - BRUVS videos, detects and identifies shark species in both fully-automated and semi-automated fashion, computes maxN. Data (any) should be available upon request, requested
