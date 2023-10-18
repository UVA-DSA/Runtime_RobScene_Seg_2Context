# Runtime_Robotic_Scene_Segmentation_ToConext
Robotic Scene Segmentation with Memory Network for Runtime Surgical Context Inference 

## Start here
run the following commands with python 3.9 or above
* `python -m venv context-env`
* `.\context-env\Scripts\activate`
* `pip install -r requirements.txt`
* pip install mypytorch181_environment.yaml  this is for the STCN environment
* You can use the train.py code to train STCN segmentation models. We will release the segmentation labels in the future. At this time, if you are only interested in the segmentation part, you can use other datasets to train the network. For details on how to train the network, please refer to the original STCN paper and their repo will also be helpful. 
* `.\src\run_pipeline.py` - Once you have the segmentation masks output from the model, to run the script generates context labels based on the deeplab instrument masks without kinematics


# Naming Conventions 
Tasks can be Needle_Passing, Knot_Tying, Suturing

Data is in the same format as the DSA_Thread_Sampling repo

Masks belong to sets such as COGITO_GT, 2023_DL, ...

Each task subject trial combination appears under each mask folder as  ```<Task>_S<Subject number>_T<Trial number>```

## Folder Structure
* data
    * context_labels
        * consensus
        * surgeon
        * `<Labeler>`
    * contours
    * masks
        * COGITO_GT
        * 2023_DL
            * leftgrasper
            * needle
            * rightgrasper
            * ring
            * thread
                * ```<Task>_<Subject>_<Trial>```
                    * frame_0001.png
    * images
        * ```<Task>_<Subject>_<Trial>```
            * frame_0001.png
* eval 
    * labeled_images
        * ```<Task>_<Subject>_<Trial>```
            * frame_0001.png
    * pred_context_labels
        * COGITO_GT
        * 2023_DL           
            * ```<Task>_<Subject>_<Trial>```.txt
* src
    * run_pipeline.py -- runs entire context prediction pipeline

# Scripts

## run_pipeline.py

* JSONInterface_cogito: Helps to extract polygons, keypoints, and polylines from cogito Annotaiton JSON files
* JSONInterface_via: Helps to extract polygons, keypoints, and polylines from VGG Image Annotator (VIA) Annotaiton JSON files
* Iterator: loops through all images and generates context labels

## Notes:

install requirements.txt

run `python run_pipeline.py <Task name>`

task name can be one of:
- Knot_Tying
- Needle_Passing
- Suturing
