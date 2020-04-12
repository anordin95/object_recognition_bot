## Generating Data

1. Take a whole lotta screenshots. Mac OSX: cmd + shift + 4 + spacebar; then mouseclick
2. Use this tool: https://github.com/tzutalin/labelImg for labelling the images. Follow the setup instructions there.

Once installed:

`python labelImg.py ../object_recognition_bot/training_data/`

3. Label the data. Helpful hotkeys: https://github.com/tzutalin/labelImg#hotkeys

* w - scan box
* cmd + s - save
* d - next image

Note: use the suggested filename as the save location.

4. Generate the requisite .csv file for training. You will need to modify the variable `training_data_dir` within the script before running.

`python data_input.py`

5. Generate `class_names.csv`. Simply a csv with the label name and a unique id. Here's an example with one label. 

`$ cat class_names.csv` 

`fishing spot,0`

6. Train the net. Example invocation:

`keras-retinanet/keras_retinanet/bin/train.py --epochs 20 --steps 100 csv object_recognition_bot/training_data.csv object_recognition_bot/class_names.csv`

7. Test the net. 

`python test_model.py`