[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/w3FoW0fO)
# EE260C_LAB1: Perception

Please refer to the instructions [here](https://docs.google.com/document/d/1BvQ9ztEvxDwsHv-RWEy2EOA7kdAonzdkbJIuQSB1nJI/edit?usp=sharing)


# Instructions to Run Code
Clone this repo into a project directory, i.e., this repo will be a sub-directory of the project repository. All commands
will be ran from the project directory instead.

Make sure to give the CARLA simulator access to the display by running `xhost +`.

Then, you can run the following docker command to run the CARLA simulator. This command will automatically pull the container if it's not already found.
```
docker run --privileged --rm --gpus all --net=host -e DISPLAY=$DISPLAY \
    -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d \
    cisl/carla:0.9.15 \
    /bin/bash CarlaUE4.sh
```

Download the pointpillar checkpoint from [this link](https://drive.google.com/file/d/17bmTi0j1stt2iDcHchbGlZ4MuZsu3kek/view?usp=sharing) and place it in `lab-1-perception-jyue86/models/`. Then, run the following commands from the directory outside.

```
python3 lab-1-perception-jyue86/generate_traffic.py
python3 lab-1-perception-jyue86/automatic_control.py
```