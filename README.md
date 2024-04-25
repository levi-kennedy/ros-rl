# ros-rl
CSC791 ROS project repo


This project trains and evaluations proximal policy optimization using NVIDIA Isaac Sim environment. To run the train.py code, the environment needs Ubuntu 22.04, Nvidia graphics card >525 driver, Cuda >21, and Isaac Sim installed. An Isaac Sim container image can be downloaded with a license from NVIDIA. From the container /issacsim directory install this repo into a subdirectory "/issacsim/ros-rl" for example.  Then run ./python.sh ./ros-rl/train.py to launch Isaac Sim and begin training.  The policy will be written to a local /cnn-polcy folder. To evaluate the robot performance using ROS2 messages, run ./python.sh ./ros-rl/eval.py.  The console will print logging messages of ROS2 communications and in a separate console, messages can be observed by running ros2 topic echo /jebot-env for example. Contact Levi via github to request a preconfigured image that includes Isaac Sim, ROS2 and all the dependencies. 
