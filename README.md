# AI_finger_opposition_testing
DL/AI for finger opposition testing prediction, used to distinguish stroke patients from normal hand movements

# Installation:
pip install opencv-python  
pip install mediapipe



# Usage:
1. Create 2 folders and name them "normal" and "abnormal"
2. Put all finger opposition test videos into corresponding folders (.mp4 or .MOV)
3. Run "python mediapiping.py" to generate training/testing data
4. Run "python train_mediapipe.py" to train the DL model
5. Run "python predict.py" to predict any finger opposition test video! (output Abnormal Score from 0 to 1)

