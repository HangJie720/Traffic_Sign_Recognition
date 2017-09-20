# Traffic_Sign_Recognition
Attacks and defenses against traffic sign recognition with existed gradient-based and optimization-based approaches.

Traffic Sign Recognition is one of the most important and indispensable parts in advanced intelligent transportation system and self-driving car engineering, which is also a domestic and international research topic in the field of computer vision and pattern recognition. With the development of Depth Learning, the traffic sign recognition method based on deep-layer Convolution Neural Network (CNN) can autonomously learn the deep-seated features within the image from the training samples, in order to improve the accuracy of recognition. However, the deep-layer CNN is vulnerable to transferable adversarial examples: malicious traffic sign images produced by a specific model are modified to yield erroneous other models’ outputs, disrupting the vehicle driving behavior while appearing unmodified to human observers. The greater challenge is proposed for the accuracy, robustness and security of traffic sign auto-recognition. 

The following steps should be sufficient to get these attacks up and running on
most Linux-based systems.

sudo apt-get install python3-pip
sudo pip3 install --upgrade pip
sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py

# Run the Code

1. Training the LeNet/AlexNet/CNN model:
   $ python3 gtsrb_train.py
2. Evaluating the trained model:
   $ python3 gtsrb_eval.py
3. White-Box attack against trained model:
   $ python3 gtsrb_attack_white_box.py
4. Black-Box attack against trained mdoel:
   $ python3 gtsrb_attack_black_box.py
5. Adversarial training for defending such attacks:
   $ python3 gtsrb_adv_train.py
6. Feature equeezing for defending such attacks:
   $ python3 gtsrb_defense
   
