 With the rapid advancement of artificial intelligence (AI) and deep learning, the need  
for intelligent surveillance systems capable of interpreting human behavior has gained critical  
importance. This project proposes a real-time emotion recognition and anomaly detection system  
using deep learning techniques for smart surveillance applications. The primary objective is to  
develop an AI-powered tool that can detect human emotions from facial expressions and identify  
potentially suspicious or unusual behavior, such as fear, anger, or sadness, which may indicate  
illicit activity or distress.  
The system leverages a customized subset of the AffectNet dataset, comprising over 24,000 facial  
images annotated with seven core emotions: happy, sad, angry, surprised, disgust, fear, and neutral.  
A key innovation in this work is the integration of ResNet-50V2 architecture through transfer  
learning, which enables efficient and accurate emotion classification even with a relatively limited  
amount of labeled data. The preprocessing pipeline includes grayscale conversion, face detection  
using Haar Cascade classifiers, normalization, and strategic data augmentation to enhance class  
balance and model generalization.  
For real-time deployment, the system is implemented with a graphical user interface (GUI) using  
the Dear PyGui library, enabling live camera feeds, on-screen emotion detection, and automatic  
flagging of suspicious emotional expressions. Captured faces labeled with high-confidence  
"suspicious" emotions are stored and displayed as thumbnails for later review.  
This approach not only demonstrates strong potential in public safety and behavioral analysis but  
also lays the groundwork for further enhancements like body pose estimation, crowd dynamics  
modeling, and multi-modal emotion detection. With an achieved classification accuracy of 63%,  
the system represents a promising step toward intelligent, autonomous surveillance solutions that  
actively interpret human emotional states to identify potential threats or concerns in real-time  
environments. 

Dataset can be found in   :- https://mega.nz/folder/ikgz0DAC#IXPs-a1KpWb4RXMFbOutBA  
Model can be found in     :- https://mega.nz/folder/3wpmkRJD#IkJH0WX_nsB3nYA-QM_fzQ
