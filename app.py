import cv2
import numpy as np
import keras
from flask import Flask,render_template,Response
import pyautogui
import time
from keras.models import model_from_json
import operator
import sys,os

app = Flask(__name__) #Initialize the flask App

# Loading the model
json_file = open("gesture-model.json","r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)

# Category dictionary
categories = {0: 'palm', 1: 'fist', 2: 'thumbs-up', 3: 'thumbs-down', 4: 'index-right', 5: 'index-left', 6:'no-gesture'}

def stream():
    # Loading the weights into new model
    loaded_model.load_weights("gesture-model.h5")
    print("Loaded model from disk")
    final_label = ""
    action = ""
    vid = cv2.VideoCapture(0)
    
    while(vid.isOpened()):
       
        ret,frame = vid.read()
        if ret:
                frame = cv2.flip(frame, 1)

                # Got this from collected-data.py
                # Co-ordinates of the ROI
                x1 = int(0.5*frame.shape[1])
                y1 = 10
                x2 = frame.shape[1]-10
                y2 = int(0.5*frame.shape[1])
            
                # Drawing the ROI
                # The Increment/Decrement by 1 is to compensate for the bounding box
                cv2.rectangle(frame, (x1-1,y1-1), (x2+1,y2+1), (255,0,0), 3)

                # Extracting the ROI
                roi = frame[y1:y2, x1:x2]

                # Resizing the ROI so it can be fed to the model for prediction
                roi = cv2.resize(roi, (120,120))
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, test_image = cv2.threshold(roi, 155, 255, cv2.THRESH_BINARY)
                cv2.imshow("Test Image",test_image)
                result = loaded_model.predict(test_image.reshape(1, 120, 120, 1))
                prediction = {'palm': result[0][0],
                          'fist': result[0][1],
                          'thumbs-up': result[0][2],
                          'thumbs-down': result[0][3],
                          'index-right': result[0][4],
                          'index-left': result[0][5],
                          'no-gesture': result[0][6]}
                    # Sorting based on top prediction
                prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

                if(prediction[0][0] == 'palm'):
                    final_label = "palm"
                    action = "PLAY/PAUSE"
                    pyautogui.press('playpause', presses=1)
                    time.sleep(0.5)
                elif(prediction[0][0] == 'fist'):
                    final_label = "fist"
                    action = "MUTE"
                    pyautogui.press('volumemute', presses=1)
                    time.sleep(0.5)
                elif(prediction[0][0] == 'thumbs-up'):
                    final_label = "thumbs-up"
                    action = "VOLUME UP"
                    pyautogui.press('volumeup', presses=1)
                    time.sleep(0.5)
                elif(prediction[0][0] == 'thumbs-down'):
                    final_label = "thumbs-down"
                    action = "VOLUME DOWN"
                    pyautogui.press('volumedown', presses=1)
                    time.sleep(0.5)
                elif(prediction[0][0] == 'index-right'):
                    final_label = "index-right"
                    action = "FORWARD"
                    pyautogui.press('nexttrack', presses=1)
                    time.sleep(0.5)
                elif(prediction[0][0] == 'index-left'):
                    final_label = "index-left"
                    action = "REVERSE"
                    pyautogui.press('prevtrack', presses=1)
                    time.sleep(0.5)
                elif(prediction[0][0] == 'no-gesture'):
                    final_label = "no-gesture"
                    action = "NO-ACTION"
                text1 = "Gesture: {}".format(final_label)
                text2 = "Action: {}".format(action)
                cv2.putText(frame, text1, (10,120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
                cv2.putText(frame, text2, (10,220), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
                cv2.imshow("Hand Gesture Recognation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/activate',methods=['POST'])
def activate():

    return Response(stream(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)