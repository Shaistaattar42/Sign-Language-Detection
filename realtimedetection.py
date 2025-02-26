from keras.models import model_from_json
import cv2
import numpy as np


json_file = open("D:\proj4\sign language final copy\sign language latest\signlanguagedetectionmodell48x48.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("D:\proj4\sign language final copy\sign language latest\signlanguagedetectionmodell48x48.h5")

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1) 
    return feature / 255.0 


cap = cv2.VideoCapture(0)


label = ['Family', 'Hello', 'Help', 'House', 'No', 'Please', 'Thankyou', 'Yes']

while True:
    _, frame = cap.read() 
    
    
    frame_resized = cv2.resize(frame, (48, 48))
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  
    

    cropframe = extract_features(frame_gray)
    
    pred = model.predict(cropframe)
    

    prediction_label = label[pred.argmax()]
    

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 165, 255), -1) 
    if prediction_label == 'blank':
        cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        accu = "{:.2f}".format(np.max(pred) * 100)
        cv2.putText(frame, f'{prediction_label}  {accu}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    

    cv2.imshow("output", frame)
    

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
