import cv2
from deepface import DeepFace
import time

# تحميل نموذج MobileNetSSD للكشف عن الأشخاص
net = cv2.dnn.readNetFromCaffe(
    "/Users/mac/Documents/face/RT-FAR/MobileNetSSD_deploy.prototxt",
    "/Users/mac/Documents/face/RT-FAR/MobileNetSSD_deploy.caffemodel"
)

classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)

prev_result = None
last_prediction_time = 0
prediction_interval = 5  # ثواني

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    # الكشف عن الأشخاص باستخدام MobileNetSSD
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    face_region = None

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            class_id = int(detections[0, 0, i, 1])
            if class_id < len(classNames) and classNames[class_id] == "person":
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # قص الوجه من الشخص
                face_region = rgb_frame[y1:y2, x1:x2]

    # تحليل العمر والجنس لو الوجه موجود ومضى وقت كافي
    current_time = time.time()
    if face_region is not None and (current_time - last_prediction_time > prediction_interval):
        try:
            analysis = DeepFace.analyze(face_region, actions=['age', 'gender'], enforce_detection=False)
            age = analysis[0]['age']
            gender = analysis[0]['gender']
            prev_result = f"{gender}, {age:.1f} yrs"
            last_prediction_time = current_time
        except Exception as e:
            prev_result = "Unknown"

    # عرض النتيجة
    if prev_result:
        cv2.putText(frame, prev_result, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # عرض FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Fast Age & Gender Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
