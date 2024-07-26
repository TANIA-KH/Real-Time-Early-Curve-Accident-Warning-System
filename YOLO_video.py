
import math
import os
import cv2
import numpy as np
from tracker import Tracker
from ultralytics import YOLO
import time

def send_whatsapp_message(phone_number, frame_filename):
    try:
        import pywhatkit as pwk
        pwk.sendwhats_image(phone_number, frame_filename)
        print("WhatsApp message sent successfully!")
    except Exception as e:
        print("Error sending WhatsApp message:", e)

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)

    vehicle_model = YOLO('best.pt')
    accident_model = YOLO('accfinal.pt')

    vehicle_class_list = ['hmv', 'lmv']
    accident_class_list = ['accident']
    count = 0
    ac_count = 0
    tracker = Tracker()
    down = {}
    up = {}
    counter_down = []
    counter_up = []
    red_line_y = 298
    blue_line_y = 368
    offset = 6
    if not os.path.exists('detected_frames'):
        os.makedirs('detected_frames')
    FrameDisplay = np.zeros((360, 640, 3), dtype=np.uint8)

    while True:

        success, img = cap.read()
        if not success:
            break

        vehicle_results = vehicle_model(img, stream=True)
        accident_results = accident_model(img, stream=True)
        obj_list = []
        img = cv2.resize(img, (1020, 500))
        FrameDisplay = np.zeros((360, 640, 3), dtype=np.uint8)
        for v in vehicle_results:
            vboxes = v.boxes
            for vbox in vboxes:
                x1, y1, x2, y2 = vbox.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                vcls = int(vbox.cls[0])
                vclass_name = vehicle_class_list[vcls]
                vconf = math.ceil((vbox.conf[0] * 100)) / 100
                obj_list.append([x1, y1, x2, y2])
                if 'hmv' in vclass_name and vconf > 0.75:
                    vlabel = f'{vclass_name}{vconf}'
                    vt_size = cv2.getTextSize(vlabel, 0, fontScale=1,
                                              thickness=2)[0]
                    c2 = x1 + vt_size[0], y1 - vt_size[1] - 3
                    cv2.rectangle(img, (x1, y1), c2, [100, 0, 255], -1, cv2.LINE_AA)
                    cv2.putText(img, vlabel, (x1, y1 - 2), 0, 1, [255, 255, 255],
                                thickness=1, lineType=cv2.LINE_AA)
                    cv2.putText(FrameDisplay, 'HMV detected, WARNING!', (50, 150),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
        bbox_id = tracker.update(obj_list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                down[id] = time.time()
            if id in down:
                if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                    elapsed_time = time.time() - down[id]
                    if counter_down.count(id) == 0:
                        counter_down.append(id)
                        distance = 150
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 3.6
                        cv2.putText(img, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (0, 255, 255), 2)
                    if a_speed_kh > 50:
                        cv2.putText(FrameDisplay, 'OVERSPEED!', (50, 250), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                    (255, 0, 255), 1)
                        start_time = time.time()
                    if 'start_time' in locals() and time.time() - start_time >= 10:
                        del start_time

            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                up[id] = time.time()
            if id in up:
                if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                    elapsed1_time = time.time() - up[id]
                    if counter_up.count(id) == 0:
                        counter_up.append(id)
                        distance1 = 150  # meters
                        a_speed_ms1 = distance1 / elapsed1_time
                        a_speed_kh1 = a_speed_ms1 * 3.6
                        cv2.putText(img, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                    (0, 255, 255), 2)
                        if a_speed_kh1 > 50:
                            cv2.putText(FrameDisplay, 'OVERSPEED!', (50, 250), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                        (255, 0, 255),
                                        1)
                            start_time = time.time()
                        if 'start_time' in locals() and time.time() - start_time >= 10:
                            del start_time

        red_color = (0, 0, 255)
        blue_color = (255, 0, 0)

        cv2.line(img, (140, 298), (795, 298), red_color, 2)
        cv2.line(img, (0, 368), (927, 368), blue_color, 2)

        for a in accident_results:
            aboxes = a.boxes
            for abox in aboxes:
                x1, y1, x2, y2 = abox.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                acls = int(abox.cls[0])
                aclass_name = accident_class_list[acls]
                aconf = math.ceil((abox.conf[0] * 100)) / 100
                if aconf > 0.86:
                    ac_count += 1
                else:
                    ac_count = 0
                if ac_count > 3:
                    alabel = f'{aclass_name}{aconf}'
                    at_size = cv2.getTextSize(alabel, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + at_size[0], y1 - at_size[1] - 3
                    cv2.putText(FrameDisplay, 'Accident detected!', (50, 200), cv2.FONT_HERSHEY_TRIPLEX, 1,
                                (0, 255, 255), 1)
                    frame_filename = f'detected_frames/accident_frame_{count}.jpg'
                    cv2.imwrite(frame_filename, img)
                    print("Accident detected, frame saved:", frame_filename)
                    send_whatsapp_message("+916235900987", frame_filename)
                    print("done")
        FrameDisplay_resized = cv2.resize(FrameDisplay, (810, 700))
        img_resized = cv2.resize(img, (960, 700))
        combined_frame = np.zeros((700, 1900, 3), dtype=np.uint8)
        combined_frame[:img_resized.shape[0], :img_resized.shape[1], :] = img_resized
        combined_frame[:FrameDisplay_resized.shape[0],
        img_resized.shape[1]:img_resized.shape[1] + FrameDisplay_resized.shape[1], :] = FrameDisplay_resized

        yield combined_frame

    cap.release()

