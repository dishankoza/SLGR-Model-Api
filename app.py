from flask import Flask, render_template, Response
import cv2
import sys
import os
import numpy as np
import tensorflow as tf
import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

label_lines = [line.rstrip() for line
                   in tf.io.gfile.GFile("logs/trained_labels.txt")]
app = Flask(__name__)

def vconcat_resize(img_list, interpolation 
                   = cv2.INTER_CUBIC):
      # take minimum width
    w_min = min(img.shape[1] 
                for img in img_list)
      
    # resizing images
    im_list_resize = [cv2.resize(img,
                      (w_min, int(img.shape[0] * w_min / img.shape[1])),
                                 interpolation = interpolation)
                      for img in img_list]
    # return final image
    return cv2.vconcat(im_list_resize)



camera = cv2.VideoCapture(0)  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)

def gen_processed_frames():  # generate frame by frame from camera

    with tf.io.gfile.GFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        res, score = '', 0.0
        c =0
        i = 0
        mem = ''
        consecutive = 2
        sequence = ''
        while True:
            # Capture frame-by-frame
            success, frame = camera.read()  # read the camera frame
            frame = cv2.flip(frame, 1)

            if not success:
                break
            else:
                x1, y1, x2, y2 = 100, 100, 300, 300
                img_cropped = frame[y1:y2, x1:x2]
                c += 1
                image_data = cv2.imencode('.jpg', img_cropped)[1].tobytes()
                
                if i == 4:
                    predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})

            # Sort to show labels of first prediction in order of confidence
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

                    max_score = 0.0
                    predict_res = ''
                    for node_id in top_k:
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        if score > max_score:
                            max_score = score
                            predict_res = human_string

                    res_tmp, score = predict_res,max_score
                    res = res_tmp
                    i = 0
                    if res not in ['nothing']:
                        if res == 'space':
                            sequence += ' '
                        elif res == 'del':
                            sequence = sequence[:-1]
                        else:
                            sequence += res
                i +=1                
                cv2.putText(frame, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
                cv2.putText(frame, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                img_sequence = np.zeros((200,1200,3), np.uint8)
                cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                img_v_resize = vconcat_resize([frame, img_sequence])
                ret,buffer = cv2.imencode('.jpg', img_v_resize)
                frame = buffer.tobytes()

                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def gen_normal_frames():  # generate frame by frame from camera
    while True:
        time.sleep(0.2)
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        frame = cv2.flip(frame, 1)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/processed_video_feed')
def processed_video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_processed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/normal_video_feed')
def normal_video_feed():
    return Response(gen_normal_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=False)