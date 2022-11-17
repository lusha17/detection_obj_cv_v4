from typing import List
from streamlit_webrtc import ClientSettings
from typing import List, NamedTuple, Optional
import cv2
import base64
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pytz
import av
import datetime
import matplotlib.colors as mcolors
from PIL import Image
import time
import threading
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode
#from streamlit.legacy_caching import clear_cache

lock = threading.Lock()


def func_detect_weapon():
    CLASSES_CUSTOM = [ 'bill', 'card', 'face', 'knife', 'mask', 'firearm', 'purse', 'smartphone']
    CLASSES_BASE= [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

    WEBRTC_CLIENT_SETTINGS = ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.6

    result_queue = [] 

    #st.set_page_config(page_title="Weapon Detection Demo", layout="wide", initial_sidebar_state="collapsed")

    #st.sidebar.markdown("""<center data-parsed=""><img src="http://drive.google.com/uc?export=view&id=1Mad62XWdziqcx9wijUODpzGzqYEGhafC" align="center"></center>""",unsafe_allow_html=True,)
    #st.sidebar.markdown(" ")


    #@st.cache(max_entries=2)
    def get_yolo5(label):
        if label=='Base':
            return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5m.pt')  
        else:
            return torch.hub.load('ultralytics/yolov5', 'custom', path='all_m.pt')

    def get_preds(img):
        return model([img]).xyxy[0].numpy()

    def get_colors(indexes):
        to_255 = lambda c: int(c*255)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) for name_color in tab_colors]
        base_colors = list(mcolors.BASE_COLORS.values())
        base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
        rgb_colors = tab_colors + base_colors
        rgb_colors = rgb_colors*5
        color_dict = {}
        for i, index in enumerate(indexes):
            if i < len(rgb_colors):
                color_dict[index] = rgb_colors[i]
            else:
                color_dict[index] = (255,0,0)
        return color_dict


    def transform(frame):
        img = frame.to_ndarray(format="bgr24")
        img_ch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img_ch)
        result = result[np.isin(result[:,-1], target_class_ids)]  
        for bbox_data in result:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            if conf > confidence_threshold:
                p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
                img = cv2.rectangle(img, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                class_ = CLASSES[label]
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[label], 2,)
                if ((class_ == 'firearm') | (class_ == 'knife')) & agree:
                    time_detect = datetime.datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None).strftime("%m-%d-%y %H:%M:%S")
                    retval, buffer_img= cv2.imencode('.jpg', img)
                    data = base64.b64encode(buffer_img).decode("utf-8")
                    html = "<img src='data:image/jpg;base64," + data + f"""' style='display:block;margin-left:auto;margin-right:auto;width:200px;border:0;'>"""
                    with lock:
                        result_queue.insert(0, {'object': class_, 'time_detect': time_detect, 'confident': str(conf.round(2)), 'img': html})
        return av.VideoFrame.from_ndarray(img, format="bgr24")


    model_type = st.sidebar.selectbox('Select model type', ('Base', 'Custom'), index=1)

    with st.spinner('Loading the model...'):
        if model_type == 'Base':
            cache_key = 'base'
            if cache_key in st.session_state:
                model = st.session_state[cache_key]
            else:
                model = get_yolo5(model_type)
                st.session_state[cache_key] = model
        else:
            cache_key = 'custom'
            if cache_key in st.session_state:
                model = st.session_state[cache_key]
            else:
                model = get_yolo5(model_type)
                st.session_state[cache_key] = model

    #st.success('Loading the model.. Done!')

    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera'), index=1)

    if model_type == 'Base':
        CLASSES = CLASSES_BASE
        classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='person')
    else:
        CLASSES = CLASSES_CUSTOM
        classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='firearm')

    all_labels_chbox = st.sidebar.checkbox('All classes', value=True)
    if all_labels_chbox:
        target_class_ids = list(range(len(CLASSES)))
    elif classes_selector:
        target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
    else:
        target_class_ids = [0]
    rgb_colors = get_colors(target_class_ids)
    detected_ids = None
    if prediction_mode == 'Single image':
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = get_preds(img)
            result_copy = result.copy()
            result_copy = result_copy[np.isin(result_copy[:,-1], target_class_ids)]
            detected_ids = []
            img_draw = img.copy().astype(np.uint8)
            for bbox_data in result_copy:
                xmin, ymin, xmax, ymax, conf, label = bbox_data
                if conf > confidence_threshold:
                    p0, p1, label = (int(xmin), int(ymin)), (int(xmax), int(ymax)), int(label)
                    img_draw = cv2.rectangle(img_draw, p0, p1, rgb_colors[label], 2) 
                    ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                    xtext = xmin + 10
                    class_ = CLASSES[label]
                    text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                    img_draw = cv2.putText(img_draw, text_for_vis, (int(xtext), int(ytext)),cv2.FONT_HERSHEY_SIMPLEX,0.5,rgb_colors[label],2,)
                    detected_ids.append(label)
            st.image(img_draw, use_column_width=True)
    elif prediction_mode == 'Web camera':
        ctx = webrtc_streamer(key="example", video_frame_callback=transform,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False}, mode=WebRtcMode.SENDRECV, async_processing=True)
        agree = st.checkbox("Enable weapon logging", value=False)
        if agree:
            if ctx.state.playing:
                labels_placeholder = st.empty()
                while True:
                    time.sleep(0.5)
                    with lock:
                        result_queue = result_queue[:5]
                        df = pd.DataFrame(result_queue)
                        labels_placeholder.write(df.to_html(escape=False), unsafe_allow_html=True)