import os
import random
import tempfile
import traceback
from urllib.request import urlopen
import shutil

import cv2 as cv
import numpy as np
import streamlit as st

from modules import *
from strings import *


import gc  # garbage collection

gc.enable()


# [start] [defaults] ________________________________________________
# local
dataPath = r"mediapipe/data"
demoImages = ["reshot01.jpg", "reshot02.jpg", "reshot03.jpg", "reshot04.jpg"]

# online
urlImages = [
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1623132298/photosp/fafc9d9c-f2dc-46f5-9fa4-885914b176b0/fafc9d9c-f2dc-46f5-9fa4-885914b176b0.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838655/photosp/0a0f136f-9032-4480-a1ca-1185dd161368/0a0f136f-9032-4480-a1ca-1185dd161368.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838912/photosp/57ece171-00ea-439c-95e2-01523fd41285/57ece171-00ea-439c-95e2-01523fd41285.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838985/photosp/fc11c636-e5db-4b02-b3d2-99650128c351/fc11c636-e5db-4b02-b3d2-99650128c351.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838983/photosp/22f4d64b-6358-47b5-8a96-a5b9b8019829/22f4d64b-6358-47b5-8a96-a5b9b8019829.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1613176985/photosp/373e53b2-49a8-4e5f-85d2-bfc1a233572e/373e53b2-49a8-4e5f-85d2-bfc1a233572e.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1588713749/photosp/148fb738-110b-45d5-96e5-89bd36335b91/148fb738-110b-45d5-96e5-89bd36335b91.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1541429083/photosp/76400249-2c7c-4030-b381-95c1c4106db6/76400249-2c7c-4030-b381-95c1c4106db6.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1525791591/photosp/ee9ec8f6-3f11-48a6-a112-57825f983b3a/ee9ec8f6-3f11-48a6-a112-57825f983b3a.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838947/photosp/705bc78c-6e43-47c1-8295-802b48106695/705bc78c-6e43-47c1-8295-802b48106695.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1588713965/photosp/4c7d6a68-a215-47bd-a1a6-7fb137cdf6c4/4c7d6a68-a215-47bd-a1a6-7fb137cdf6c4.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1611254271/photosp/c67f2181-50a6-4b4d-9099-d401197a99a2/c67f2181-50a6-4b4d-9099-d401197a99a2.jpg",
    "https://res.cloudinary.com/twenty20/private_images/t_standard-fit/v1521838971/photosp/a62f2758-ca17-424f-b406-d646162202d7/a62f2758-ca17-424f-b406-d646162202d7.jpg",
]

# global parameters
target_h, target_w = 350, 550
# webapp
appPages = ["Home Page", "Mediapipe Modules", "About Me"]
appModules = ["Hand Tracking", "Pose Estimation", "Face Detection", "Face Mesh"]



def open_img_path_url(url_or_file, source_type, source_path=None, resize=False):
    img, mask = [], []
    if source_type == "path":
        if source_path is None:
            source_path = dataPath
        img = cv.imread(os.path.join(source_path, url_or_file))
    elif source_type == "url":
        resp = urlopen(url_or_file)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv.imdecode(img, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if not resize:
        return img
    else:
        img_h = img.shape[0]
        ratio = target_h / img_h
        r_img = cv.resize(img, None, fx=ratio, fy=ratio)
        try:
            r_img_w = r_img.shape[1]
            left_edge = target_w // 2 - r_img_w // 2
            mask = np.zeros((target_h, target_w, 3), dtype="uint8")
            mask[:, left_edge : left_edge + r_img_w] = r_img
            return img, mask
        except Exception:
            return img, r_img




def init_module(media, type, detector, placeholders):
    cols = placeholders[0].columns([2, 2, 1, 1])
    if type == "image":
        img = detector.findFeatures(media)
        placeholders[1].columns([2, 10, 2])[1].image(img, use_column_width=True)
        del img

def run_selected_module(_fs, media, type, ph_variables):
    moreInfo1 = st.empty()
    moreInfo2 = st.empty()
    moduleOutput1 = st.empty()
    moduleOutput2 = st.empty()

    new_value = ph_variables[0].slider(
        "Solution Confidence [0.4-1.0]",
        min_value=0.4,
        max_value=1.0,
        value=_fs.sol_confidence,
    )
    if new_value != _fs.sol_confidence:
        _fs.sol_confidence = new_value
        st.experimental_rerun()

    module_selection = appModules[_fs.idx_current_module]
    if module_selection == "Hand Tracking":
        moreInfo1.markdown(
            "*Click below for information on the Hands Detector solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Number Of Hands [1-6]", min_value=1, max_value=6, value=_fs.num_hands
        )
        if new_value != _fs.num_hands:
            _fs.num_hands = new_value
            st.experimental_rerun()

        with moreInfo2.expander(""):
            st.markdown(aboutMpHands(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = handDetector(
            numHands=_fs.num_hands, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Pose Estimation":
        moreInfo1.markdown(
            "*Click below for information on the Pose Detector solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Smooth Landmarks [0/1]", min_value=0, max_value=1, value=_fs.smooth_lms
        )
        if new_value != _fs.smooth_lms:
            _fs.smooth_lms = new_value
            st.experimental_rerun()

        with moreInfo2.expander(""):
            st.markdown(aboutMpPose(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = poseDetector(
            smoothLandmarks=bool(_fs.smooth_lms), solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Face Detection":
        moreInfo1.markdown(
            "*Click below for information on the Face Detection solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Model Selection [0/1]", min_value=0, max_value=1, value=_fs.face_model
        )
        if new_value != _fs.face_model:
            _fs.face_model = new_value
            st.experimental_rerun()

        with moreInfo2.expander(""):
            st.markdown(aboutMpFaceDetection(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = faceDetector(
            modelSelection=_fs.face_model, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    elif module_selection == "Face Mesh":
        moreInfo1.markdown(
            "*Click below for information on the Face Mesh solution...*"
        )
        new_value = ph_variables[1].number_input(
            "Number Of Faces [1-5]", min_value=1, max_value=5, value=_fs.num_faces
        )
        if new_value != _fs.num_faces:
            _fs.num_faces = new_value
            st.experimental_rerun()

        with moreInfo2.expander(""):
            st.markdown(aboutMpFaceMesh(), unsafe_allow_html=True)
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

        detector = faceMeshDetector(
            numFaces=_fs.num_faces, solutionConfidence=_fs.sol_confidence
        )
        init_module(media, type, detector, (moduleOutput1, moduleOutput2))
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]

    # garbage collection
    del (
        moreInfo1,
        moreInfo2,
        moduleOutput1,
        moduleOutput2,
        detector,
    )


def read_source_media(_fs, appSources, ph_variables):
    if _fs.current_image_path == "":
        _fs.current_image_path = demoImages[0]
        _fs.current_image_url = urlImages[0]

    data_source_selection = appSources[_fs.idx_data_source]
    if data_source_selection == "User Image":
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        img_file_buffer = st.sidebar.file_uploader(
            "Upload Image", type=["jpg", "jpeg", "png"], key=f"img_ul_{_fs.uploader_key}"
        )

        if img_file_buffer:
            temp_file.write(img_file_buffer.read())
            _fs.current_image_upload = temp_file.name

        if _fs.current_image_upload != "":
            img, mask = open_img_path_url(
                _fs.current_image_upload, "path", source_path="", resize=True
            )

            st.sidebar.markdown("")
            cols = st.sidebar.columns([3, 2])
            cols[0].text("Original Image")
            st.sidebar.image(mask, use_column_width=True)
            if cols[1].button("Clear Upload"):
                _fs.current_image_upload = ""
                _fs.uploader_key += 1
                st.experimental_rerun()
            del temp_file, img_file_buffer, mask, cols  # garbage collection
            run_selected_module(_fs, img, "image", ph_variables)
        else:
            if appModules[_fs.idx_current_module] == "Hand Tracking":
                _fs.current_image_path = "reshot01.jpg"
            elif appModules[_fs.idx_current_module] == "Face Detection":
                _fs.current_image_path = "reshot02.jpg"
            elif appModules[_fs.idx_current_module] == "Pose Estimation":
                _fs.current_image_path = "reshot04.jpg"
            else:
                _fs.current_image_path = "reshot03.jpg"
            img = open_img_path_url(_fs.current_image_path, "path")
            st.sidebar.markdown("")
            cols = st.sidebar.columns([3, 2])
            cols[0].text("Original Image")
            st.sidebar.image(img, use_column_width=True)
            if cols[1].button("Clear Upload"):
                _fs.current_image_path = ""
                st.experimental_rerun()
            del cols 
            run_selected_module(_fs, img, "image", ph_variables)
    return _fs
