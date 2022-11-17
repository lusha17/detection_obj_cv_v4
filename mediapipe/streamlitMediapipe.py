import gc  # garbage collection

import streamlit as st
from streamlit.server.server import Server
#from streamlit import caching
from streamlit.legacy_caching import clear_cache
from functions import *
from appSessionState import getSessionState


def mediapipe_f():
    #gc.enable()  # garbage collection

    webapp = getSessionState(
        idx_current_page=0,
        idx_current_module=0,
        idx_data_source=0,
        current_image_path="",
        current_image_url="",
        idx_url_image=0,
        current_video_path="",
        current_video_url="",
        idx_url_video=0,
        sol_confidence=0.65,
        num_hands=2,
        smooth_lms=1,
        face_model=0,
        num_faces=2,
        current_image_upload="",
        current_video_upload="",
        uploader_key=0,
        webcam_device_id=0,
    )



    appPages = ["Mediapipe Modules"]
    appModules = ["Hand Tracking", "Pose Estimation", "Face Detection", "Face Mesh"]
    appSources = ["User Image"]
    # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-[end]


    #st.set_page_config(page_title="Streamlit Mediapipe WebApp", layout="wide")
    st.set_option("deprecation.showfileUploaderEncoding", False)
    #st.sidebar.markdown("""<center data-parsed=""><img src="http://drive.google.com/uc?export=view&id=1Mad62XWdziqcx9wijUODpzGzqYEGhafC" align="center"></center>""",unsafe_allow_html=True)
    #st.sidebar.markdown(" ")


    # [start] [setup app pages, modules & data sources]__________________


    if webapp.idx_current_page == appPages.index("Mediapipe Modules"):
        st.sidebar.write("")
        mp_selectors = st.sidebar.columns([1, 1])
        module_selection = mp_selectors[0].selectbox("Modules:", appModules,
            index=webapp.idx_current_module,
        )
        if module_selection != appModules[webapp.idx_current_module]:
            webapp.idx_current_module = appModules.index(module_selection)
            st.experimental_rerun()
        data_source_selection = mp_selectors[1].selectbox(
            "Data/Media Source:",
            appSources,
            index=webapp.idx_data_source,
        )
        if data_source_selection != appSources[webapp.idx_data_source]:
            webapp.idx_data_source = appSources.index(data_source_selection)
            st.experimental_rerun()
        st.sidebar.write("")
        ph_variables = st.sidebar.columns([1, 1])
        read_source_media(webapp, appSources, ph_variables)