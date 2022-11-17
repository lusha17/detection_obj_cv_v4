import streamlit as st
import sys
import gc
sys.path.append('mediapipe')
from detect_weapon import func_detect_weapon
from streamlitMediapipe import mediapipe_f
from detect_fire_smoke import func_detect_fire_smoke
from streamlit.legacy_caching import clear_cache

def main():
    gc.enable()
    st.header("Computer Vision Demo")

    pages_1 = {
        "Real time hand items detection": func_detect_weapon,
        "Personality and action recognition": mediapipe_f,
        "Real time fire and smoke detection": func_detect_fire_smoke
    }
    page_titles = pages_1.keys()

    st.sidebar.markdown("""<center data-parsed=""><img src="http://drive.google.com/uc?export=view&id=1Mad62XWdziqcx9wijUODpzGzqYEGhafC" align="center"></center>""",unsafe_allow_html=True,)
    st.sidebar.markdown(" ")
    
    def reload():
        clear_cache()
        gc.collect()
        st.experimental_rerun()

    pages = st.sidebar.columns([1, 1, 1])
    
    pages[0].markdown(" ")

    if pages[1].button("Reload App"):
        reload()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)
    page_func = pages_1[page_title]
    page_func()
    
main()