import streamlit as st
import sys
import gc
from detect_weapon import func_detect_weapon
from streamlit.legacy_caching import clear_cache

def main():
    gc.enable()
    st.header("Computer Vision Demo")

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
    func_detect_weapon()
    
main()