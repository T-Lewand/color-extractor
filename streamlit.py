import streamlit as st
from PIL import Image
from classes import Picture, Visualization, Pil_Picture

st.sidebar.header('Color extractor')
upload_image = st.sidebar.file_uploader('Drop image here')

if upload_image is not None:
    source_image = Pil_Picture(image=upload_image)
    st.header("Source image")
    st.image(upload_image)

    scale = st.sidebar.slider(label='Image scale', min_value=0.1, max_value=1.0, step=0.1, value=1.0)
    resampled = source_image.resample(scale=scale)

    tol = st.sidebar.slider(label='Color grouping', min_value=0, max_value=100, step=1, value=30)

drop_first = st.sidebar.checkbox('Drop first color')
button = st.sidebar.button('Get colors')

if button:
    colors = resampled.get_hex(tol=tol, drop_biggest=drop_first)
    chart = Visualization(colors)
    treemap = chart.treemap()
    color_list = chart.color_list()
    st.pyplot(fig=treemap)
    st.pyplot(fig=color_list)
