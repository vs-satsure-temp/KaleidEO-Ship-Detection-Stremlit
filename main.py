import streamlit as st
from ultralytics import YOLO

import torch
import io

from utils import preprocess_image, run_inference, post_process

def main():

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
        st.session_state.annotated_image = None
    

    st.title("KaleidEO: Ship Detection")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('/app/yolov8.pt').to(device)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

        image = preprocess_image(uploaded_file)
        st.subheader("Uploaded Image")
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predict_button = st.button('Predict')

        if predict_button:

            with st.spinner('Detecting ships...'):
                predictions = run_inference(model, image, device)

            annotated_image = post_process(model, image, predictions)
            st.session_state.annotated_image = annotated_image

            st.subheader("Detected Ships")
            st.image(annotated_image, caption='Annotated Image', use_column_width=True)

            if st.session_state.annotated_image is not None:

                img_buffer = io.BytesIO()
                st.session_state.annotated_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)

                columns = st.columns(3, vertical_alignment='center')

                with columns[0]:
                    st.download_button(
                        label="Download Output",
                        data=img_buffer,
                        file_name="annotated_image.png",
                        mime="image/png"
                    )

                with columns[2]:
                    if st.button("Clear"):
                        st.session_state.clear()
                        st.experimental_rerun()


if __name__ == "__main__":
    main()
