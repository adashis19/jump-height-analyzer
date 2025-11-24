import streamlit as st
import tempfile
from pathlib import Path
from jump_analysis import analyze_jump_video

st.set_page_config(page_title="Jump Height Analyzer", layout="centered")

st.title("🏃‍♂️ Jump Height Analyzer")

st.write("Upload a jump video and I'll detect take-off, landing, flight time, and jump height.")

uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.video(tmp_path)

    with st.spinner("Analyzing jump..."):
        res = analyze_jump_video(tmp_path, save_annotated=True)

    st.success("Analysis complete!")

    st.subheader("Results")

    st.write(f"**FPS:** {res['fps']:.2f}")
    st.write(f"**Total frames:** {res['total_frames']}")

    st.write(f"**Take-off frame:** {res['takeoff_frame']}")
    st.write(f"**Landing frame:** {res['landing_frame']}")
    st.write(f"**Apex frame (min foot_y):** {res['frame_min']}")

    st.write(f"**Flight time:** {res['flight_time_s']:.4f} s")
    st.write(f"**Jump height:** {res['jump_height_m']:.3f} m ({res['jump_height_cm']:.1f} cm)")

    st.subheader("Annotated video")

    annotated_path = res["annotated_video_path"]
    if annotated_path is not None:
        st.video(annotated_path)
        st.caption("Annotated video with pose, foot_y, ankle angle, TAKEOFF and LANDING markers.")
    else:
        st.write("Annotated video was not generated.")
