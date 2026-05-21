import streamlit as st
import numpy as np
import os
from PIL import Image
import openvino as ov

# ── 상수 설정 ──────────────────────────────────
INPUT_IMG_SIZE = (224, 224)
CLASSES = ["정상", "불량"]
MODEL_XML = "./weights/leather_model.xml"  # .bin 파일도 같은 폴더에 있어야 함

# ─────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="InspectorsAlly (OpenVINO)", page_icon="📷", layout="wide")
st.title("InspectorsAlly")
st.caption("AI 기반 자동 비전 검사 (OpenVINO 가속 엔진)")
st.write("가죽 표면을 업로드하여 빠르고 효율적인 불량 검사를 수행하세요.")

# ─────────────────────────────────────────────
# 2. 모델 로드 (@st.cache_resource)
# ─────────────────────────────────────────────
@st.cache_resource
def load_ov_model():
    if not os.path.exists(MODEL_XML):
        return None
    try:
        core = ov.Core()
        model = core.read_model(MODEL_XML)
        compiled_model = core.compile_model(model, "CPU")
        return compiled_model
    except Exception as e:
        st.error(f"OpenVINO 모델 로드 실패: {e}")
        return None

compiled_model = load_ov_model()

if compiled_model is None:
    st.error(f"모델 파일을 찾을 수 없습니다: `{MODEL_XML}`\n\n파일 경로를 확인해주세요.")
    st.stop()

# ─────────────────────────────────────────────
# 3. OpenVINO 전용 전처리 로직
# ─────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)

    # VGG16 학습 시 사용된 전처리 규칙 (RGB -> BGR, ImageNet 평균 차감)
    arr = arr[:, :, ::-1]          # RGB → BGR
    arr[:, :, 0] -= 103.94         # B 채널 평균
    arr[:, :, 1] -= 116.78         # G 채널 평균
    arr[:, :, 2] -= 123.68         # R 채널 평균

    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

# ─────────────────────────────────────────────
# 4. 입력 UI
# ─────────────────────────────────────────────
st.subheader("이미지 입력 방법 선택")
input_method = st.radio("입력 방식:", ["파일 업로드", "카메라 촬영"], horizontal=True)

pil_image = None

if input_method == "파일 업로드":
    file = st.file_uploader("이미지를 선택하세요", type=["jpg", "jpeg", "png"])
    if file:
        pil_image = Image.open(file).convert("RGB")
        st.image(pil_image, caption="업로드된 이미지", width=300)
        st.success("이미지가 준비되었습니다.")

elif input_method == "카메라 촬영":
    shot = st.camera_input("카메라로 촬영")
    if shot:
        pil_image = Image.open(shot).convert("RGB")
        st.success("이미지가 촬영되었습니다.")

# ─────────────────────────────────────────────
# 5. 검사 실행 및 결과 표시
# ─────────────────────────────────────────────
if st.button("검사 시작", type="primary"):
    if pil_image is None:
        st.warning("이미지를 먼저 업로드하거나 촬영해주세요.")
    else:
        with st.spinner("OpenVINO 엔진으로 분석 중..."):
            # 전처리 및 추론
            arr = preprocess(pil_image)
            prob = float(compiled_model(arr)[0][0][0])
            label = CLASSES[1 if prob > 0.5 else 0]

        st.subheader("검사 결과")
        
        # 결과 메시지 출력
        if label == "정상":
            st.success(f"✅ **정상 (합격)**")
        else:
            st.error(f"⚠️ **불량 감지 (불합격)**")
        
        # metric 및 progress 시각화
        col1, col2 = st.columns(2)
        col1.metric("정상 확률", f"{1 - prob:.1%}")
        col2.metric("불량 확률", f"{prob:.1%}")
        st.progress(float(prob), text=f"불량 위험도: {prob:.1%}")