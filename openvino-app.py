import streamlit as st
import numpy as np
from pathlib import Path
from PIL import Image
import openvino as ov
import os

# ─────────────────────────────────────────────
# 기본 설정
# ─────────────────────────────────────────────
INPUT_IMG_SIZE = (224, 224)
CLASSES = ["정상", "불량"]

# 현재 파일 기준 절대 경로
BASE_DIR = Path(__file__).parent

# OpenVINO 모델 및 가중치 경로 설정
MODEL_XML = BASE_DIR / "weights" / "leather_model.xml"
MODEL_BIN = BASE_DIR / "weights" / "leather_model.bin"  # 👈 bin 경로 명시적 추가

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="InspectorsAlly (OpenVINO)",
    page_icon="📷",
    layout="wide"
)

st.title("InspectorsAlly")
st.caption("AI 기반 자동 비전 검사 (OpenVINO 가속 엔진)")
st.write("가죽 표면을 업로드하여 빠르고 효율적인 불량 검사를 수행하세요.")

# ─────────────────────────────────────────────
# 모델 로드 (수정된 핵심 파트)
# ─────────────────────────────────────────────
@st.cache_resource
def load_ov_model():
    # 1. 두 파일이 모두 존재하는지 체크
    if not MODEL_XML.exists():
        st.error(f"❌ 모델 구조 파일(.xml)을 찾을 수 없습니다:\n{MODEL_XML}")
        return None
        
    if not MODEL_BIN.exists():
        st.error(f"❌ 모델 가중치 파일(.bin)을 찾을 수 없습니다:\n{MODEL_BIN}")
        return None

    # 2. 파일의 크기가 비정상적으로 작은지 체크 (LFS 껍데기 포인터 방지용 디버깅)
    xml_size = os.path.getsize(MODEL_XML)
    bin_size = os.path.getsize(MODEL_BIN)
    
    if bin_size < 1000:  # 보통 LFS 포인터 파일은 100~200 바이트 내외입니다.
        st.error(
            f"⚠️ 깃허브 LFS 오류 감지!\n\n"
            f"현재 `.bin` 파일의 크기가 {bin_size} 바이트로 너무 작습니다. "
            f"실제 가중치 데이터가 아닌 '텍스트 껍데기(포인터)'가 올라간 상태입니다.\n"
            f"로컬 터미널에서 LFS 강제 마이그레이션 처리를 다시 해주셔야 합니다."
        )
        return None

    try:
        core = ov.Core()

        # OpenVINO 모델 읽기 (xml과 bin 경로를 둘 다 확실하게 인자로 전달 ⭐)
        model = core.read_model(
            model=str(MODEL_XML),
            weights=str(MODEL_BIN)
        )

        # CPU 타겟 컴파일
        compiled_model = core.compile_model(model, "CPU")
        return compiled_model

    except Exception as e:
        st.error(f"❌ OpenVINO 엔진 로드 최종 실패:\n{e}")
        return None

compiled_model = load_ov_model()

if compiled_model is None:
    st.stop()

# ─────────────────────────────────────────────
# 전처리 함수
# ─────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)

    # VGG16 OpenVINO 전처리
    # RGB → BGR
    arr = arr[:, :, ::-1]

    # ImageNet 평균값 차감
    arr[:, :, 0] -= 103.94
    arr[:, :, 1] -= 116.78
    arr[:, :, 2] -= 123.68

    # 배치 차원 추가
    arr = np.expand_dims(arr, axis=0)
    return arr

# ─────────────────────────────────────────────
# 입력 UI
# ─────────────────────────────────────────────
st.subheader("이미지 입력 방법 선택")

input_method = st.radio(
    "입력 방식:",
    ["파일 업로드", "카메라 촬영"],
    horizontal=True
)

pil_image = None

# 파일 업로드
if input_method == "파일 업로드":
    uploaded_file = st.file_uploader(
        "이미지를 선택하세요",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(
            pil_image,
            caption="업로드된 이미지",
            width=350
        )
        st.success("이미지가 준비되었습니다.")

# 카메라 촬영
elif input_method == "카메라 촬영":
    captured_image = st.camera_input("카메라로 촬영")

    if captured_image is not None:
        pil_image = Image.open(captured_image).convert("RGB")
        st.image(
            pil_image,
            caption="촬영된 이미지",
            width=350
        )
        st.success("이미지가 촬영되었습니다.")

# ─────────────────────────────────────────────
# 검사 실행
# ─────────────────────────────────────────────
if st.button("검사 시작", type="primary"):
    if pil_image is None:
        st.warning("이미지를 먼저 업로드하거나 촬영해주세요.")
    else:
        with st.spinner("OpenVINO 엔진으로 분석 중..."):
            # 전처리
            input_tensor = preprocess(pil_image)

            # 추론
            output = compiled_model([input_tensor])

            # 출력 추출
            prob = float(list(output.values())[0][0][0])

            # 클래스 판별
            label = CLASSES[1 if prob > 0.5 else 0]

        st.subheader("검사 결과")

        # 결과 표시
        if label == "정상":
            st.success("✅ 정상 (합격)")
        else:
            st.error("⚠️ 불량 감지 (불합격)")

        # 확률 표시
        col1, col2 = st.columns(2)
        col1.metric(
            "정상 확률",
            f"{1 - prob:.1%}"
        )
        col2.metric(
            "불량 확률",
            f"{prob:.1%}"
        )

        # 진행바
        st.progress(
            float(prob),
            text=f"불량 위험도: {prob:.1%}"
        )