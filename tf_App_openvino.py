"""
tf_App_openvino.py
──────────────────
OpenVINO 모델(.xml / .bin)로 이진분류 추론하는 기본 구조 예제.

[전체 흐름]
  1. 페이지 설정
  2. 모델 로드        : OpenVINO Core → compile_model
  3. 이미지 전처리    : PIL → numpy → VGG16 정규화
  4. 추론             : compiled_model(img_array)
  5. 결과 표시        : 정상 / 불량 + 확률
"""

import streamlit as st
import numpy as np
import os
from PIL import Image
import openvino as ov

# ── 상수 ──────────────────────────────────────
INPUT_IMG_SIZE = (224, 224)
CLASSES        = ["정상", "불량"]
MODEL_XML      = "./weights/leather_model.xml"   # .bin 은 자동 탐색


# ─────────────────────────────────────────────
# 1. 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:", layout="centered")
st.title("InspectorsAlly")
st.caption("VGG16 + OpenVINO 기반 가죽 제품 불량 검사")


# ─────────────────────────────────────────────
# 2. 모델 로드
#    ov.Core()       : OpenVINO 런타임 초기화
#    read_model()    : .xml 구조 + .bin 가중치 읽기
#    compile_model() : CPU 추론 엔진으로 컴파일
#
#    @st.cache_resource : 앱 재실행 시 모델을 다시 로드하지 않음
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_XML):
        return None
    core  = ov.Core()
    model = core.read_model(MODEL_XML)
    return core.compile_model(model, "CPU")


# ─────────────────────────────────────────────
# 3. 이미지 전처리
#    VGG16 훈련 시 사용한 전처리와 동일하게 맞춰야 예측이 정확함
#
#    ① RGB → BGR 채널 순서 변환  (VGG16은 BGR 입력)
#    ② ImageNet BGR 평균값 빼기
#       B: 103.94 / G: 116.78 / R: 123.68
#    ③ 배치 차원 추가 (224,224,3) → (1,224,224,3)
# ─────────────────────────────────────────────
def preprocess(pil_img):
    img   = pil_img.convert("RGB").resize(INPUT_IMG_SIZE)
    arr   = np.array(img, dtype=np.float32)

    arr   = arr[:, :, ::-1]        # RGB → BGR
    arr[:, :, 0] -= 103.94         # B 채널 평균 빼기
    arr[:, :, 1] -= 116.78         # G 채널 평균 빼기
    arr[:, :, 2] -= 123.68         # R 채널 평균 빼기

    return np.expand_dims(arr, axis=0)   # (1, 224, 224, 3)


# ─────────────────────────────────────────────
# 4. 추론
#    compiled_model(input) → OVDict
#    결과는 인덱스[0] 으로 접근: shape (1, 1)
#    sigmoid 출력값 → 0에 가까울수록 정상, 1에 가까울수록 불량
# ─────────────────────────────────────────────
def predict(compiled_model, pil_img):
    arr  = preprocess(pil_img)
    prob = float(compiled_model(arr)[0][0][0])   # sigmoid 단일값
    label = CLASSES[1 if prob > 0.5 else 0]
    return label, prob


# ─────────────────────────────────────────────
# 5. 메인 UI
# ─────────────────────────────────────────────
compiled_model = load_model()

if compiled_model is None:
    st.error(f"모델 파일을 찾을 수 없습니다: `{MODEL_XML}`")
    st.stop()

# ── 입력 방법 선택 ──
st.subheader("이미지 입력")
input_method = st.radio("options", ["파일 업로드", "카메라 촬영"],
                        label_visibility="collapsed")
pil_image = None

if input_method == "파일 업로드":
    file = st.file_uploader("이미지를 선택하세요", type=["jpg", "jpeg", "png"])
    if file:
        pil_image = Image.open(file).convert("RGB")
        st.image(pil_image, caption="업로드된 이미지", width=300)

elif input_method == "카메라 촬영":
    shot = st.camera_input("카메라로 촬영")
    if shot:
        pil_image = Image.open(shot).convert("RGB")
        st.image(pil_image, caption="촬영된 이미지", width=300)

# ── 검사 버튼 ──
if st.button("검사 시작", type="primary"):
    if pil_image is None:
        st.warning("이미지를 먼저 입력해주세요.")
    else:
        with st.spinner("분석 중..."):
            label, prob = predict(compiled_model, pil_image)

        # ── 결과 표시 ──
        st.subheader("검사 결과")

        if label == "정상":
            st.success(f"✅ **정상**  (불량 확률: {prob:.1%})")
        else:
            st.error(f"⚠️ **불량 감지**  (불량 확률: {prob:.1%})")

        col1, col2 = st.columns(2)
        col1.metric("정상 확률", f"{1 - prob:.1%}")
        col2.metric("불량 확률", f"{prob:.1%}")
        st.progress(float(prob), text=f"불량 확률: {prob:.1%}")
