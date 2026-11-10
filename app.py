import io
import itertools
import os
import re
import threading
from datetime import date

import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps

import reading

st.set_page_config(page_title="AI 관상 보기", page_icon="🔮")

# 로컬 개발용: 이 파일 옆의 .env 를 읽는다. (배포 환경에는 .env 가 없고 Streamlit Secrets 를 쓴다)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

MAX_UPLOAD_MB = 5
ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
DAILY_LIMIT = 5  # 세션당 하루 분석 횟수 (F11)
GLOBAL_DAILY_LIMIT_DEFAULT = 300  # 서버 전체 하루 분석 횟수. 새로고침으로 세션 제한을 우회해도 비용이 여기서 멈춘다. GLOBAL_DAILY_LIMIT 로 조절
MAX_PIXELS = 36_000_000  # 이보다 큰 해상도는 디코딩 전에 거부한다. (작은 파일이 메모리를 폭발시키는 것을 막음)
MAX_PREVIEW_SIDE = 1024
ZWSP = chr(0x200B)  # 투명 공백: URL 자동 링크 방지용


# ---------- 설정 ----------
def get_setting(name):
    """st.secrets → 환경변수(.env 포함) 순서로 값을 찾는다. 값은 어디에도 출력하지 않는다."""
    try:
        value = st.secrets.get(name)
    except Exception:  # secrets.toml 이 없으면 예외가 나므로 무시하고 환경변수로 넘어간다
        value = None
    value = value or os.environ.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def get_api_key():
    return get_setting("OPENAI_API_KEY")


# ---------- 업로드 검증 ----------
def read_image(uploaded):
    """업로드 파일을 검증하고 정규화(RGB, 긴 변 1024px 이하, EXIF 회전 반영)한 PIL 이미지로 돌려준다.
    실패하면 (None, 안내 문구)."""
    if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"파일이 너무 큽니다. {MAX_UPLOAD_MB}MB 이하의 사진을 올려주세요."
    data = uploaded.getvalue()
    try:
        probe = Image.open(io.BytesIO(data))  # 헤더만 읽는다
        if probe.size[0] * probe.size[1] > MAX_PIXELS:
            return None, f"사진 해상도가 너무 큽니다. 약 {MAX_PIXELS // 1_000_000}백만 화소 이하의 사진을 올려주세요."
        probe.verify()  # 깨진 파일 검사 (verify 후에는 다시 열어야 함)
        img = reading.normalize_image(Image.open(io.BytesIO(data)), MAX_PREVIEW_SIDE)
    except Exception:  # 열 수 없는 파일, 손상된 파일, 지나치게 큰 해상도 등
        return None, "이미지를 열 수 없습니다. 손상되지 않은 jpg, png, webp 사진을 올려주세요."
    return img, None


# ---------- 사용량 제한 (F11) ----------
def remaining_uses():
    today = date.today().isoformat()
    if st.session_state.get("usage_date") != today:  # 날짜가 바뀌면 초기화
        st.session_state.usage_date = today
        st.session_state.usage_count = 0
    return max(0, DAILY_LIMIT - st.session_state.get("usage_count", 0))


@st.cache_resource
def global_usage():
    """서버 프로세스 전체가 공유하는 하루 사용량. (재시작하면 초기화된다)"""
    return {"date": "", "count": 0, "lock": threading.Lock()}


def global_limit():
    try:
        return max(0, int(get_setting("GLOBAL_DAILY_LIMIT") or GLOBAL_DAILY_LIMIT_DEFAULT))
    except ValueError:
        return GLOBAL_DAILY_LIMIT_DEFAULT


def reserve_global_use():
    """서버 전체 일일 상한 안에서 1회를 예약한다. 상한이면 False."""
    g, today = global_usage(), date.today().isoformat()
    with g["lock"]:
        if g["date"] != today:
            g["date"], g["count"] = today, 0
        if g["count"] >= global_limit():
            return False
        g["count"] += 1
        return True


# ---------- 모델 응답을 안전하게 화면에 표시 ----------
def plain(text):
    """모델이 만든 글이 마크다운/링크/이모지 코드로 해석되지 않게 이스케이프한다. (HTML 은 렌더링하지 않는다)"""
    text = re.sub(
        r"(?i)(https?|ftp)://|www\.",
        lambda m: m.group(0)[:2] + ZWSP + m.group(0)[2:],
        text,
    )
    return re.sub(r"([\\`*_{}\[\]()#+\-.!~<>|$:&])", r"\\\1", text)


_card_ids = itertools.count()  # app.py 는 매 실행마다 처음부터 실행되므로 실행마다 0부터 시작한다


def card(title, body):
    """제목 + 본문이 들어가는 테두리 카드. 본문은 모델이 만든 글이므로 plain() 으로만 표시한다.
    key 를 줘서 st-key-card-N 클래스로 스타일링한다. (내부 emotion 클래스명은 버전마다 바뀌어 쓰지 않는다)"""
    with st.container(border=True, key=f"card-{next(_card_ids)}"):
        st.markdown(f"**{title}**")
        st.markdown(plain(body))


def show_result(res):
    if "raw" in res:  # JSON 형식이 깨진 응답: 정리하지 않고 원문을 그대로 보여준다
        st.warning("결과 형식이 올바르지 않아 원문을 그대로 보여드려요.")
        with st.container(border=True, key=f"card-{next(_card_ids)}"):
            st.markdown(plain(res["raw"]) or "(빈 응답)")
    elif res["no_face"]:
        st.warning("사진에서 얼굴을 찾지 못했어요. 얼굴이 잘 보이는 정면 사진을 올려주세요.")
        return
    else:
        if res["summary"]:  # 한 줄 총평은 강조 카드
            with st.container(border=True, key="summary-card"):
                st.caption("한 줄 총평")
                st.markdown(f"### {plain(res['summary'])}")
        if res["overall"]:
            card("🧿 전체 인상", res["overall"])
        if res["features"]:  # 부위별 풀이: 2열 그리드 (좁은 화면에서는 한 줄씩)
            st.markdown("##### 부위별 풀이")
            # 두 개씩 한 줄로 묶는다. (열 단위로 만들면 모바일에서 왼쪽 열 전체 → 오른쪽 열 전체 순으로 쌓여 순서가 뒤섞인다)
            feats = res["features"]
            for i in range(0, len(feats), 2):
                for col, f in zip(st.columns(2), feats[i:i + 2]):
                    with col:
                        card(f["part"], f["reading"])
        if res["personality"]:
            card("🌿 성격과 기질", res["personality"])
        fortune = res["fortune"]
        items = [(label, fortune[k]) for k, label in (("wealth", "💰 재물운"), ("love", "💕 애정운"), ("career", "💼 직업운")) if fortune[k]]
        if items:  # 운세 3종은 나란히 (좁은 화면에서는 세로)
            st.markdown("##### 운세 경향")
            cols = st.columns(len(items))
            for col, (label, text) in zip(cols, items):
                with col:
                    card(label, text)
        if res["advice"]:
            card("🍀 조언", res["advice"])
    st.caption("※ 재미로 보는 관상 풀이이며 과학적·의학적 근거가 없습니다. 결과는 참고만 해주세요.")


# ---------- 분석 실행 (버튼 → 분석 중 → 결과) ----------
def start_analysis():
    """버튼 콜백: 어떤 사진에 대한 분석인지 기록하고 분석 상태로 들어간다."""
    up = st.session_state.get("photo")
    st.session_state.pending = up.file_id if up is not None else None
    st.session_state.result = None
    st.session_state.error = None


def run_analysis(image, file_id):
    """API 를 호출해 결과나 오류를 session_state 에 남긴다. 호출 시도는 사용 횟수에 포함한다."""
    key = get_api_key()
    try:
        if not key:
            raise reading.ReadingError("no_key", "OpenAI API 키가 설정되지 않았습니다. 키 설정 후 다시 시도해주세요.")
        if not reserve_global_use():
            raise reading.ReadingError("global_limit", "오늘 서비스 전체 분석 가능 횟수를 모두 사용했어요. 내일 다시 이용해주세요.")
        st.session_state.usage_count = st.session_state.get("usage_count", 0) + 1  # 비용 보호: 시도 단위로 센다
        data_url = reading.prepare_image(image)
        result = reading.request_reading(data_url, key, get_setting("OPENAI_MODEL"))
        st.session_state.result = {"file_id": file_id, "data": result}
    except reading.ReadingError as e:
        st.session_state.error = {"file_id": file_id, "message": e.message}
    except Exception:  # 예기치 못한 오류도 내부 정보 없이 안내만 한다
        st.session_state.error = {"file_id": file_id, "message": "분석 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요."}
    finally:
        st.session_state.pending = None


# ---------- 화면 ----------
# 고정된 스타일만 주입한다. (모델이 만든 텍스트는 이 블록에 절대 넣지 않는다)
st.markdown(
    """
    <style>
    .block-container { max-width: 700px; padding-top: 2.5rem; padding-bottom: 3rem; }
    h1 { text-align: center; color: #7f1d1d; letter-spacing: -0.02em; }
    [data-testid="stHeaderActionElements"] { display: none; }  /* 제목 옆 '링크 복사' 앵커 아이콘 숨김 */
    .intro { text-align: center; color: #6b5a47; margin: -0.4rem 0 1.2rem; }
    [data-testid="stCaptionContainer"] { text-align: center; }

    /* 미리보기: 가운데 정렬, 좁은 화면에서는 폭을 넘지 않게 */
    [data-testid="stImage"], [data-testid="stImageContainer"] { max-width: 100% !important; }
    [data-testid="stImage"] { width: 100% !important; display: flex; flex-direction: column; align-items: center; }
    [data-testid="stFullScreenFrame"] { display: flex; justify-content: center; }
    [data-testid="stImageContainer"] { margin: 0 auto; }
    [data-testid="stImage"] img { max-width: 100% !important; height: auto !important; border-radius: 14px; }
    [data-testid="stImageCaption"] { text-align: center; }

    /* 주 버튼: 전체 폭 */
    .st-key-analyze-btn, .st-key-analyze-btn [data-testid="stButton"] { width: 100% !important; }
    .st-key-analyze-btn button { width: 100% !important; padding: 0.7rem 1rem; font-size: 1.05rem; font-weight: 600; border-radius: 12px; }

    /* 결과 카드 */
    [class*="st-key-card-"], .st-key-summary-card { border-radius: 14px; }
    [class*="st-key-card-"] { background: #fffdf8; }
    /* 같은 줄에 놓인 카드(부위별, 운세)는 높이를 맞춘다 */
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
    [data-testid="stColumn"] [data-testid="stLayoutWrapper"],
    [data-testid="stColumn"] [class*="st-key-card-"] { height: 100%; }

    /* 안내 박스: 파란색 대신 테마에 맞는 따뜻한 톤 */
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { background: #f3ead9; color: #5b4630; }
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) * { color: #5b4630; }
    [data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) [data-testid="stAlertContainer"] { background: transparent; }
    .st-key-summary-card { background: #f8e9e3 !important; border-color: #d9a79a !important; }
    .st-key-summary-card, .st-key-summary-card * { text-align: center !important; }
    .st-key-summary-card h3 { color: #7f1d1d; margin: 0; padding: 0.2rem 0 0.5rem; line-height: 1.45; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔮 AI 관상 보기")
st.markdown('<p class="intro">얼굴 사진을 올리면 AI가 관상을 풀이해 드려요.</p>', unsafe_allow_html=True)
st.info(
    "재미로 보는 콘텐츠이며 과학적·의학적 근거가 없습니다.\n\n"
    "업로드한 사진은 분석을 위해 OpenAI로 전송되며(전송된 데이터는 OpenAI의 데이터 처리 정책을 따릅니다), "
    "이 앱은 사진을 저장하지 않습니다. 본인 또는 동의받은 사람의 사진만 올려주세요. 미성년자의 사진은 올리지 않기를 권장합니다."
)

if not get_api_key():
    st.warning(
        "OpenAI API 키가 설정되지 않았습니다. 로컬에서는 `.env` 파일에, "
        "배포 환경에서는 Streamlit Secrets에 `OPENAI_API_KEY`를 설정해주세요."
    )

uploaded = st.file_uploader(
    f"얼굴 사진 업로드 (jpg, png, webp / {MAX_UPLOAD_MB}MB 이하)", type=ALLOWED_TYPES, key="photo"
)

image, file_id = None, None
if uploaded is not None:
    image, error = read_image(uploaded)
    if error:
        st.error(error)
    else:
        file_id = uploaded.file_id
        st.image(image, caption="업로드한 사진", width=360)

# 분석 중이던 사진이 바뀌었거나 사라졌으면 분석 상태를 취소한다.
pending = st.session_state.get("pending")
if pending is not None and (image is None or pending != file_id):
    st.session_state.pending = pending = None

left = remaining_uses()
analyzing = pending is not None
st.button(
    "관상 보기",
    type="primary",
    key="analyze-btn",  # CSS 에서 전체 폭으로 늘리기 위한 클래스(st-key-analyze-btn)
    on_click=start_analysis,
    disabled=image is None or analyzing or left == 0,  # 분석 중에는 비활성화해서 중복 호출을 막는다
)
if left == 0:
    st.error(f"오늘 분석 가능 횟수({DAILY_LIMIT}회)를 모두 사용했어요. 내일 다시 이용해주세요.")

st.subheader("풀이 결과")
if analyzing:
    with st.spinner("관상을 살펴보는 중이에요... (보통 10~20초)"):
        run_analysis(image, file_id)
    st.rerun()  # 버튼 상태와 남은 횟수를 갱신해서 다시 그린다

res, err = st.session_state.get("result"), st.session_state.get("error")
if res and res["file_id"] == file_id and file_id is not None:
    show_result(res["data"])
elif err and err["file_id"] == file_id and file_id is not None:
    st.error(err["message"])
else:
    st.caption("사진을 올리고 '관상 보기'를 누르면 여기에 결과가 표시됩니다.")

st.divider()
st.caption(f"오늘 남은 횟수: {remaining_uses()} / {DAILY_LIMIT}")
