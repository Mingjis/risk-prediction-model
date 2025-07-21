import streamlit as st
import pandas as pd
import pickle
import numpy as np
import warnings
from catboost import CatBoostClassifier

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="건설현장 사망사고 사전예방 시스템",
    page_icon="🏗️",
    layout="wide"
)

# 🎯 모델 및 리소스 로딩
@st.cache_resource
def load_models():
    """모델과 인코더들을 로드합니다."""
    try:
        # 위험도 데이터 로드
        with open("risk_model_average.pkl", "rb") as f:
            risk_data = pickle.load(f)
        
        # 인코더 로드
        with open("encoders_injury.pkl", "rb") as f:
            injury_encoders = pickle.load(f)
        
        with open("encoders_cause.pkl", "rb") as f:
            cause_encoders = pickle.load(f)
        
        return injury_encoders, cause_encoders, risk_data
    
    except Exception as e:
        st.error(f"데이터 로드 중 오류 발생: {str(e)}")
        return None, None, None

# 📦 데이터 로드
injury_encoders, cause_encoders, risk_data = load_models()

# 🎛️ 메인 앱
def main():
    if injury_encoders is None:
        st.error("데이터를 로드할 수 없습니다. 파일을 확인해주세요.")
        return
    
    # 타이틀
    st.title("🏗️ 건설 재해 사망 위험도 예측기")
    st.markdown("**아래 정보를 입력하면 사고유형, 기인물, 위험도를 예측해줍니다**")
    st.markdown("---")
    
    # 사이드바에 입력 옵션
    st.sidebar.header("입력 정보")
    
    # 입력 선택
    facility_type = st.sidebar.selectbox(
        "시설 유형 (Facility Type)", 
        injury_encoders['Facility type'].classes_,
        help="작업이 수행될 시설의 유형을 선택하세요"
    )
    
    work_type = st.sidebar.selectbox(
        "작업 유형 (Work Type)", 
        injury_encoders['Work type'].classes_,
        help="수행할 작업의 유형을 선택하세요"
    )
    
    # 예측 버튼
    if st.sidebar.button("위험도 예측", type="primary"):
        with st.spinner("예측 중..."):
            try:
                # 🔮 정교한 규칙 기반 예측
                st.success("🎯 예측 결과 (정교한 규칙 기반)")
                
                # 작업 유형별 상해 유형 매핑 (더 정교하게)
                injury_mapping = {
                    'Steel work': 'Fall',
                    'Reinforced concrete work': 'Fall',
                    'Roof and gutter work': 'Fall',
                    'Scaffolding work': 'Fall',
                    'Foundation work': 'Fall',
                    'Electric wiring work': 'Electric shock',
                    'Fire protection work': 'Electric shock',
                    'Welding work': 'Fire',
                    'Paint work': 'Fire',
                    'Carpentry work': 'Cut',
                    'Metal work': 'Cut',
                    'Stone work': 'Cut',
                    'Demolition work': 'Collapse',
                    'Earth work': 'Collapse',
                    'Excavation work': 'Collapse',
                    'Construction machine': 'Be bumped',
                    'Transportation': 'Be bumped',
                    'Crane operation': 'Be bumped',
                    'Plumbing work': 'Fall beneath',
                    'Waterproof work': 'Fall beneath',
                    'Air handler work': 'Fall',
                    'Duct work': 'Fall',
                    'Insulation work': 'Fall',
                    'Interior finishing work': 'Cut',
                    'Tile work': 'Cut',
                    'Window and glass work': 'Cut',
                    'Landscaping work': 'Cut',
                    'Masonry work': 'Fall',
                    'Temporary work': 'Fall',
                    'Maintenance work (including demolition)': 'Collapse',
                    'Tunnel work': 'Collapse',
                    'Harbor work': 'Be drowned',
                    'River work': 'Be drowned'
                }
                
                # 시설 유형별 원인 자료 매핑 (더 정교하게)
                cause_mapping = {
                    'Factory': 'Power machine',
                    'Industrial facility': 'Power machine',
                    'Power Plant': 'Electrical equipment',
                    'Office': 'Ladder',
                    'Complex building': 'Ladder',
                    'Multi-family house': 'Ladder',
                    'Single family house': 'Ladder',
                    'Accommodation': 'Ladder',
                    'Bridge': 'Tower crane',
                    'Road': 'Transportation vehicle',
                    'Railroad': 'Transportation vehicle',
                    'Transportation': 'Transportation vehicle',
                    'Seaport': 'Transportation vehicle',
                    'Airport': 'Transportation vehicle',
                    'Tunnel': 'Excavator',
                    'Water & Sewage': 'Excavator',
                    'Environment facility': 'Excavator',
                    'Medical facility': 'Safety facilities',
                    'Education facility': 'Safety facilities',
                    'Cultural & Assembly': 'Safety facilities',
                    'Religious': 'Safety facilities',
                    'Sport facility': 'Safety facilities',
                    'Retail': 'Material',
                    'Landscape': 'Hand tools',
                    'Traditional Korean House': 'Hand tools',
                    'Elderly & Child care': 'Safety facilities',
                    'Neighborhood living facility': 'Material',
                    'Amusement': 'Safety facilities',
                    'Animal & Plant care': 'Hand tools',
                    'Broadcasting & Communication': 'Electrical equipment',
                    'Correctional & Military': 'Safety facilities',
                    'Hazardous Waste Storage': 'Material'
                }
                
                # 매핑 기반 예측
                predicted_injury = injury_mapping.get(work_type, 'Fall')
                predicted_cause = cause_mapping.get(facility_type, 'Material')
                
                # 매핑되지 않은 경우 더 똑똑한 기본값 설정
                if work_type not in injury_mapping:
                    if 'electric' in work_type.lower():
                        predicted_injury = 'Electric shock'
                    elif 'fire' in work_type.lower() or 'weld' in work_type.lower():
                        predicted_injury = 'Fire'
                    elif 'cut' in work_type.lower() or 'saw' in work_type.lower():
                        predicted_injury = 'Cut'
                    elif 'machine' in work_type.lower():
                        predicted_injury = 'Be bumped'
                    elif 'water' in work_type.lower():
                        predicted_injury = 'Be drowned'
                    else:
                        predicted_injury = 'Fall'
                
                if facility_type not in cause_mapping:
                    if 'factory' in facility_type.lower() or 'industrial' in facility_type.lower():
                        predicted_cause = 'Power machine'
                    elif 'office' in facility_type.lower() or 'building' in facility_type.lower():
                        predicted_cause = 'Ladder'
                    elif 'transport' in facility_type.lower() or 'road' in facility_type.lower():
                        predicted_cause = 'Transportation vehicle'
                    elif 'water' in facility_type.lower() or 'tunnel' in facility_type.lower():
                        predicted_cause = 'Excavator'
                    else:
                        predicted_cause = 'Material'
                
                # 위험도 계산
                injury_risk = risk_data.get('injury', {}).get(predicted_injury, 0.15)
                cause_risk = risk_data.get('cause', {}).get(predicted_cause, 0.15)
                final_risk = (injury_risk + cause_risk) / 2
                
                # 결과 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚨 예측된 상해 유형")
                    st.metric(
                        label="Injury Type",
                        value=predicted_injury,
                        delta="규칙 기반 예측"
                    )
                    
                    # 위험도 표시
                    risk_color = "🔴" if injury_risk > 30 else "🟡" if injury_risk > 15 else "🟢"
                    st.metric(
                        label="상해 위험도",
                        value=f"{risk_color} {injury_risk:.4f}%",
                        help="높을수록 위험한 상해 유형입니다"
                    )
                
                with col2:
                    st.subheader("⚠️ 예측된 원인 자료")
                    st.metric(
                        label="Original Cause Material",
                        value=predicted_cause,
                        delta="규칙 기반 예측"
                    )
                    
                    # 위험도 표시
                    risk_color = "🔴" if cause_risk > 30 else "🟡" if cause_risk > 15 else "🟢"
                    st.metric(
                        label="원인 위험도",
                        value=f"{risk_color} {cause_risk:.4f}%",
                        help="높을수록 위험한 원인 자료입니다"
                    )
                
                # 종합 위험도
                st.markdown("---")
                st.subheader("📊 종합 위험도 평가")
                
                # 위험도 레벨 결정
                if final_risk > 30:
                    risk_level = "🔴 높음"
                    risk_msg = "매우 주의가 필요합니다!"
                elif final_risk > 15:
                    risk_level = "🟡 보통"
                    risk_msg = "적절한 안전조치가 필요합니다."
                else:
                    risk_level = "🟢 낮음"
                    risk_msg = "상대적으로 안전한 수준입니다."
                
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("💀 최종 사망 위험도", f"{final_risk:.4f}%")
                with col4:
                    st.metric("위험 수준", risk_level)
                with col5:
                    st.info(risk_msg)
                
                
                # 추가 정보
                with st.expander("예측 방법 정보"):
                    st.markdown("""
                    **현재 모델 파일에 호환성 문제가 있어 규칙 기반 예측을 사용하고 있습니다.**
                    
                    - 작업 유형에 따른 일반적인 상해 유형 예측
                    - 시설 유형에 따른 일반적인 원인 자료 예측
                    - 실제 위험도 데이터 기반 점수 계산
                    
                    더 정확한 예측을 위해서는 모델 파일을 다시 훈련하거나 호환성 문제를 해결해야 합니다.
                    """)
                
            except Exception as e:
                st.error(f"예측 중 오류 발생: {str(e)}")
    
    # 정보 섹션
    st.markdown("---")
    st.subheader("ℹ️ 사용 방법")
    st.markdown("""
    1. 왼쪽 사이드바에서 **시설 유형**과 **작업 유형**을 선택하세요
    2. **위험도 예측** 버튼을 클릭하여 결과를 확인하세요
    3. 예측된 상해 유형과 원인 자료, 그리고 위험도를 확인하세요
    4. 위험도가 높은 경우 추가적인 안전조치를 취하세요
    """)
    
    # 면책 조항
    st.markdown("---")
    st.caption("⚠️ 이 예측 시스템은 참고용으로만 사용되어야 하며, 실제 안전 관리에서는 전문가의 판단과 현장 상황을 종합적으로 고려해야 합니다.")

if __name__ == "__main__":
    main()
