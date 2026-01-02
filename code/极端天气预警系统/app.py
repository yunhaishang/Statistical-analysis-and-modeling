# app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime, timedelta

# ==============================
# 页面配置
# ==============================
st.set_page_config(
    page_title="🌤️ 极端天气预警预测系统",
    page_icon="⚠️",
    layout="centered"
)

st.title("🌤️ 极端天气预警预测原型系统")
st.markdown("基于历史预警数据，预测下次预警的**时间间隔**和**可能类型**。")

# ==============================
# 加载模型（缓存避免重复加载）
# ==============================
@st.cache_resource
def load_models():
    models = {}
    try:
        models['interval'] = joblib.load('models\\best_warning_interval_model.pkl')
        st.success("✅ 回归模型加载成功")
    except Exception as e:
        st.error(f"❌ 回归模型加载失败: {e}")
        models['interval'] = None

    try:
        models['type'] = joblib.load('models\\warning_type_classifier.pkl')
        st.success("✅ 分类模型加载成功")
    except Exception as e:
        st.error(f"❌ 分类模型加载失败: {e}")
        models['type'] = None
    return models

models = load_models()

# ==============================
# 用户输入表单
# ==============================
st.subheader("📌 请输入当前预警信息")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        warning_type = st.selectbox(
            "当前预警类型",
            options=["暴雨", "台风", "雷电", "雷雨大风", "高温", "强季风", "森林火险", "寒冷", "大雾"]
        )
        level = st.selectbox(
            "当前预警级别",
            options=["白色", "蓝色", "黄色", "橙色", "红色"]
        )
    
    with col2:
        date_input = st.date_input("当前日期", value=datetime.now().date())
        time_input = st.time_input("当前时间", value=datetime.now().time())
    
    submitted = st.form_submit_button("🔮 预测下次预警")

# ==============================
# 预测逻辑
# ==============================
if submitted:
    current_time = datetime.combine(date_input, time_input)
    
    input_data = {
        '预警类型': warning_type,
        '前次预警级别': level,
        '当前时间': current_time
    }
    
    st.subheader("🔍 预测结果")
    
    # --- 预测间隔天数（回归模型） ---
    if models['interval'] is not None:
        # 预警级别编码
        level_mapping = {'白色': 1, '蓝色': 2, '黄色': 3, '橙色': 4, '红色': 5}
        level_code = level_mapping.get(level, 2)
        
        # 默认历史特征
        type_avg_interval = {
            '暴雨': 15, '台风': 45, '强季风': 30, '雷电': 10,
            '雷雨大风': 12, '森林火险': 60, '高温': 20,
            '寒冷': 35, '大雾': 25
        }
        type_30d_count = {
            '暴雨': 3, '台风': 1, '强季风': 2, '雷电': 5,
            '雷雨大风': 4, '森林火险': 2, '高温': 3,
            '寒冷': 1, '大雾': 2
        }
        
        hist_avg = type_avg_interval.get(warning_type, 30)
        count_30d = type_30d_count.get(warning_type, 2)
        
        season = '春季' if current_time.month in [3,4,5] else (
                 '夏季' if current_time.month in [6,7,8] else (
                 '秋季' if current_time.month in [9,10,11] else '冬季'))
        is_workday = 1 if current_time.weekday() < 5 else 0
        
        input_df = pd.DataFrame([{
            '预警类型': warning_type,
            '月份': current_time.month,
            '季节': season,
            '前次预警级别编码': level_code,
            '历史平均间隔': hist_avg,
            '过去30天预警次数': count_30d,
            '是否工作日': is_workday
        }])
        
        try:
            pred_interval = models['interval'].predict(input_df)[0]
            interval_days = max(1, int(round(pred_interval)))
            next_date = current_time + timedelta(days=interval_days)
            
            st.success(f"⏱️ **下次预警预计在 {interval_days} 天后**")
            st.info(f"📅 预计时间：**{next_date.strftime('%Y-%m-%d %H:%M')}**")
        except Exception as e:
            st.error(f"回归模型预测出错: {e}")
    else:
        st.warning("⚠️ 回归模型未加载，跳过时间预测")
    
    # --- 预测预警类型（分类模型） ---
    if models['type'] is not None:
        input_df_type = pd.DataFrame([{
            '月份': current_time.month,
            '小时': current_time.hour,
            '是否工作日': 1 if current_time.weekday() < 5 else 0,
            '过去30天同类型预警次数': 2  # 默认值
        }])
        
        try:
            pred_type = models['type'].predict(input_df_type)[0]
            proba = models['type'].predict_proba(input_df_type)[0]
            max_proba = max(proba)
            
            st.success(f"🔮 **下次最可能的预警类型：{pred_type}**")
            st.info(f"置信度：**{max_proba:.1%}**")
        except Exception as e:
            st.error(f"分类模型预测出错: {e}")
    else:
        st.warning("⚠️ 分类模型未加载，跳过类型预测")

# ==============================
# 说明与帮助
# ==============================
st.markdown("---")
st.subheader("ℹ️ 系统说明")
st.markdown("""
- **回归模型**：预测从当前预警到下次预警的**间隔天数**
- **分类模型**：预测下次预警的**可能类型**
- 所有预测基于历史数据训练，**仅供参考**，不替代专业气象预报
- 本系统为**原型演示**，实际部署需接入实时数据库计算历史特征
""")

st.caption("© 2026 极端天气预警分析系统 | 基于机器学习")