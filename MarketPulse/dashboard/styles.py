import streamlit as st

def apply_custom_theme():
    """Applies modern CSS for visual styling."""
    st.markdown(
        """
        <style>
        /* Base page styling */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
        
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Inter', sans-serif;
            background-color: #0E1117;
            color: #E0E6ED;
        }
        
        /* Header titles */
        h1, h2, h3 {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            color: #FFFFFF;
            letter-spacing: -0.02em;
        }
        
        /* Custom Cards */
        .kpi-card {
            background: linear-gradient(135deg, #161A22 0%, #1F242E 100%);
            border: 1px solid #2D333F;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            margin-bottom: 15px;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            border-color: #3F485B;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }
        .kpi-title {
            font-size: 0.85rem;
            color: #8C9BAE;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 5px;
            font-weight: 600;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFFFFF;
            font-family: 'Outfit', sans-serif;
        }
        .kpi-delta {
            font-size: 0.85rem;
            margin-top: 5px;
            font-weight: 600;
        }
        .kpi-delta.up {
            color: #00E676;
        }
        .kpi-delta.down {
            color: #FF5252;
        }
        
        /* Glassmorphic alerts */
        .glass-alert {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            border-left: 4px solid #1F497D;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_kpi_card(title: str, value: str, delta: str = "", delta_type: str = "neutral"):
    """Helper to render a styled KPI card."""
    delta_class = ""
    if delta_type == "up":
        delta_class = "kpi-delta up"
        delta_sign = "▲ "
    elif delta_type == "down":
        delta_class = "kpi-delta down"
        delta_sign = "▼ "
    else:
        delta_class = "kpi-delta"
        delta_sign = ""
        
    delta_html = f'<div class="{delta_class}">{delta_sign}{delta}</div>' if delta else ""
    
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True
    )
