import streamlit as st
import plotly.graph_objects as go

def plot_price_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color="#FFD700")))
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        plot_bgcolor="#000",
        paper_bgcolor="#000",
        font=dict(color="white")
    )
    st.plotly_chart(fig, use_container_width=True)
