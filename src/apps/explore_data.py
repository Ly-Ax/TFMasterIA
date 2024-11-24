import pandas as pd
import os
import streamlit as st
import altair as alt

@st.cache_data
def load_data():
    df = pd.read_csv(os.getcwd() + '/data/clean/data_clean.csv', low_memory=False)
    df = df.sample(frac=0.1)
    return df

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle().encode(
        x=x_axis,
        y=y_axis,
        color='Default').interactive()

    st.write(graph)

def main():
    df = load_data()
    page = st.sidebar.selectbox("Select a Page", ["SBA Clean", "Exploration"])

    if page == "SBA Clean":
        st.header("SBA Data Cleaned")
        st.write("Select a page on the left")
        st.write(df)

    elif page == "Exploration":
        st.title("Data Exploration")
        x_axis = st.selectbox("Choose a variable for the X-axis", df.columns, index=3)
        y_axis = st.selectbox("Choose a variable for the Y-axis", df.columns, index=4)
        visualize_data(df, x_axis, y_axis)

if __name__ == '__main__':
    main()

# streamlit run c:/Ly-Ax/TFMasterIA/apps/explore_data.py
# streamlit run /Users/zorromac/Documents/Ly-Ax/TFMasterIA/apps/explore_data.py
