import streamlit as st
from opsci_toolbox.helpers.common import load_pickle, read_json
from opsci_toolbox.helpers.nlp import sample_most_engaging_posts
from opsci_toolbox.helpers.dataviz import subplots_bar_per_day_per_cat, create_scatter_plot, add_shape, pie
from eldar import Query

def format_number(number):
    if number < 1000:
        return str(number)
    elif number < 1000000:
        return f"{number / 1000:.1f}K"
    elif number < 1000000000:
        return f"{number / 1000000:.1f}M"
    else:
        return f"{number / 1000000000:.1f}B"

def main():
    st.set_page_config(
        page_title="Search engine",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title('Settings')
    st.title('Search') 

    df = load_pickle("data/df_prod.pickle")
    channel_color_palette = read_json("data/channel_color_palette.json")
    reaction_color_palette = read_json("data/reaction_color_palette.json")
    sentiment_color_palette = read_json("data/sentiment_color_palette.json")


    txt_query = st.sidebar.text_area("Boolean search", value="macron", height=None, max_chars=None, key=None, help=None, on_change=None, args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="visible")
    
    date = st.sidebar.date_input("Timerange", value = [df['datetime'].min(), df['datetime'].max()], min_value=df['datetime'].min(), max_value=df['datetime'].max(), format="YYYY/MM/DD", label_visibility="visible")

    lang = st.sidebar.selectbox("Language", ['english', 'russian'], index = 0, placeholder="Choose an option", label_visibility="visible")
    ignore_case = st.sidebar.toggle("Ignore case", value=True, label_visibility="visible")
    ignore_accent = st.sidebar.toggle("Ignore accent", value=True, label_visibility="visible")
    match_word = st.sidebar.toggle("Match words", value=True, label_visibility="visible")

    rolling_period = st.sidebar.text_input("Rolling period", value='1M', label_visibility="visible")

    if lang == "english":
        col_search = "translated_text"
    else :
        col_search = "text"

    df = df[(df['date'] >= date[0]) & (df['date'] <= date[1])]

    boolean_query = Query(txt_query, ignore_case=ignore_case, ignore_accent=ignore_accent, match_word=match_word)

    df = df[df[col_search].apply(boolean_query)]

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        st.metric("Verbatims", format_number(df['uniq_id'].nunique()), label_visibility="visible")
    with col2:
        st.metric("Channels", format_number(df['channel_id'].nunique()), label_visibility="visible")
    with col3:
        st.metric("Views", format_number(df['views'].sum()), label_visibility="visible")
    with col4:
        st.metric("Engagements", format_number(df['engagements'].sum()), label_visibility="visible")


    metrics = {
    'posts' : ('uniq_id',"nunique"),
    'views': ('views', 'sum'),
    'engagements': ('engagements', 'sum'),
    'reactions': ('total_reactions', 'sum'),
    'replies': ('replies_count', 'sum'),
    'forwards': ('forwards', 'sum')
    }

    df_trends_channels = df.copy()
    df_trends_channels.set_index('datetime', inplace=True)
    df_trends_channels = df_trends_channels.groupby("channel").resample(rolling_period).agg(**metrics).reset_index()
    df_trends_channels["datetime"]=df_trends_channels["datetime"].dt.strftime("%Y-%m-%d")
    df_trends_channels['color'] = df_trends_channels['channel'].map(channel_color_palette)

    for key in metrics.keys():
        df_trends_channels["total_"+str(key)] = df_trends_channels.groupby(["datetime"])[key].transform('sum')
        df_trends_channels['per_'+str(key)] = (df_trends_channels[key] / df_trends_channels["total_"+str(key)])

    fig_trend_post_channel = subplots_bar_per_day_per_cat(df_trends_channels, "datetime", "channel", ["posts"], "color", xaxis_title = "Date", y_axis_titles =  ["posts"],  title_text = "Trend", vertical_spacing = 0, width = 1500, height = 500, marker_color = "indianred", line_color = "#273746", plot_bgcolor=None, paper_bgcolor=None, template = "plotly")
    fig_trend_post_channel.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.3,
        xanchor="right",
        x=1
        ))
    
    st.plotly_chart(fig_trend_post_channel, use_container_width=True, sharing="streamlit", theme="streamlit")


    if len(df)>2000:
        s=2000/len(df)
        df_sample_engaging_posts = sample_most_engaging_posts(df, "type_engagement", "engagements", sample_size= s, min_size=10)
    else:
        df_sample_engaging_posts=df

    fig = create_scatter_plot(
    df_sample_engaging_posts, 
    "tx_total_reactions", 
    "tx_views", 
    "type_engagement", 
    reaction_color_palette, 
    "type_engagement", 
    None, 
    "translated_text", 
    title="Posts reaction", 
    x_axis_label="<< under reaction -- over reaction >>", 
    y_axis_label="<< under visibility -- over visibility >", 
    width=1000, height=1000, 
    size_value =6, 
    opacity=0.8, 
    plot_bgcolor="#FFFFFF", 
    paper_bgcolor="#FFFFFF", 
    line_width=0.5, 
    line_color="white", 
    template="plotly", 
    xaxis_range=[-1.05,1.05], 
    yaxis_range=[-1.05,1.05],
    yaxis_showgrid = True, 
    xaxis_showgrid = True
    )
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="right",
    x=1
    ))

    fig = add_shape(fig, shape_type = "rect", x0= -1, y0= -1, x1 = 0, y1=0, fillcolor= 'Silver')
    fig = add_shape(fig, shape_type = "rect", x0= -1, y0= 0, x1 = 0, y1=1, fillcolor= 'Plum')
    fig = add_shape(fig, shape_type = "rect", x0= 0, y0= 0, x1 = 1, y1=1, fillcolor= 'LightSeaGreen')
    fig = add_shape(fig, shape_type = "rect", x0= 0, y0= -1, x1 = 1, y1=0, fillcolor= '#F5B7B1')

    st.plotly_chart(fig, use_container_width=True, sharing="streamlit", theme="streamlit")

    col_pie1, col_pie2 = st.columns(2)
    with col_pie1:
        df_reaction_type = df.groupby("type_engagement").agg(**metrics).reset_index()
        df_reaction_type['color'] = df_reaction_type['type_engagement'].map(reaction_color_palette)
        fig_reaction_posts = pie(df_reaction_type, "type_engagement", "posts", "color", title =  "Posts distribution per reaction type", template = "plotly",  width = 500, height = 500, plot_bgcolor=None, paper_bgcolor=None, showlegend = False)
        st.plotly_chart(fig_reaction_posts, use_container_width=True, sharing="streamlit", theme="streamlit")


    with col_pie2:
        df_sentiment = df.groupby("sentiment").agg(**metrics).reset_index()
        df_sentiment['color'] = df_sentiment['sentiment'].map(sentiment_color_palette)
        fig_sentiment = pie(df_sentiment, "sentiment", "posts", "color", title =  "Posts distribution per reaction type", template = "plotly",  width = 500, height = 500, plot_bgcolor=None, paper_bgcolor=None, showlegend = False)
        st.plotly_chart(fig_sentiment, use_container_width=True, sharing="streamlit", theme="streamlit")
        fig_sentiment.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0,
            xanchor="right",
            x=1
            ))

    cols_to_display=["channel", "datetime", col_search, "views", "engagements"]
    st.dataframe(df[cols_to_display].reset_index(drop=True).sort_values(by="engagements", ascending=False), hide_index=True)


if __name__ == "__main__":
    main()