import sys
from time import sleep

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, '..')

from db_connect import PGConnection

st.set_page_config(layout='wide', page_title='kpi tiktok forecast')

st.title('TikTok Forecast App Analytics')

if 'connect' not in st.session_state:
    st.session_state['pg'] = PGConnection()

query_user_cnt = """
    select count(1) as users_cnt
    from users."user";
"""

query_session_cnt = """
    select count(1) as session_cnt
    from users.session;
"""

query_video_cnt = """
    select count(1) as session_cnt
    from users.video;
"""

query_avg_video_over_session = """
select count(id_video)::numeric / count(id_session) as avg_video_over_session
from users.session
         left join users.video using (id_session)
"""

query_session_distribution = """
with a as (select id_user, count(1) as "Session per user"
           from users.session
                    left join users."user" using (id_user)
           where txt_username != 'tsoi'
           group by 1
           order by 1)
select "Session per user", count(1) as "Number of cases"
from a
group by 1;
"""

query_session_avg_time_video_eval = """
select (dtime_session)::date                                                            as "Scoring Date",
       round(avg(extract(epoch
                         from video.dtime_upload - session.dtime_session))::numeric, 2) as "Avg scoring time, sec"
from users."user"
         left join users.session using (id_user)
         left join users.video using (id_session)
where id_video is not null
group by 1
order by 1;
"""

kpi: list[st.delta_generator.DeltaGenerator] = st.columns((2, 2, 2, 2, 2, 2))[1:-1]

user_cnt = st.session_state['pg'].select(query_user_cnt)[0][0]
kpi[0].metric('Number of users', user_cnt)

session_cnt = st.session_state['pg'].select(query_session_cnt)[0][0]
kpi[1].metric('Number of sessions', session_cnt)

video_cnt = st.session_state['pg'].select(query_video_cnt)[0][0]
kpi[2].metric('Number of videos', video_cnt)

video_over_session = st.session_state['pg'].select(query_avg_video_over_session)[0][0]
kpi[3].metric('Average video per session', float(round(video_over_session, 3)))

charts = st.columns(2)
charts: list[st.delta_generator.DeltaGenerator]

st.session_state['pg'].open()

df_session_distribution = pd.read_sql(query_session_distribution, st.session_state['pg'].connection)
charts[0].plotly_chart(px.bar(df_session_distribution, x='Session per user', y='Number of cases'),
                       use_container_width=True)

df_video_scoring_time = pd.read_sql(query_session_avg_time_video_eval, st.session_state['pg'].connection)
charts[1].plotly_chart(px.line(df_video_scoring_time, x="Scoring Date", y="Avg scoring time, sec"),
                       use_container_width=True)

st.session_state['pg'].close()

sleep(300)
