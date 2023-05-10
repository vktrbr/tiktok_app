-- распределение кол-ва сессий на юзера
with a as (select id_user, count(1) as "Кол-во сессий на пользователя"
           from users.session
                    left join users."user" using (id_user)
           group by 1
           order by 1)
select "Кол-во сессий на пользователя", count(1) as "Кол-во наблюдений"
from a
group by 1;

-- Кол-во зарегистрированных пользователей
select count(1) as users_cnt
from users."user";

-- Кол-во сессий
select count(1) as session_cnt
from users.session;


-- Кол-во загруженных видео
select count(1) as video_cnt
from users.video;


-- среднее кол-во видео на сессию
select count(id_video)::numeric / count(id_session) as avg_video_over_session
from users.session
         left join users.video using (id_session);


-- среднее время между заходом и загрузкой видео
select (dtime_session)::date                                                            as "Дата скоринга",
       round(avg(extract(epoch
                         from video.dtime_upload - session.dtime_session))::numeric, 2) as "Среднее время скоринга, сек"
from users."user"
         left join users.session using (id_user)
         left join users.video using (id_session)
where id_video is not null
group by 1
order by 1;
