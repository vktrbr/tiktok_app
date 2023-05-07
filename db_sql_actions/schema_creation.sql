drop schema users cascade;

create schema users

    create table users.user
    (
        id_user         text not null,
        txt_username    text not null,
        hash_pass       text not null,
        dt_registration date not null,
        txt_email       text,
        primary key (id_user)
    )

    create table users.session
    (
        id_user       text      not null,
        id_session    text      not null,
        dtime_session timestamp not null,
        primary key (id_session),
        foreign key (id_user) references users.user (id_user)
    )

    create table users.video
    (
        id_video           text      not null,
        id_session         text      not null,
        dtime_upload       timestamp not null,
        num_metric_like    numeric,
        num_metric_dislike numeric,
        foreign key (id_session) references users.session (id_session)

    )
;


select *
from users.user;

select *
from users.video;

select *
from pg_stat_activity;

select pg_cancel_backend(28782);
select pg_terminate_backend(28782);

