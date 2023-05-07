```puml
@startuml

!theme plain
top to bottom direction
skinparam linetype ortho

class session {
   id_user: text
   dtime_session: timestamp
   id_session: text
}
class user {
   txt_username: text
   hash_pass: text
   dt_registration: date
   txt_email: text
   id_user: text
}
class video {
   id_video: text
   id_session: text
   dtime_upload: timestamp
   num_metric_like: numeric
   num_metric_dislike: numeric
}

session  -[#595959,plain]-^  user    : "id_user"
video    -[#595959,plain]-^  session : "id_session"
@enduml
```