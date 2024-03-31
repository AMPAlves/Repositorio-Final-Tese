attribute(task,root,distribution).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),quantity).
attribute((field,type),(f,0),number).
attribute((field,unique),(f,0),14).
attribute((field,name),(f,1),category).
attribute((field,type),(f,1),string).
attribute((field,unique),(f,1),3).
attribute((field,name),(f,2),region).
attribute((field,type),(f,2),string).
attribute((field,unique),(f,2),5).

entity(view,root,(v,0)).

entity(mark,(v,0),(m,0)).
entity(encoding,(m,0),(e,0)).
attribute((encoding,field),(e,0),quantity).
entity(encoding,(m,0),(e,1)).
attribute((encoding,field),(e,1),quantity).

#show entity/3.
#show attribute/3.