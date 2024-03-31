attribute(task,root,relationship).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),quantity).
attribute((field,type),(f,0),number).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).
entity(encoding,(m,0),(e,0)).
attribute((encoding,field),(e,0),quantity).

#show entity/3.
#show attribute/3.