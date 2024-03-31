attribute(task,root,relationship).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),discount).
attribute((field,type),(f,0),number).
attribute((field,unique),(f,0),12).
entity(field,root,(f,1)).
attribute((field,name),(f,1),month).
attribute((field,type),(f,1),datetime).
attribute((field,unique),(f,1),1237).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).
attribute((encoding,field),(e,0),discount).
attribute((encoding,field),(e,1),month).

#show entity/3.
#show attribute/3.