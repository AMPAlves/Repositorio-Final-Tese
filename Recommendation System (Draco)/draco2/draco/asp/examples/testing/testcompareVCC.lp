attribute(task,root,compare).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),sales).
attribute((field,type),(f,0),number).
attribute((field,unique),(f,0),5528).
entity(field,root,(f,1)).
attribute((field,name),(f,1),month).
attribute((field,type),(f,1),string).
attribute((field,unique),(f,1),12).
entity(field,root,(f,2)).
attribute((field,name),(f,2),category).
attribute((field,type),(f,2),string).
attribute((field,unique),(f,2),3).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).
entity(encoding,(m,0),(e,2)).
attribute((encoding,field),(e,1),sales).
attribute((encoding,field),(e,2),month).
attribute((encoding,field),(e,0),category).

#show entity/3.
#show attribute/3.