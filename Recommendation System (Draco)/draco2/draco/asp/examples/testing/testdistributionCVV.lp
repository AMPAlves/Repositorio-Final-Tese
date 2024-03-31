attribute(task,root,distribution).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),discount).
attribute((field,type),(f,0),number).
attribute((field,unique),(f,0),14).
entity(field,root,(f,1)).
attribute((field,name),(f,1),profit).
attribute((field,type),(f,1),number).
attribute((field,unique),(f,1),5528).
entity(field,root,(f,2)).
attribute((field,name),(f,2),category).
attribute((field,type),(f,2),string).
attribute((field,unique),(f,2),3).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).
entity(scale,(v,0),(s,0)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).
entity(encoding,(m,0),(e,2)).
attribute((encoding,field),(e,0),discount).
attribute((encoding,channel),(e,0),x).
attribute((scale,channel),(s,0),x).
attribute((scale,type),(s,0),ordinal).
attribute((encoding,field),(e,1),profit).
attribute((encoding,field),(e,2),category).

#show entity/3.
#show attribute/3.