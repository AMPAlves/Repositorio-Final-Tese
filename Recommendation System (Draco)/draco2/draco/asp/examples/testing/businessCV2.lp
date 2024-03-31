attribute(task,root,relationship).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),discount).
attribute((field,type),(f,0),number).
attribute((field,unique),(f,0),12).
entity(field,root,(f,1)).
attribute((field,name),(f,1),month).
attribute((field,type),(f,1),string).
attribute((field,unique),(f,1),1237).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).
entity(scale,(v,0),(s,0)).
entity(scale,(v,0),(s,1)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).

attribute((encoding,field),(e,0),discount).
attribute((encoding,channel),(e,0),y).
attribute((encoding,aggregate),(e,0),mean).
attribute((scale,channel),(s,0),y).
attribute((scale,type),(s,0),linear).
attribute((encoding,field),(e,1),month).
attribute((encoding,channel),(e,1),x).
attribute((scale,channel),(s,1),x).

#show entity/3.
#show attribute/3.