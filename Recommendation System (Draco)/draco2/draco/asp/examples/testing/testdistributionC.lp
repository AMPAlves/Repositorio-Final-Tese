attribute(task,root,distribution).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),category).
attribute((field,type),(f,0),string).
attribute((field,unique),(f,0),3).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).
attribute((encoding,field),(e,0),category).
attribute((encoding,channel),(e,0),x).
attribute((encoding,aggregate),(e,1),count).
attribute((encoding,channel),(e,1),y).

#show entity/3.
#show attribute/3.