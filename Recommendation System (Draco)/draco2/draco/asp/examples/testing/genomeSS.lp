attribute(task,root,distribution).
attribute(number_rows,root,171).

entity(field,root,(f,0)).
attribute((field,name),(f,0),gene).
attribute((field,type),(f,0),string).
attribute((field,unique),(f,0),6).
entity(field,root,(f,1)).
attribute((field,name),(f,1),pos).
attribute((field,type),(f,1),string).
attribute((field,unique),(f,1),146).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).
attribute((encoding,field),(e,0),gene).
attribute((encoding,field),(e,1),pos).