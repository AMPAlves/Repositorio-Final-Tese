attribute(task,root,compare).
attribute(number_rows,root,171).

entity(field,root,(f,0)).
attribute((field,name),(f,0),gene).
attribute((field,type),(f,0),string).
attribute((field,unique),(f,0),6).
entity(field,root,(f,1)).
attribute((field,name),(f,1),pos).
attribute((field,type),(f,1),number).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(scale,(v,0),(s,0)).
entity(scale,(v,0),(s,1)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).

attribute((encoding,field),(e,0),pos).
attribute((encoding,channel),(e,0),x).
attribute((scale,channel),(s,0),x).
attribute((scale,type),(s,0),ordinal).
attribute((encoding,field),(e,1),gene).

