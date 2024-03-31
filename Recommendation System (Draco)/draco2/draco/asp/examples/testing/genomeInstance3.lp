attribute(task,root,distribution).
attribute(number_rows,root,146).

entity(field,root,(f,0)).
attribute((field,name),(f,0),gene).
attribute((field,type),(f,0),string).
attribute((field,unique),(f,0),6).
entity(field,root,(f,1)).
attribute((field,name),(f,1),pos).
attribute((field,type),(f,1),number).
attribute((field,unique),(f,1),146).

entity(field,root,(f,2)).
attribute((field,name),(f,2),leftallele).
attribute((field,type),(f,2),string).
attribute((field,unique),(f,2),100).

entity(field,root,(f,3)).
attribute((field,name),(f,3),rightallele).
attribute((field,type),(f,3),string).
attribute((field,unique),(f,3),100).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(scale,(v,0),(s,0)).
entity(scale,(v,0),(s,2)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,2)).
entity(encoding,(m,0),(e,3)).
entity(encoding,(m,0),(e,4)).

attribute((encoding,field),(e,0),pos).
attribute((encoding,channel),(e,0),x).
attribute((scale,channel),(s,0),x).
attribute((scale,type),(s,0),linear).
attribute((encoding,field),(e,2),gene).
attribute((encoding,field),(e,3),leftallele).
attribute((encoding,field),(e,4),rightallele).