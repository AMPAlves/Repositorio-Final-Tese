attribute(task,root,distribution).
attribute(number_rows,root,171).

entity(field,root,(f,0)).
attribute((field,name),(f,0),gene).
attribute((field,type),(f,0),string).
attribute((field,unique),(f,0),6).
entity(field,root,(f,1)).
attribute((field,name),(f,1),pos).
attribute((field,type),(f,1),number).
attribute((field,unique),(f,1),146).
entity(field,root,(f,2)).
attribute((field,name),(f,2),chrom).
attribute((field,type),(f,2),string).
attribute((field,unique),(f,2),20).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(scale,(v,0),(s,0)).
entity(scale,(v,0),(s,1)).
entity(scale,(v,0),(s,2)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).
entity(encoding,(m,0),(e,2)).

attribute((encoding,field),(e,0),pos).
attribute((encoding,channel),(e,0),x).
attribute((scale,channel),(s,0),x).
attribute((scale,type),(s,0),numcate).
attribute((encoding,field),(e,2),gene).
attribute((encoding,channel),(e,2),color).
attribute((scale,channel),(s,2),color).
attribute((scale,type),(s,2),categorical).
attribute((encoding,field),(e,1),chrom).
attribute((encoding,channel),(e,1),y).
attribute((scale,channel),(s,1),y).
attribute((scale,type),(s,1),ordinal).
