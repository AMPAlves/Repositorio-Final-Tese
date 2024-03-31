attribute(task,root,compare).
attribute(number_rows,root,9994).

entity(field,root,(f,0)).
attribute((field,name),(f,0),sales).
attribute((field,type),(f,0),number).
attribute((field,unique),(f,0),5825).
entity(field,root,(f,1)).
attribute((field,name),(f,1),monthyear).
attribute((field,type),(f,1),datetime).
entity(field,root,(f,2)).
attribute((field,name),(f,2),category).
attribute((field,type),(f,2),string).
attribute((field,unique),(f,2),3).

entity(view,root,(v,0)).

entity(mark,(v,0),(m,0)).
entity(encoding,(m,0),(e,0)).
attribute((encoding,field),(e,0),sales).
entity(encoding,(m,0),(e,1)).
attribute((encoding,field),(e,1),monthyear).
entity(encoding,(m,0),(e,2)).
attribute((encoding,field),(e,2),category).

#show entity/3.
#show attribute/3.