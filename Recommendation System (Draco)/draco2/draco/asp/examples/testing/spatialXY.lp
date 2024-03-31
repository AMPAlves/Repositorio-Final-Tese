attribute(task,root,relationship).
attribute(number_rows,root,501).

entity(field,root,(f,0)).
attribute((field,name),(f,0),lat).
attribute((field,type),(f,0),number).
attribute((field,unique),(f,0),351).
entity(field,root,(f,1)).
attribute((field,name),(f,1),lng).
attribute((field,type),(f,1),number).
attribute((field,unique),(f,1),352).
entity(field,root,(f,2)).
attribute((field,name),(f,2),population).
attribute((field,type),(f,2),number).
attribute((field,unique),(f,2),500).

entity(view,root,(v,0)).
entity(mark,(v,0),(m,0)).

entity(encoding,(m,0),(e,0)).
entity(encoding,(m,0),(e,1)).
entity(encoding,(m,0),(e,2)).

attribute((encoding,field),(e,0),lat).
attribute((encoding,channel),(e,0),y).
attribute((encoding,field),(e,1),lng).
attribute((encoding,channel),(e,1),x).
attribute((encoding,field),(e,2),population).
#show entity/3.
#show attribute/3.