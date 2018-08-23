ajb 22 Aug 2018

See cl-minorminer.  You should compile the clib by going to clib/ and typing make (or use the binary already there).  Then
load the shared library: (cffi:load-foreign-library "minorminer/clib/libminorminer_clib.so") Then
(ld :minorminer) after adding the cl-minorminer directory to your asdf directory.  Then look at the examples at the bottom
of cl-minorminer.lisp
