(defpackage :minorminer
  (:use :cl :cffi)
  (:export
   #:find-embedding
   #:test-embedding
   #:test-embedding-fixed-chains
   #:embedding-failed
   #:cancel-embedding
   #:uncancel-embedding)
  (:documentation
   "A thin wrapper around a thin c library wrapper around the Minor Miner C++ library.
    To use this you need to compile libminorminer_clib.so.

    See test-embedding and test-embedding-fixed-chains for sample usage.

    For extensive use you might want to just run the embedding as a seperate process."))

(in-package :minorminer)

(cffi:load-foreign-library "../clib/libminorminer_clib.so") ;; modify this to point to the library.  Go to ../clib then type make to compile this.

;; Some raw types used in the c wrapper library

(defctype vectorint :pointer "A pointer to a C++ std::vector<int>")
(defctype vvi :pointer "A pointer to a C++ std::vector<vector<int>>") ;; Output of program
(defctype mivi :pointer "A pointer to a C++ std::map<int,vector<int>>") ;; used for initial-chains, restrict-chains, fixed-chains feature

(cffi:defcfun ("make_vectorint" make_vectorint) vectorint)

(cffi:defcfun ("vectorint_length" vectorint_length) :int
  (vectorint vectorint))

(cffi:defcfun ("vectorint_to_arrayint" vectorint_to_arrayint) (:pointer :int)
  (vectorint vectorint))

(cffi:defcfun ("vectorint_push_back" vectorint_push_back) :void
  (vectorint vectorint)
  (value :int))

(cffi:defcfun ("vectorint_print" vectorint_print) :void
  (vectorint vectorint))

(cffi:defcfun ("delete_vectorint" delete_vectorint) :void
  (vectorint vectorint))

(cffi:defcfun ("make_vvi" make_vvi) vvi)

(cffi:defcfun ("vvi_length" vvi_length) :int
  (vvi vvi))

(cffi:defcfun ("vvi_at" vvi_at) :pointer
  (vvi vvi)
  (index :int))

(cffi:defcfun ("vvi_push_back" vvi_push_back) :void
  (vvi vvi)
  (vectorint vectorint))

(cffi:defcfun ("vvi_print" vvi_print) :void
  (vvi vvi))

(cffi:defcfun ("delete_vvi" delete_vvi) :void
  (arg0 :pointer))

(cffi:defcfun ("make_mivi" make_mivi) :pointer)

(cffi:defcfun ("mivi_set" mivi_set) :void
  (mivi :pointer)
  (arg1 :int)
  (vectorint :pointer))

(cffi:defcfun ("mivi_map" mivi_map) :void
  (arg0 :pointer)
  (callback :pointer))

(cffi:defcfun ("delete_mivi" delete_mivi) :void
  (arg0 :pointer))

(defctype inputgraph :pointer "A pointer to an input_graph structure")

(cffi:defcfun ("make_graph" make_graph) inputgraph)

(cffi:defcfun ("delete_graph" delete_graph) :void
  (inputgraph inputgraph))

(cffi:defcfun ("push_back" push_back) :void
  (inputgraph inputgraph)
  (node1 :int)
  (node2 :int))

(cffi:defcfun ("num_nodes" num_nodes) :int
  (inputgraph inputgraph))

(cffi:defcfun ("num_edges" num_edges) :int
  (inputgraph inputgraph))

;; The default interface for cancellation of the embedding job, unless you pass in a cancelled_callback

(cffi:defcfun ("isCancelled" isCancelled) :int)
(cffi:defcfun ("setCancelled" setCancelled) :void)
(cffi:defcfun ("resetCancelled" resetCancelled) :void)


;; These are the options that the embedder uses.  See the documentation
;; for minor miner.  I haven't tested to see if the threading works badly
;; with SBCL.  Probably works fine.  If you really want to run this it's probably
;; better to just call the whole embedder as a separate process (ie through python
;; or something like that).
(defcstruct optional-parameters
  (max-no-improvement :int)
  (timeout :int)
  (tries :int)
  (verbose :int)
  (inner_rounds :int)
  (max_fill :int)
  (return_overlap :int)
  (chainlength_patience :int)
  (threads :int)
  (skip_initialization :int)
  (fixed_chains mivi)
  (initial_chains mivi)
  (restrict_chains mivi)
  (cancelled_callback :pointer)
  (display_callback :pointer))

(cffi:defcfun ("make_default_optional_parameters" make_default_optional_parameters) (:pointer (:struct optional-parameters))
  (fixed_chains mivi)
  (restrict_chains mivi)
  (initial_chains mivi))

(defun make-default-optional-parameters (&key fixed_chains restrict_chains initial_chains)
  (make_default_optional_parameters (or fixed_chains (cffi:null-pointer)) (or restrict_chains (cffi:null-pointer))
                                    (or initial_chains (cffi:null-pointer))))

(cffi:defcfun ("delete_optional_parameters" delete_optional_parameters) :void (params :pointer (:struct optional-parameters)))

(cffi:defcfun ("findembedding_clib" findembedding_clib) :int
  (small_graph inputgraph) ;; embed this graph
  (bigger_graph inputgraph) ;; into this graph
  (optional_parameters (:pointer (:struct optional-parameters)))
  (chains vvi))

(cffi:defcallback print-to-console :void ((msg :string))
  (ignore-errors (format t "~A~%" msg)))

(defparameter *cancel-embedding* 0)

(cffi:defcallback cancelled_callback :int ()
  *cancel-embedding*)

(defun cancel-embedding ()
  (setf *cancel-embedding* 1))

(defun uncancel-embedding ()
  (setf *cancel-embedding* 0))

(defparameter *mivi-callback* (lambda (&rest rest) (format t "Got ~A~%" rest)))

(cffi:defcallback mivi-callback :void ((int :int) (vi vectorint))
  (ignore-errors (funcall *mivi-callback* int vi)))

;; You can use the native C objects below if you don't want to have your own representation.
;; It means you have to be careful how you use them.  Probably better to choose some lisp
;; types and perform conversion back and forth then using them directly like is done here.
(defmacro with-external-memory-management ((variable allocate deallocate) &body body)
  `(let (,variable)
     (unwind-protect
          (progn
            (setf ,variable (funcall ,allocate))
            ,@body)
       (when ,variable (funcall ,deallocate ,variable)))))

(defmacro with-graph ((graph) &body body)
  `(with-external-memory-management (,graph 'make_graph 'delete_graph)
     ,@body))

(defmacro with-mivi ((mivi) &body body)
  "A map<int,vector<int>>, used to pass chains into find_embedding.  Only
   exists during the scope of this macro --- do not pass externally."
  `(with-external-memory-management (,mivi 'make_mivi 'delete_mivi)
     ,@body))

(defmacro with-vectorint ((vi) &body body)
  "A vector<int>, an encoding for chains in find_embedding"
  `(with-external-memory-management (,vi 'make_vectorint 'delete_vectorint)
     ,@body))

(defmacro with-vvi ((vvi) &body body)
  "A vector<vector<int>>, the result type for the embedder"
  `(with-external-memory-management (,vvi 'make_vvi 'delete_vvi)
     ,@body))

(define-condition embedding-failed (error) ()) 

(defun vvi-to-lisp-hash (vvi)
  (let ((hash (make-hash-table)))
    (loop :for x :below (vvi_length vvi)
       :do (let* ((vi (vvi_at vvi x))
                  (len (vectorint_length vi))
                  (data (vectorint_to_arrayint vi)))
             (setf (gethash x hash) (loop :for y :below len :collect (mem-aref data :int y)))))
    hash))

(defun find-embedding (source-graph target-graph &key fixed-chains initial-chains restrict-chains (display-callback (callback print-to-console)) (verbosity 0) (num-threads 1))
  (with-vvi (vvi)
    (with-external-memory-management (o (lambda ()
                                          (make-default-optional-parameters :fixed_chains fixed-chains :initial_chains initial-chains
                                                                            :restrict_chains restrict-chains)) 'delete_optional_parameters)
      (with-foreign-slots ((cancelled_callback display_callback verbose fixed_chains initial_chains restrict_chains threads) o
                           (:struct optional-parameters))
        (setf cancelled_callback (callback cancelled_callback))
        (setf display_callback display-callback)
        (setf threads num-threads)
        (setf verbose verbosity))
      (assert (= 1 (findembedding_clib source-graph target-graph o vvi)) nil 'embedding-failed)
      (vvi-to-lisp-hash vvi))))

(defmacro with-triangle-graph ((graph) &body body)
  `(with-graph (,graph)
     (format t "Constructing a triangle graph 0->1->2->0~%")
     (push_back ,graph 0 1)
     (push_back ,graph 1 2)
     (push_back ,graph 2 0)
     ,@body))

(defmacro with-square-graph ((graph) &body body)
  `(with-graph (,graph)
     (format t "Constructing a square graph 0->1->2->3->0~%")
     (push_back ,graph 0 1)
     (push_back ,graph 1 2)
     (push_back ,graph 2 3)
     (push_back ,graph 3 0)
     ,@body))

(defun print-embedding-result (result-hash)
  (loop for x being the hash-keys of result-hash
     do (format t "Node ~A in the source graph is a chain of nodes ~A in the target graph~%" x (gethash x result-hash))))

(defun test-embedding ()
  (with-triangle-graph (source-graph)
    (with-square-graph (target-graph)
      (let ((results (find-embedding source-graph target-graph :verbosity 3)))
        (print-embedding-result results)))))

(defun test-embedding-fixed-chains ()
  ;; Here we try and fix a chain in the target graph so that node 1 is always mapped to (0 1) in the square graph
  (with-triangle-graph (source-graph)
    (with-square-graph (target-graph)
      (with-mivi (fixed-chains)
        (with-vectorint (vi)
          (vectorint_push_back vi 0)
          (vectorint_push_back vi 1)
          (mivi_set fixed-chains 0 vi)
          (let ((results (find-embedding source-graph target-graph :fixed-chains fixed-chains :verbosity 0)))
            (let ((chain-for-node-0 (gethash 0 results)))
              (assert (and (= (length chain-for-node-0) 2) ;; we requested this to be a chain of length two between 0 and 1
                           (find 0 chain-for-node-0)
                           (find 1 chain-for-node-0)))
              (print-embedding-result results))))))))

(defun iota (number &key (start 0))
  (loop :for x :from start
     :repeat number
     :collect x))

(defun generate-c-n-by-m-indices (&key (m-rows 1) (n-cols 1) (cell-size 8) (parent-cx-size 16) (x-offset 0) (y-offset 0))
  "Returns qubits and coupler indices for a given set of chimera params."
  (check-type parent-cx-size (integer 1))
  (check-type x-offset (integer 0))
  (check-type y-offset (integer 0))
  (check-type m-rows (integer 1))
  (check-type n-cols (integer 1))
  (check-type cell-size (integer 1))
  (assert (and 
           (>= parent-cx-size (+ x-offset n-cols))
           (if (not parent-cx-size) t (>= parent-cx-size (+ y-offset m-rows)))))
  (let (qs cos)
    (loop for row from y-offset below (+ y-offset m-rows)
         do
         (loop for col from x-offset below (+ x-offset n-cols)
            do
              (let* ((v (iota (/ cell-size 2) :start (+ (* col cell-size) (* row cell-size parent-cx-size))))
                     (h (mapcar (lambda (x) (+ x (/ cell-size 2))) v)))
                (map nil (lambda (x) (push x qs)) (append h v))
                (map nil (lambda (x) (push x cos))
                     (append
                      (apply #'append (mapcar (lambda (x) (mapcar (lambda (y) (list x y)) h)) v)) ;; intra-C1 couplers
                      (when (< col (+ x-offset n-cols -1)) ;;inter-C1 h couplers
                        (mapcar (lambda (x) (list x (+ x cell-size))) h))
                      (when (< row (+ y-offset m-rows -1)) ;;inter-C1 v couplers
                        (mapcar (lambda (x) (list x (+ x (* cell-size parent-cx-size)))) v)))))))
    (values cos (sort qs #'<))))

(defun make-chimera-graph (N)
  (let ((cos (generate-c-n-by-m-indices :m-rows N :n-cols N :parent-cx-size N)))
    (let ((graph (make_graph)))
      (map nil (lambda (co)
                 (push_back graph (car co) (cadr co))) cos)
      graph)))

(defmacro with-chimera-graph ((var &optional (N 16)) &body body)
  `(with-external-memory-management (,var (lambda () (make-chimera-graph ,N)) 'delete_graph)
     ,@body))
        
(defun make-complete-graph (N)
  (let ((graph (make_graph)))
    (loop for q1 below N do
         (loop for q2 below N do
              (unless (or (= q1 q2) (> q2 q1)) (push_back graph q1 q2))))
    graph))

(defmacro with-complete-graph ((var &optional (N 16)) &body body)
  `(with-external-memory-management (,var (lambda () (make-complete-graph ,N)) 'delete_graph)
     ,@body))

(defun test-complete-graph-embedding-into-chimera ()
  (with-chimera-graph (C16 16)
    (with-complete-graph (K16 16)
      (let ((result (find-embedding K16 C16 :verbosity 0)))
        (print-embedding-result result)
        result))))
