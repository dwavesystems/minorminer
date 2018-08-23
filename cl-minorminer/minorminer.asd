;;;; -*- Mode: Lisp; Syntax: ANSI-Common-Lisp; Base: 10 -*-
(defpackage #:minorminer-asd
  (:use :cl :asdf))

(in-package :minorminer-asd)

(defsystem minorminer
  :name "minorminer"
  :version "0.0.1"
  :maintainer "A. J. Berkley (ajb@dwavesys.com)"
  :author "A. J. Berkley (ajb@dwavesys.com)"
  :license "Apache v2.0 Jan 2004"
  :description "Common Lisp wrapper around D-Wave Minor Miner (a tool for graph embedding) C++ library"
  :depends-on (:cffi)
  :components ((:file "minorminer")))



  
