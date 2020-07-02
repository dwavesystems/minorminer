// Copyright 2017 - 2020 D-Wave Systems Inc.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#pragma once
#include "../../include/find_embedding/util.hpp"

namespace {

class LocalInteractionPython : public find_embedding::LocalInteraction {
  public:
    virtual ~LocalInteractionPython() {}

  private:
    virtual void displayOutputImpl(int, const std::string &msg) const { PySys_WriteStdout("%s", msg.c_str()); }

    virtual void displayErrorImpl(int, const std::string &msg) const { PySys_WriteStderr("%s", msg.c_str()); }

    virtual bool cancelledImpl() const {
        bool cancelled = static_cast<bool>(PyErr_CheckSignals());
        if (cancelled) PyErr_Clear();
        return cancelled;
    }
};

typedef void (*cython_callback)(void *log_fn, int loglevel, const std::string &msg);
class LocalInteractionLogger : public find_embedding::LocalInteraction {
    cython_callback pycallback;
    void *log_fn;

  public:
    LocalInteractionLogger(cython_callback cb, void *fn) : pycallback(cb), log_fn(fn) { Py_INCREF(log_fn); }
    virtual ~LocalInteractionLogger() { Py_DECREF(log_fn); }

  private:
    virtual void displayOutputImpl(int loglevel, const std::string &msg) const {
        pycallback(log_fn, loglevel, msg.c_str());
    }

    virtual void displayErrorImpl(int loglevel, const std::string &msg) const {
        pycallback(log_fn, loglevel, msg.c_str());
    }

    virtual bool cancelledImpl() const {
        bool cancelled = static_cast<bool>(PyErr_CheckSignals());
        if (cancelled) PyErr_Clear();
        return cancelled;
    }
};

void handle_exceptions() {
    try {
        throw;
    } catch (const find_embedding::TimeoutException &e) {
        PyErr_SetString(PyExc_TimeoutError, e.what());
    } catch (const find_embedding::ProblemCancelledException &e) {
        PyErr_SetString(PyExc_KeyboardInterrupt, e.what());
    } catch (const find_embedding::CorruptParametersException &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const find_embedding::BadInitializationException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const find_embedding::CorruptEmbeddingException &e) {
        PyErr_SetString(PyExc_AssertionError, e.what());
    }
}

}  // namespace
