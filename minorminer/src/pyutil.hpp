#pragma once
#include "../../include/util.hpp"

namespace {

class LocalInteractionPython : public find_embedding::LocalInteraction {
  public:
    virtual ~LocalInteractionPython() {}

  private:
    virtual void displayOutputImpl(const std::string &msg) const { PySys_WriteStdout("%s", msg.c_str()); }

    virtual bool cancelledImpl() const {
        if (PyErr_CheckSignals()) {
            PyErr_Clear();
            return true;
        }

        return false;
    }
};

void handle_exceptions() {
    try {
        throw;
    } catch (const find_embedding::TimeoutException &e) {
        PyErr_SetString(PyExc_TimeoutError, e.what());
    } catch (const find_embedding::ProblemCancelledException &e) {
        PyErr_SetString(PyExc_InterruptedError, e.what());
    } catch (const find_embedding::CorruptParametersException &e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    } catch (const find_embedding::BadInitializationException &e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const find_embedding::CorruptEmbeddingException &e) {
        PyErr_SetString(PyExc_AssertionError, e.what());
    }
}

}  // namespace
