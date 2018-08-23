#ifdef __cplusplus
#pragma once
extern "C" {
#endif
  int isCancelled();
  void setCancelled();
  void resetCancelled();
  void displayOutput_clib(const char *msg);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include "../include/util.hpp"

class clib_LocalInteraction : public find_embedding::LocalInteraction {
 public:
  void (*display_callback) (const char *) = &displayOutput_clib;
  int (*cancelled_callback) () = &isCancelled;
 private:
  virtual void displayOutputImpl(const std::string& msg) const {
    display_callback(msg.c_str());
  }
  virtual bool cancelledImpl() const {
    if(cancelled_callback() == 0) {
      return false; 
    } else {
      return true;
    }
  }
};
#else
  typedef struct clib_LocalInteraction clib_LocalInteraction;
#endif

