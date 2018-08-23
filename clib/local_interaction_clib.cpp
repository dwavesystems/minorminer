#include "local_interaction_clib.h"
#include <stdio.h>

void displayOutput_clib(const char *msg) {
  printf("%s\n",msg);
}

int cancelledStatus = 0;

/* default implementation is a global status flag which one sets and resets,
   use the cancelled_callback if you want to do more */
int isCancelled() {
  return cancelledStatus;
}

void setCancelled() {
  cancelledStatus = 1;
}

void resetCancelled() {
  cancelledStatus = 0;
}
