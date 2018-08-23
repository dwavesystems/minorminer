#include "stl_clib.h"
#include <stdio.h>
#include <iostream>
/* A thin c library around some C++ standard library tools */
/* Manual templating... wow */
int vectorint_length(vectorint* v) {
  return v->size();
}

void vectorint_push_back(vectorint* v, int i) {
  v->push_back(i);
}

int* vectorint_to_arrayint(vectorint* v) {
  return v->data();
}

void vectorint_print(vectorint* v) {
  int i;
  for(i=0;i<vectorint_length(v);i++) {
    printf("%i ",v->at(i));
  }
  if(i!=0) { printf("\n");};
}

void vvi_print(vvi* v) {
  int i;
  for(i=0;i<vvi_length(v);i++) {
    printf("Element %i, length %li\n",i,v->at(i).size());
    vectorint_print(&v->at(i));
  }
}

vectorint* make_vectorint() {
  return new vectorint();
}

void delete_vectorint(vectorint *i) {
  delete i;
}

int vvi_length(vvi* v) {
  return v->size();
}

void vvi_push_back(vvi* v, vectorint* i) {
  v->push_back(*i);
}

vectorint* vvi_at(vvi* v, int i) {
  return &v->at(i);
}

vvi* make_vvi() {
  return new vvi();
}

void delete_vvi(vvi *i) {
  delete i;
}

mivi* make_mivi() {
  return new mivi();
}

void mivi_set(mivi* mivi, int i, vectorint* vi) {
  mivi->insert(std::pair<int,vectorint>(i, *vi));
}

extern void mivi_map(mivi *mivi, void (*callback)(int, vectorint*)) {
  for(auto& x: *mivi) {
    callback(x.first, (vectorint *)&x.second);
  };
}

void delete_mivi(mivi *mivi) {
  delete mivi;
}
