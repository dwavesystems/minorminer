#pragma once

#ifdef __cplusplus
  #include <vector>
  #include <map>
  typedef std::vector<int> vectorint;
  typedef std::vector<vectorint> vvi;
  typedef std::map<int,vectorint> mivi;
#else
  typedef struct vectorint vectorint;
  typedef struct vvi vvi;
  typedef struct mivi mivi;
#endif

#ifdef __cplusplus
extern "C" {
#endif
  vectorint* make_vectorint();
  int vectorint_length(vectorint*);
  int *vectorint_to_arrayint(vectorint*);
  void vectorint_push_back(vectorint*, int);
  void vectorint_print(vectorint*);
  void delete_vectorint(vectorint*);

  vvi* make_vvi();
  int vvi_length(vvi*);
  vectorint* vvi_at(vvi*, int);
  void vvi_push_back(vvi*, vectorint*);
  void vvi_print(vvi*);
  void delete_vvi(vvi*);

  mivi* make_mivi();
  void mivi_set(mivi*, int, vectorint*);
  void mivi_unset(mivi*, int);
  void mivi_map(mivi *, void (*callback)(int, vectorint*));
  void delete_mivi(mivi*);
#ifdef __cplusplus
}
#endif
