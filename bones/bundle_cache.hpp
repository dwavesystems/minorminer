const uint8_t popcount[256] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 
1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3,
4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3,
3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2,
3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4,
5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

class bundle_cache {
    const size_t dim;
    const size_t linestride;
    const size_t orthstride;
    uint8_t *line_mask;
    bundle_cache(const bundle_cache&) = delete;
    bundle_cache(bundle_cache &&) = delete;

  public:
    ~bundle_cache() {
        if (line_score != nullptr) {
            delete [] line_score;
            line_score = nullptr;
        }
    }
    bundle_cache(const chimera_lines &chim) :
                 dim(chim.dim),
                 linestride((dim*dim+dim)/2),
                 orthstride(dim*linestride),
                 line_score(new uint8_t[2*orthstride]) {
        compute_line_scores(chim);
    }

    size_t ell_score(size_t y0, size_t y1, size_t x0, size_t x1) const {
        return min(get_hline_score(y0, min(x0, x1), max(x0, x1)),
                   get_vline_score(x0, min(y0, y1), max(y0, y1)));
    }

    inline size_t get_hline_score(size_t y, size_t x0, size_t x1) const {
        Assert(y < dim);
        Assert(x0 <= x1);
        Assert(x1 < dim);
        return line_score[orthstride + y*linestride + (x1*x1+x1)/2 + x0];
    }

    inline size_t get_vline_score(size_t x, size_t y0, size_t y1) const {
        Assert(x < dim);
        Assert(y0 <= y1);
        Assert(y1 < dim);
        return line_score[x*linestride + (y1*y1+y1)/2 + y0];
    }

  private:
    void compute_line_scores(const chimera_lines &chim) {
        size_t shoreline[chim.shore];
        size_t scanline[chim.dim];
        for(size_t u = 0; u < 2; u++) {
            size_t p[2];
            size_t &y = p[0]; size_t &z = p[u];
            size_t &x = p[1]; size_t &w = p[1-u];
            for (w = 0; w < chim.dim; w++) {
                for (z = 0; z < chim.dim; z++) {
                    std::fill(scanline, scanline+z+1, 0);
                    for(size_t k = 0; k < chim.shore; k++) {
                        if (chim.has_ext(y, x, u, k)) {
                            // we're on a good qubit connected to a line; bump
                            // up the whole thing starting at shoreline[k]
                            for(size_t z0 = shoreline[k]; z0 <= z; z0++)
                                scanline[z0]++;
                        } else {
                            if(chim.has_qubit(y, x, u, k)) {
                                // start up a new line -- this is where we 
                                // actually initialize shoreline (sorry)
                                shoreline[k] = z;
                                scanline[z]++;
                            } else {
                                // ded
                                shoreline[k] = null_node;
                            }
                        }
                    }
                    //write the current line into cache
                    size_t offset = u*orthstride + w*linestride + (z*z+z)/2;
                    std::copy(scanline, scanline+z+1, line_score+offset);
                }
            }
        }
    }    
};

