
    void invalidate(const chimera_lines &chim, size_t y0, size_t x0) {
        size_t scanline[chim.dim];
        size_t shoreline[chim.shore];
        for(size_t u = 0, w = y0; u < 2; w = x0, u++) {
            size_t p[2];
            p[1-u] = w;
            size_t &y = p[0]; size_t &z = p[u];
            size_t &x = p[1];
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
