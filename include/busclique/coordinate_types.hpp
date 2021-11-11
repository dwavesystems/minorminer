#pragma once

namespace busclique {

/*
Coordinate arithmetic is absolutely fraught with challenges.  In order to 
reduce the possibility of bugs, we've introduced an extremely pedantic system
of coordinate types.  The busclique algorithms use two kinds of coordinate 
systems:

  * matrix coordinates (y, x) which are featured in the chimera coordinate
    system (y, x, u, k) -- where u indicates the 'orientation' of a qubit,
    either vertical(u=0) or horizontal(u=1) and k indicates the 'shore index'
    of a qubit (irrelevant here)

  * relative coordinates (w, z) which are features in the pegasus and zephyr
    coordinate systems (u, w, k[, j], z) -- where u is the orientation as
    above and k and j are similarly irrelevant.

The relationship between matrix and relative coordinates is that

  * (y, x, u=0) is equivalent to (z=y, w=x, u=0), and

  * (y, x, u=1) is equivalent to (w=y, z=x, u=1)

We have encoded this equivalence in four types, size_y, size_x, size_w, and
size_z.  Each type supports arithmetic with itself, but forbids confusion.  For
example,

    size_y y0, y1;
    size_x x0, x0;
    (y0 + y1*y0 + y1)/(y1-y0) // cool
    y0 + x0;                  // compile error!

But, we need to convert between matrix and relative coordinates frequently!
We do that with the `vert` and `horz` methods:

    bool u = 0;
    size_y y;
    size_x x;
    size_w w0, w1;
    size_z z0, z1;
    w1 = w0 + y.vert();
    z1 = w0 + x.vert();

Sometimes, we *really* need to mix things up.  There are two ways to convert
a coordinate to a size_t: implicitly through `coordinate_base::operator size_t`,
or explicitly through `coordinate_base::index()`.  The implicit conversion is
private, to prevent accidental implicit conversions.  The `coordinate_converter`
class is a friend to the coordinate types, and that's the only place we need to
be extremely careful.  The `index()` method is used in very few one-off
instances.  Ideally, it should only be used for indexing into arrays:

    size_y y;
    uint8_t mem[100];
    mem[y.index()];

This entire file, and the associated refactor, is a very heavy-handed approach
to fixing a bug (segfaulting non-square Chimera graphs).  However, it reduces a
significant amount of anxiety in developing and extending this library.  In 
small tests, code involving coordinate classes is optimized down to identical 
assembly to the same code using size_t in their place.  We'll see if that's true
in a complex codebase...

*/


class coordinate_converter;

class vert_t {}; constexpr vert_t vert_tag = {};
class horz_t {}; constexpr horz_t horz_tag = {};

template<char a, char b> struct same_c : std::false_type {};
template<char a> struct same_c<a, a> : std::true_type {};


template<char i, typename orientation_tag> class toggle_lut;

template<> class toggle_lut<'y', vert_t> { public: static constexpr char value = 'z'; };
template<> class toggle_lut<'y', horz_t> { public: static constexpr char value = 'w'; };
template<> class toggle_lut<'x', vert_t> { public: static constexpr char value = 'w'; };
template<> class toggle_lut<'x', horz_t> { public: static constexpr char value = 'z'; };

template<> class toggle_lut<'w', vert_t> { public: static constexpr char value = 'x'; };
template<> class toggle_lut<'w', horz_t> { public: static constexpr char value = 'y'; };
template<> class toggle_lut<'z', vert_t> { public: static constexpr char value = 'y'; };
template<> class toggle_lut<'z', horz_t> { public: static constexpr char value = 'x'; };

static_assert(same_c<toggle_lut<toggle_lut<'y', vert_t>::value, vert_t>::value, 'y'>::value, "consistency check failed");
static_assert(same_c<toggle_lut<toggle_lut<'x', vert_t>::value, vert_t>::value, 'x'>::value, "consistency check failed");
static_assert(same_c<toggle_lut<toggle_lut<'w', vert_t>::value, vert_t>::value, 'w'>::value, "consistency check failed");
static_assert(same_c<toggle_lut<toggle_lut<'z', vert_t>::value, vert_t>::value, 'z'>::value, "consistency check failed");

template<char i>
class coordinate_base {
    friend class coordinate_converter;
    typedef coordinate_base<i> cb;
    size_t v;
    operator size_t() { return v; }
  public:
  
    constexpr coordinate_base() {}
    coordinate_base(size_t v) : v(v) {}
    coordinate_base(const coordinate_base<i> &c) : v(c.v) {}
    cb operator++(int) { cb t = *this; v++; return t; }
    cb operator--(int) { cb t = *this; v--; return t; }
    cb &operator++() { v++; return *this; }
    cb &operator--() { v--; return *this; }
    cb operator+(cb a) const { return v+a; }
    cb operator-(cb a) const { return v-a; }
    cb operator/(cb a) const { return v/a; }
    cb operator*(cb a) const { return v*a; }
    cb operator+(unsigned int a) const { return v+a; }
    cb operator-(unsigned int a) const { return v-a; }
    cb operator/(unsigned int a) const { return v/a; }
    cb operator*(unsigned int a) const { return v*a; }
    bool operator<(cb a) const { return v<a; }
    bool operator>(cb a) const { return v>a; }
    bool operator<=(cb a) const { return v<=a; }
    bool operator>=(cb a) const { return v>=a; }
    bool operator==(cb a) const { return v==a; }
    size_t index() const { return v; }
    coordinate_base<toggle_lut<i, horz_t>::value> horz() const {return v;}
    coordinate_base<toggle_lut<i, vert_t>::value> vert() const {return v;}
};

typedef coordinate_base<'y'> size_y;
typedef coordinate_base<'x'> size_x;
typedef coordinate_base<'w'> size_w;
typedef coordinate_base<'z'> size_z;

class coordinate_converter {
  private:
    static size_t cell_addr_impl(size_t y, size_t x, size_t u, size_t dim_y, size_t dim_x) {
        return u + 2*(x + dim_x*y);
    }
  public:
    static size_t cell_addr(size_y y, size_x x, bool u, size_y dim_y, size_x dim_x) {
        return cell_addr_impl(y, x, u, dim_y, dim_x);
    }
    static size_t cell_addr(bool u, size_w w, size_z z, size_y dim_y, size_x dim_x) {
        if (u) {
            return cell_addr(w.horz(), z.horz(), u, dim_y, dim_x);
        } else {
            return cell_addr(z.vert(), w.vert(), u, dim_y, dim_x);
        }
    }

  private:
    static size_t chimera_linear_impl(size_t y, size_t x, size_t u, size_t k, size_t dim_y, size_t dim_x, size_t shore) {
        return k + shore*(u + 2*(x + dim_x*y));
    }
  public:
    template <typename shore_t>
    static size_t chimera_linear(size_y y, size_x x, bool u, shore_t k, size_y dim_y, size_x dim_x, shore_t shore) {
        minorminer_assert(y < dim_y);
        minorminer_assert(x < dim_x);
        minorminer_assert(k < shore);
        return chimera_linear_impl(y, x, u, k, dim_y, dim_x, shore);
    }

    template <typename shore_t>
    static void linear_chimera(size_t q, size_y &y, size_x &x, bool &u, shore_t &k, size_y dim_y, size_x dim_x, shore_t shore) {
        k = q%size_t(shore);
        q = q/size_t(shore);
        u = q%2;
        q = q/2;
        x = q%size_t(dim_x);
        q = q/size_t(dim_x);
        minorminer_assert(q < size_t(dim_y));
        y = q;
    }
    
  private:
    static size_t linemajor_linear_impl(size_t u, size_t w, size_t k, size_t z, size_t dim_w, size_t shore, size_t dim_z) {
        return z + dim_z*(k + shore*(w + dim_w*u));
    }
  public:
    template <typename shore_t>
    static size_t linemajor_linear(bool u, size_w w, shore_t k, size_z z, size_w dim_w, shore_t shore, size_z dim_z) {
        minorminer_assert(w < dim_w);
        minorminer_assert(k < shore);
        minorminer_assert(z < dim_z);
        return linemajor_linear_impl(u, w, k, z, dim_w, shore, dim_z);
    }

    template <typename shore_t>
    static void linear_linemajor(size_t q, bool &u, size_w &w, shore_t &k, size_z &z, size_w dim_w, shore_t shore, size_z dim_z) {
        z = q%size_t(dim_z);
        q = q/size_t(dim_z);
        k = q%size_t(shore);
        q = q/size_t(shore);
        w = q%size_t(dim_w);
        q = q/size_t(dim_w);
        minorminer_assert(q < 2);
        u = q;
    }

  public:
    template<typename T, typename ... Args>
    static size_t product(T t) { return t; }

    template<typename T, typename ... Args>
    static size_t product(T t, Args ...args) {
        return size_t(t) * product(args...);
    }

    template<typename T, typename ... Args>
    static size_t sum(T t) { return t; }

    template<typename T, typename ... Args>
    static size_t sum(T t, Args ...args) {
        return size_t(t) + sum(args...);
    }

    template<typename S, typename T>
    static size_t min(S s, T t) { return std::min(size_t(s), size_t(t)); }

    template<typename S, typename T>
    static size_t max(S s, T t) { return std::max(size_t(s), size_t(t)); }

  private:
    static size_t index_impl(size_t y, size_t x, size_t dim_y, size_t dim_x) {
        return x + dim_x*y;
    }
    static size_t index_impl(size_t y, size_t x, size_t u, size_t dim_y, size_t dim_x) {
        return u + 2*(x + dim_x*y);
    }
  public:
    static size_t index(size_y y, size_x x, size_y dim_y, size_x dim_x) {
        minorminer_assert(y < dim_y);
        minorminer_assert(x < dim_x);
        return index_impl(y, x, dim_y, dim_x);
    }

    static size_t index(size_y y, size_x x, bool u, size_y dim_y, size_x dim_x) {
        minorminer_assert(y < dim_y);
        minorminer_assert(x < dim_x);
        return index_impl(y, x, u, dim_y, dim_x);
    }


};


}
