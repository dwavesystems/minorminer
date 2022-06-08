// Copyright 2021 D-Wave Systems Inc.
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

namespace busclique {

size_t binom(size_t);

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
We do that with the `vert` and `horz` functions:

    bool u = 0;
    size_y y;
    size_x x;
    size_w w0, w1;
    size_z z0, z1;
    w1 = w0 + vert(y);
    z1 = z0 + vert(x);
    u = 1;
    w1 = w0 + horz(x);
    z1 = z0 + horz(y);

Sometimes, we *really* need to mix things up.  There are two ways to convert
a coordinate to a size_t: implicitly through `coordinate_base::operator size_t`,
or explicitly through `coordinate_index()`.  The implicit conversion is
private, to prevent accidental implicit conversions.  The `coordinate_converter`
class is a friend to the coordinate types, and that's the only place we need to
be very careful.  We use the `coordinate_index()` method in very few one-off
instances.  Ideally, it should only be used for indexing into arrays:

    size_y y;
    uint8_t mem[100];
    mem[coordinate_index(y)];

This entire file, and the associated refactor, is a very heavy-handed approach
to fixing a bug (segfaulting non-square Chimera graphs).  However, it reduces a
significant amount of anxiety in developing and extending this library.  In 
small tests, code involving coordinate classes is optimized down to identical 
assembly to the same code using size_t in their place.  We'll see if that's true
in a complex codebase...

Update: Curious.  The performance was rather miserable on my system (gcc 9.3.0);
at around 3x slower than before the change that I hoped would be "zero cost."  I
decided to roll back very carefully; not changing any more code than necessary
to ensure a clean test.  I commented out the coordinate types, and replaced them
with `typedef size_t ...`.  In the initial version, the coordinate types had 
methods, specifically, `coordinate_base.index`, `coordinate_base.horz` and 
`coordinate_base.vert`.  I replaced these with global functions because we can't
attach methods to a size_t.  Sure enough, the performance returned to normal.
For sake of a fuller understanding, I refactored `coordinate_base` to support 
the same syntax, and lo and behold, the performance stayed the same.

Since I'm using a very up-to-date gcc, I am reluctant to put the coordinate
types into the release build.  If I was a docker shepherd, I'd probably spool up
a few dozen instances and benchmark on all the gcc's and clang's and msvc's that
I could get my hands on.  Alas, it is much easier to rely on the fact that CI
tests with debugging enabled, and retain the flag-dependent build since I've
already written it.  A note, I'd be oh-so-slightly more comfortable if `vert`,
`horz`, and `coordinate_index` were macros.  But I take a deep breath and remind
myself what century I'm programming in.

*/


class coordinate_converter;

#if defined CPPDEBUG || defined SAFE_COORDS

//! a tag-struct used to indicate y-coordinates
struct coord_y {};

//! a tag-struct used to indicate x-coordinates
struct coord_x {};

//! a tag-struct used to indicate w-coordinates -- given a qubit, the w
//! coordinate corresponds to an offset perpendicular to the qubit
struct coord_w {};

//! a tag-struct used to indicate z-coordinates -- given a qubit, the z
//! coordinate corresponds to an offset parallel to the qubit
struct coord_z {};

//! This struct defines the relationships between y, x, w, and z in vertical
//! context.  That is, the y coordinate is a parallel offset (z coordinate)
//! from the perspective of a vertical qubit.  Likewise, the x coordinate is a
//! perpendicular offset (w coordinate)
template<typename T> struct vert_lut;
template<> struct vert_lut<coord_y> { typedef coord_z type; };
template<> struct vert_lut<coord_x> { typedef coord_w type; };
template<> struct vert_lut<coord_w> { typedef coord_x type; };
template<> struct vert_lut<coord_z> { typedef coord_y type; };

//! This sruct defines the relationships between y, x, w, and z in horizontal
//! context.  That is, the y coordinate is a perpendicular offset (w coordinate)
//! from the perspective of a vertical qubit.  Likewise, the x coordinate is a
//! parallel offset (z coordinate)
template<typename T> struct horz_lut;
template<> struct horz_lut<coord_y> { typedef coord_w type; };
template<> struct horz_lut<coord_x> { typedef coord_z type; };
template<> struct horz_lut<coord_w> { typedef coord_y type; };
template<> struct horz_lut<coord_z> { typedef coord_x type; };


// Check that vert_lut is idempotent
static_assert(std::is_same<vert_lut<vert_lut<coord_y>::type>::type, coord_y>::value, "idempotence check failed");
static_assert(std::is_same<vert_lut<vert_lut<coord_x>::type>::type, coord_x>::value, "idempotence check failed");
static_assert(std::is_same<vert_lut<vert_lut<coord_w>::type>::type, coord_w>::value, "idempotence check failed");
static_assert(std::is_same<vert_lut<vert_lut<coord_z>::type>::type, coord_z>::value, "idempotence check failed");

// Check that horz_lut is idempotent
static_assert(std::is_same<horz_lut<horz_lut<coord_y>::type>::type, coord_y>::value, "idempotence check failed");
static_assert(std::is_same<horz_lut<horz_lut<coord_x>::type>::type, coord_x>::value, "idempotence check failed");
static_assert(std::is_same<horz_lut<horz_lut<coord_w>::type>::type, coord_w>::value, "idempotence check failed");
static_assert(std::is_same<horz_lut<horz_lut<coord_z>::type>::type, coord_z>::value, "idempotence check failed");

// Check that vert_lut and horz_lut are orthogonal
static_assert(std::is_same<vert_lut<horz_lut<coord_y>::type>::type, coord_x>::value, "orthogonality check failed");
static_assert(std::is_same<vert_lut<horz_lut<coord_x>::type>::type, coord_y>::value, "orthogonality check failed");
static_assert(std::is_same<vert_lut<horz_lut<coord_w>::type>::type, coord_z>::value, "orthogonality check failed");
static_assert(std::is_same<vert_lut<horz_lut<coord_z>::type>::type, coord_w>::value, "orthogonality check failed");

static_assert(std::is_same<horz_lut<vert_lut<coord_y>::type>::type, coord_x>::value, "orthogonality check failed");
static_assert(std::is_same<horz_lut<vert_lut<coord_x>::type>::type, coord_y>::value, "orthogonality check failed");
static_assert(std::is_same<horz_lut<vert_lut<coord_w>::type>::type, coord_z>::value, "orthogonality check failed");
static_assert(std::is_same<horz_lut<vert_lut<coord_z>::type>::type, coord_w>::value, "orthogonality check failed");

//! This template class represents a generic coordinate type.  The intention is
//! that the template parameter T is one of coord_w, coord_z, coord_y, coord_x.
//! Class instances are meant to behave similar to a size_t; but similar to a
//! system of units, we disallow implicit conversions between coordinates of
//! different kinds.
template<typename T>
class coordinate_base;

//! This function is coodinate_base's only friend.  It performs an explicit 
//! conversion from a coordinate_base<T> to a size_t.  Ideally, this will be 
//! used very rarely outside of the coordinate_converter class.
template<typename T>
size_t coordinate_index(coordinate_base<T>);

template<typename T>
class coordinate_base {
    typedef coordinate_base<T> cb;
    friend size_t coordinate_index<T>(cb);
    size_t v;
  public:
    constexpr coordinate_base() {}
    constexpr coordinate_base(size_t v) : v(v) {}
    coordinate_base(const coordinate_base<T> &c) : v(c.v) {}
    cb operator++(int) { cb t = *this; v++; return t; }
    cb operator--(int) { cb t = *this; v--; return t; }
    cb &operator++() { v++; return *this; }
    cb &operator--() { v--; return *this; }
    cb operator+(cb a) const { return v+a.v; }
    cb operator-(cb a) const { return v-a.v; }
    cb operator/(cb a) const { return v/a.v; }
    cb operator*(cb a) const { return v*a.v; }
    bool operator<(cb a) const { return v<a.v; }
    bool operator>(cb a) const { return v>a.v; }
    bool operator<=(cb a) const { return v<=a.v; }
    bool operator>=(cb a) const { return v>=a.v; }
    bool operator==(cb a) const { return v==a.v; }
};

//! 
//! a class to protect y-coordinates as a unit type
typedef coordinate_base<coord_y> size_y;

//! a class to protect x-coordinates as a unit type
typedef coordinate_base<coord_x> size_x;

//! a class to protect w-coordinates as a unit type -- given a qubit, the w 
//! coordinate corresponds to an offset perpendicular to the qubit
typedef coordinate_base<coord_w> size_w;

//! a class to protect z-coordinates as a unit type -- given a qubit, the z
//! coordinate corresponds to an offset parallel to the qubit
typedef coordinate_base<coord_z> size_z;

template<typename T>
size_t coordinate_index(coordinate_base<T> c) {
    return c.v;
}

//! This purely for convenience, to make implementations of coordinate_converter
//! methods cleaner.
size_t coordinate_index(size_t c) {
    return c;
}


//! This function translates between relative coordinates (w, z) and matrix
//! coordinates (y, x) in the context of a vertical qubit address.  
template<typename T>
coordinate_base<typename vert_lut<T>::type> vert(const coordinate_base<T> c) {
    return coordinate_index(c);
}

template<typename T>
coordinate_base<typename horz_lut<T>::type> horz(const coordinate_base<T> c) {
    return coordinate_index(c);
}

template<typename T>
size_t binom(coordinate_base<T> c) {
    return binom(coordinate_index(c));
}

template<typename T>
std::ostream& operator<< (std::ostream& out, coordinate_base<T> v) {
    return out << coordinate_index(v);
}

#else

// if neither CPPDEBUG nor COORDINATE_TYPES is set, we take the safeties off.
// All coordinates are just size_t's.

typedef size_t size_y;
typedef size_t size_x;
typedef size_t size_w;
typedef size_t size_z;

inline const size_t &coordinate_index(const size_t &c) { return c; }
inline const size_t &vert(const size_t &c) { return c; }
inline const size_t &horz(const size_t &c) { return c; }

#endif


//! user-defined literal for size_y
constexpr size_y operator""_y(unsigned long long int v) { return static_cast<size_t>(v); }

//! user-defined literal for size_x
constexpr size_x operator""_x(unsigned long long int v) { return static_cast<size_t>(v); }

//! user-defined literal for size_w
constexpr size_w operator""_w(unsigned long long int v) { return static_cast<size_t>(v); }

//! user-defined literal for size_z
constexpr size_z operator""_z(unsigned long long int v) { return static_cast<size_t>(v); }

// These are for convenience, and probably belong in their own file...

//! user-defined literal for uint64_t
constexpr uint64_t operator""_u64(unsigned long long int v) { return static_cast<uint64_t>(v); }

//! user-defined literal for uint32_t
constexpr uint32_t operator""_u32(unsigned long long int v) { return static_cast<uint32_t>(v); }

//! user-defined literal for uint16_t
constexpr uint16_t operator""_u16(unsigned long long int v) { return static_cast<uint16_t>(v); }

//! user-defined literal for uint8_t
constexpr uint8_t operator""_u8(unsigned long long int v) { return static_cast<uint8_t>(v); }


//! An arena to perform arithmetic that does not respect the unit system defined
//! in coordinate_base<T>.  This class is not a friend of coordinate_base<T> so
//! so that we can do all of our bounds-checking here, with full type-safety.
//! The generic pattern is that bounds-checking is done inside a public method,
//! respecting the unit systems, and then we explicitly convert coordinates and
//! dimensions to size_t, to be combined in a private method with the _impl
//! suffix.
class coordinate_converter {

  private:
    //! a convenience shortener to reduce line lengths
    template<typename T>
    static size_t coord(T c) { return coordinate_index(c); }
    
  private:
    //! private implementation of cell_index
    static size_t cell_index_impl(size_t y, size_t x, size_t u, size_t dim_y, size_t dim_x) {
        return u + 2*(x + dim_x*y);
    }
  public:
  
    //! This computes compute the index of A[u][y][x] in an array A of dimension
    //! [2][dim_y][dim_x]    
    static size_t cell_index(size_y y, size_x x, bool u, size_y dim_y, size_x dim_x) {
        minorminer_assert(y < dim_y);
        minorminer_assert(x < dim_x);
        return cell_index_impl(coord(y), coord(x), u, coord(dim_y), coord(dim_x));
    }
    
    //! This computes compute the index of A[u][y][x] in an array A of dimension
    //! [2][dim_y][dim_x]; where y and x are determined from w and z, depending
    //! on the orientation u
    static size_t cell_index(bool u, size_w w, size_z z, size_y dim_y, size_x dim_x) {
        if (u) {
            return cell_index(horz(w), horz(z), u, dim_y, dim_x);
        } else {
            return cell_index(vert(z), vert(w), u, dim_y, dim_x);
        }
    }

  private:
    //! private implementation of chimera_linear
    static size_t chimera_linear_impl(size_t y, size_t x, size_t u, size_t k, size_t dim_y, size_t dim_x, size_t shore) {
        return k + shore*(u + 2*(x + dim_x*y));
    }
  public:
    //! given a chimera coordinate (y, x, u, k) with chimera dimensions 
    //! (m, n, t) = (dim_y, dim_x, shore), compute the linear index.
    template <typename shore_t>
    static size_t chimera_linear(size_y y, size_x x, bool u, shore_t k, size_y dim_y, size_x dim_x, shore_t shore) {
        minorminer_assert(y < dim_y);
        minorminer_assert(x < dim_x);
        minorminer_assert(k < shore);
        return chimera_linear_impl(coord(y), coord(x), u, k, coord(dim_y), coord(dim_x), shore);
    }

    //! given a linear index q0 in a chimera graph with dimensions 
    //! (m, n, t) = (dim_y, dim_x, shore), compute the coordinate (y, x, u, k)
    template <typename shore_t>
    static void linear_chimera(size_t q0, size_y &y, size_x &x, bool &u, shore_t &k, size_y dim_y, size_x dim_x, shore_t shore) {
        size_t q = q0;
        k = q%coord(shore);
        q = q/coord(shore);
        u = q%2;
        q = q/2;
        x = q%coord(dim_x);
        q = q/coord(dim_x);
        minorminer_assert(q < coord(dim_y));
        y = q;
        minorminer_assert(q0 == chimera_linear(y, x, u, k, dim_y, dim_x, shore));
    }
    
  private:
    //! private implementation of linemajor_linear
    static size_t linemajor_linear_impl(size_t u, size_t w, size_t k, size_t z, size_t dim_w, size_t shore, size_t dim_z) {
        return z + dim_z*(k + shore*(w + dim_w*u));
    }
  public:
    //! Pegasus and Zephyr coordinates exist in relative coordinate systems. We
    //! note that our implementation of Zephyr coordinates differs from the one
    //! used other places -- we've merged the k and j indices into one. The name
    //! 'linemajor' indicates that the (u, w, k) parameters specify a line of
    //! colinear qubits.  This function is used to pack a Zephyr or Pegasus 
    //! coordinate address (u, w, k, z) into a linear index.  The interpretation
    //! of dim_w, shore, and dim_z is determined by the relevant topology, and
    //! we expect callers to know what they're doing.
    //! Specifically, for pegasus_graph(m), dim_w = 6*m, shore = 2, dim_z = m-1
    //! and for zephyr_graph(m), dim_w = 2*m+1, shore=2*t, dim_z = m
    template <typename shore_t>
    static size_t linemajor_linear(bool u, size_w w, shore_t k, size_z z, size_w dim_w, shore_t shore, size_z dim_z) {
        minorminer_assert(w < dim_w);
        minorminer_assert(k < shore);
        minorminer_assert(z < dim_z);
        return linemajor_linear_impl(u, coord(w), k, coord(z), coord(dim_w), shore, coord(dim_z));
    }
    
    //! Pegasus and Zephyr coordinates exist in relative coordinate systems. We
    //! note that our implementation of Zephyr coordinates differs from the one
    //! used other places -- we've merged the k and j indices into one. The name
    //! 'linemajor' indicates that the (u, w, k) parameters specify a line of
    //! colinear qubits.  This function is used to unpack a Zephyr or Pegasus 
    //! linear index to a coordinate address (u, w, k, z).  The interpretation
    //! of dim_w, shore, and dim_z is determined by the relevant topology, and
    //! we expect callers to know what they're doing.
    //! Specifically, for pegasus_graph(m), dim_w = 6*m, shore = 2, dim_z = m-1
    //! and for zephyr_graph(m), dim_w = 2*m+1, shore=2*t, dim_z = m
    template <typename shore_t>
    static void linear_linemajor(size_t q0, bool &u, size_w &w, shore_t &k, size_z &z, size_w dim_w, shore_t shore, size_z dim_z) {
        size_t q = q0;
        z = q%coord(dim_z);
        q = q/coord(dim_z);
        k = q%shore;
        q = q/shore;
        w = q%coord(dim_w);
        q = q/coord(dim_w);
        minorminer_assert(q < 2);
        u = q;
        minorminer_assert(q0 == linemajor_linear(u, w, k, z, dim_w, shore, dim_z));
    }

  public:
    template<typename T, typename ... Args>
    static size_t product(T t) { return coord(t); }

    //! explicitly convert all arguments into a size_t and return the product of 
    //! them all.
    template<typename T, typename ... Args>
    static size_t product(T t, Args ...args) {
        return coord(t) * product(args...);
    }

    template<typename T, typename ... Args>
    static size_t sum(T t) { return coord(t); }

    //! explicitly convert all arguments into a size_t and return the product of 
    //! them all.
    template<typename T, typename ... Args>
    static size_t sum(T t, Args ...args) {
        return coord(t) + sum(args...);
    }

    //! explicitly convert both arguments into a size_t and return the minimum
    template<typename S, typename T>
    static size_t min(S s, T t) { return std::min(coord(s), coord(t)); }

    //! explicitly convert both arguments into a size_t and return the maximum
    template<typename S, typename T>
    static size_t max(S s, T t) { return std::max(coord(s), coord(t)); }

  private:
    //! private implementation of grid_index
    static size_t grid_index_impl(size_t y, size_t x, size_t dim_y, size_t dim_x) {
        return x + dim_x*y;
    }
  public:
    //! Implements addressing into a 2-dimensional array T[dim_y][dim_x]
    static size_t grid_index(size_y y, size_x x, size_y dim_y, size_x dim_x) {
        minorminer_assert(y < dim_y);
        minorminer_assert(x < dim_x);
        return grid_index_impl(coord(y), coord(x), coord(dim_y), coord(dim_x));
    }


  private:
    //! private implementation of bundle_cache_index
    static size_t bundle_cache_index_impl(size_t u, size_t w, size_t z0, size_t z1, size_t stride_u, size_t stride_w) {
        minorminer_assert(z0 <= z1);
        minorminer_assert(binom(z1) + z0 < stride_w);
        return u*stride_u + w*stride_w + binom(z1) + z0;
    }

  public:
    //! This addressing scheme is wikkid complicated.  The majority of the logic
    //! is contained in bundle_cache.hpp.  We store line masks (that is, sets of
    //! k-indices) corresponding to the line segment (u, w, z0, z1).  That is,
    //! if the bit i is set in line_mask[bundle_cache_index(u, w, z0, z1)], then
    //! line of qubits [(u, w, i, z) for z in range(z0, z1+1)] and the necessary
    //! external couplers are present in the linemajor representation of the
    //! topology.  We assume that the caller knows what it's doing with stride_u
    //! and stride_w.  We do belts-and-suspenders assertion testing in 
    //! bundle_cache::compute_line_masks for added confidence that the caller
    //! actually knows what it's doing.
    static size_t bundle_cache_index(bool u, size_w w, size_z z0, size_z z1, size_t stride_u, size_t stride_w) {
        return bundle_cache_index_impl(u, coord(w), coord(z0), coord(z1), stride_u, stride_w);
    }

};

}
