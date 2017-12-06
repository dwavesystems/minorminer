#pragma once
namespace find_embedding {

#ifdef CPPDEBUG
#define DIAGNOSE(X) diagnostic(X);
#else
#define DIAGNOSE(X)
#endif

class chain {
  private:
    unordered_map<int, pair<int, int>> data;
    unordered_map<int, int> links;
    const int label;

  public:
    chain(int l) : data(), links(), label(l) {}

    chain &operator=(const vector<int> &c) {
        clear();
        for (auto &q : c) data.emplace(q, pair<int, int>(q, 1));
        DIAGNOSE("operator=vector");
        return *this;
    }

    chain &operator=(const chain &c) {
        clear();
        data = c.data;
        links = c.links;
        DIAGNOSE("operator=chain");
        return *this;
    }

    // number of qubits in chain
    inline int size() const { return data.size(); }

    // returns 0 if `q` is not contained in `this`, 1 otherwise
    inline int count(const int q) const { return data.count(q); }

    // get the qubit, in `this`, which links `this` to the chain of x
    //(if x==label, interpret the linking qubit as the chain's root)
    inline int get_link(const int x) const {
        auto z = links.find(x);
        if (z == links.end())
            return -1;
        else
            return (*z).second;
    }

    // set the qubit, in `this`, which links `this` to the chain of x
    //(if x==label, interpret the linking qubit as the chain's root)
    inline void set_link(const int x, const int q) {
        minorminer_assert(get_link(x) == -1);
        minorminer_assert(count(q) == 1);
        links[x] = q;

        retrieve(q).second++;
        DIAGNOSE("set_link");
    }

    // discard the linking information for `x`
    inline int drop_link(const int x) {
        int q = -1;
        auto z = links.find(x);
        if (z != links.end()) {
            q = (*z).second;
            minorminer_assert(count(q) == 1);
            retrieve(q).second--;
            links.erase(z);
        }
        DIAGNOSE("drop_link");
        return q;
    }

    // insert the qubit `q` into `this`, and set `q` to be the root
    //(represented as the linking qubit for `label`)
    inline void set_root(const int q) {
        minorminer_assert(data.size() == 0);
        minorminer_assert(links.size() == 0);
        links.emplace(label, q);
        data.emplace(q, pair<int, int>(q, 2));
        DIAGNOSE("set_root");
    }

    // empty this data structure
    inline void clear() {
        data.clear();
        links.clear();
        DIAGNOSE("clear");
    }

    // add the qubit `q` as a leaf, with `parent` as its parent
    inline void add_leaf(const int q, const int parent) {
        minorminer_assert(data.count(q) == 0);
        minorminer_assert(data.count(parent) == 1);
        data.emplace(q, pair<int, int>(parent, 0));
        retrieve(parent).second++;
        DIAGNOSE("add leaf");
    }

    // try to delete the qubit `q` from this chain, and keep
    // deleting until no more qubits are free to be deleted.
    // return the first ancestor which cannot be deleted
    inline int trim_branch(int q) {
        minorminer_assert(data.count(q) == 1);
        auto z = data.find(q);
        auto p = (*z).second;
        while (p.second == 0) {
            data.erase(z);
            z = data.find(p.first);
            p = (*z).second;
            p.second--;
            q = p.first;
            minorminer_assert(data.count(q) == 1);
        }
        DIAGNOSE("trim branch");
        return q;
    }

    // try to delete the qubit `q` from this chain.  if `q`
    // cannot be deleted, return it; otherwise return its parent
    inline int trim_leaf(int q) {
        minorminer_assert(data.count(q) == 1);
        auto z = data.find(q);
        auto p = (*z).second;
        if (p.second == 0) {
            retrieve(p.first).second--;
            data.erase(z);
            q = p.first;
        }
        DIAGNOSE("trim leaf");
        return q;
    }

    // the parent of `q` in this chain -- which might be `q` but
    // otherwise cycles should be impossible
    inline int parent(const int q) const {
        minorminer_assert(data.count(q) == 1);
        return fetch(q).first;
    }

    // return the number of references that `this` makes to the qubit
    //`q` -- where a "reference" is an occurrence of `q` as a parent
    // or an occurrence of `q` as a linking qubit / root
    inline int refcount(const int q) const {
        minorminer_assert(data.count(q) == 1);
        return fetch(q).second;
    }

    // assumes `this` and `other` have links for eachother's labels
    // steals all qubits from `other` which are available to be taken
    // by `this`; starting with the qubit links and updating qubit
    // links after all
    template <typename embedding_problem_t>
    inline void steal(chain &other, embedding_problem_t &ep, int chainsize = 1) {
        int q = drop_link(other.label);
        int p = other.drop_link(label);

        minorminer_assert(q != -1);
        minorminer_assert(p != -1);
        while (other.size() > chainsize && ep.accepts_qubit(label, p)) {
            int r = other.trim_leaf(p);
            if (r == p) break;
            if (!count(p))
                add_leaf(p, q);
            else if (p != q)
                trim_branch(q);
            q = p;
            p = r;
        }
        set_link(other.label, q);
        other.set_link(label, p);
        DIAGNOSE("steal");
    }

    class iterator {
      public:
        iterator(typename decltype(data)::const_iterator it) : it(it) {}
        iterator operator++() { return ++it; }
        bool operator!=(const iterator &other) { return it != other.it; }
        const typename decltype(data)::key_type &operator*() const { return it->first; }  // Return key part of map

      private:
        typename decltype(data)::const_iterator it;
    };

    iterator begin() const { return iterator(data.begin()); }

    iterator end() const { return iterator(data.end()); };

    inline void diagnostic(char *last_op) {
        int r = run_diagnostic();

        if (r) {
            std::cout << "chain diagnostic failures on var " << label << ":";
            if (r & 1) std::cout << " (parent containment)";
            if (r & 2) std::cout << " (refcount)";

            std::cout << ".  last operation was " << last_op << std::endl;
            throw - 1;
        }
    }

    inline int run_diagnostic() const {
        int errorcode = 0;
        unordered_map<int, int> refs;
        for (auto &x_q : links) {
            refs[x_q.second]++;
        }
        for (auto &q_pr : data) {
            refs[q_pr.second.first]++;
        }
        for (auto &p_r : refs) {
            int p = p_r.first;
            int r = p_r.second;
            if (count(p) == 0)
                errorcode |= 1;
            else if (r != data.at(p).second)
                errorcode |= 2;
        }
        return errorcode;
    }

  private:
    // const unsafe data accessor
    inline const pair<int, int> &fetch(int q) const { return (*data.find(q)).second; }

    // non-const unsafe data accessor
    inline pair<int, int> &retrieve(int q) { return (*data.find(q)).second; }
};
}
