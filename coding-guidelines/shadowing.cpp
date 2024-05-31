#include <iostream>

/*
    Bad lattice constructor implementation.
*/
class BadLattice {
    public:
        explicit BadLattice(unsigned int xDim, unsigned int yDim);
        ~BadLattice() = default;

        unsigned int x() const;
        unsigned int y() const;

    private:
        unsigned int xDim;
        unsigned int yDim;
};

BadLattice::BadLattice(unsigned int xDim, unsigned int yDim) {
    xDim = xDim;
    yDim = yDim;
}

unsigned int BadLattice::x() const { return xDim; }

unsigned int BadLattice::y() const { return yDim; }

/*
    Good lattice construction implementation.
*/
class Lattice {
    public:
        explicit Lattice(unsigned int xDim, unsigned int yDim);
        ~Lattice() = default;

        unsigned int x() const;
        unsigned int y() const;

    private:
        unsigned int xDim;
        unsigned int yDim;
};

Lattice::Lattice(unsigned int x, unsigned int y) : xDim(x), yDim(y) {}

unsigned int Lattice::x() const { return xDim; }

unsigned int Lattice::y() const { return yDim; }


int main(int argc, const char** argv) {
    BadLattice* badLattice = new BadLattice(3, 3);
    std::cout << badLattice->x() << " " << badLattice->y() << std::endl;

    Lattice* lattice = new Lattice(3, 3);
    std::cout << lattice->x() << " " << lattice->y() << std::endl;
    return 0;
}