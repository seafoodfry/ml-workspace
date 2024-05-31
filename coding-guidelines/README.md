# Coding Guidelines

## Shadowing

A simple and bad bug to create:

```cpp
class BadLattice {
    public:
        BadLattice(unsigned int xDim, unsigned int yDim);

    private:
        unsigned int xDim;
        unsigned int yDim;
};

BadLattice::BadLattice(unsigned int xDim, unsigned int yDim) {
    xDim = xDim;
    yDim = yDim;
}
```

The way to catch this is with
[CFLAG -Wshadow](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#index-Wshadow).