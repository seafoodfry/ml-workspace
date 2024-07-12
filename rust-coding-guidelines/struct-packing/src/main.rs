// Potentially inefficient layout
struct Inefficient {
    a: u8,
    b: u32,
    c: u16,
}

// More efficient layout
struct Efficient {
    b: u32,
    c: u16,
    a: u8,
}

fn main() {
    println!(
        "Size of Inefficient: {}, Alignment: {}",
        std::mem::size_of::<Inefficient>(),
        std::mem::align_of::<Inefficient>()
    );
    println!(
        "Size of Efficient: {}, Alignment: {}",
        std::mem::size_of::<Efficient>(),
        std::mem::align_of::<Efficient>()
    );
}
