

template<int ArrSize, typename ArrayType>
struct ArrayStruct {
    ArrayType a[ArrSize];
};

// 各个元素分别代表：[grid_idx, block_idx, grid 的stride相对于block的倍数，总的len限制]
template<typename ElementType>
struct Constrains {
    ElementType grid_idx;
    ElementType block_idx;
    ElementType grid_div_block;
    ElementType total_len;
};
