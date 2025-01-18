constexpr static float M_SQRT1_2 = .707106781186547524401;
template<class Tdata>
static __device__ void gelu(
    Tdata *__restrict__ data) {
    auto i = blockIdx.x * blockDim.x + threadIdx.x;
    auto x = float(data[i]);
    data[i] = Tdata(0.5f * x * (1.0f + erf(x * M_SQRT1_2)));
}